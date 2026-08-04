"""Training loop for the explicit-feedback supervised HPN baseline."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from snmpp.recommendation.config import RecommendationExperimentConfig
from snmpp.recommendation.data import (
    ExplicitRecommendationDataset,
    RQCodebookStore,
    collate_recommendation_groups,
    load_recommendation_manifest,
    move_batch_to_device,
    validate_codebook_store_against_manifest,
)
from snmpp.recommendation.model import (
    ExplicitFeedbackHPN,
    multi_positive_candidate_loss,
    multi_positive_hpn_loss,
)
from snmpp.recommendation.provenance import (
    recommendation_source_fingerprints,
    runtime_environment,
)
from snmpp.recommendation.sampling import RecommendationCandidateSampler
from snmpp.utils import (
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_json,
    resolve_device,
    seed_everything,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_recommendation_model(
    config: RecommendationExperimentConfig,
    store: RQCodebookStore,
) -> ExplicitFeedbackHPN:
    model = config.model
    allowed_levels = {store.semantic_levels, store.sid_levels}
    if model.sid_levels not in allowed_levels:
        raise ValueError(
            "model must predict either the semantic prefix or the legacy full path; "
            f"semantic={store.semantic_levels}, full={store.sid_levels}, "
            f"requested={model.sid_levels}"
        )
    return ExplicitFeedbackHPN(
        item_dim=store.item_dim,
        d_model=model.d_model,
        max_history=config.data.history_len,
        num_event_types=model.num_event_types,
        include_source=model.include_source,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        dropout=model.dropout,
        sid_levels=model.sid_levels,
        sid_vocab_size=model.sid_vocab_size,
        sid_temperature=model.sid_temperature,
        path_conditioning=model.path_conditioning,
        time_scale_seconds=model.time_scale_seconds,
        state_pooling=model.state_pooling,
    )


def recommendation_method_name(
    config: RecommendationExperimentConfig,
    store: RQCodebookStore,
) -> str:
    if config.model.sid_levels == store.semantic_levels < store.sid_levels:
        if config.model.path_conditioning == "prefix_autoregressive":
            if config.model.num_event_types == 5:
                if config.model.include_source:
                    return "supervised_listen_history_source_autoregressive_semantic_hpn"
                return "supervised_listen_history_autoregressive_semantic_hpn"
            return "supervised_explicit_feedback_autoregressive_semantic_hpn"
        return "supervised_explicit_feedback_semantic_hpn"
    return "supervised_explicit_feedback_hpn_legacy_joint_path"


def _split_max_rows(config: RecommendationExperimentConfig, split: str) -> int | None:
    return {
        "train": config.data.max_train_rows,
        "validation": config.data.max_validation_rows,
        "test": config.data.max_test_rows,
    }[split]


def make_recommendation_loader(
    config: RecommendationExperimentConfig,
    store: RQCodebookStore,
    split: str,
    *,
    epoch: int,
    shuffle: bool,
    batch_size: int | None = None,
    max_rows: int | None = None,
) -> tuple[DataLoader, int | None]:
    dataset = ExplicitRecommendationDataset(
        config.data.dataset_dir,
        split,
        seed=config.training.seed,
        epoch=epoch,
        shuffle=shuffle,
        shuffle_buffer_size=config.data.shuffle_buffer_size if shuffle else 0,
        max_rows=_split_max_rows(config, split) if max_rows is None else max_rows,
        limited_row_selection=config.data.limited_row_selection,
        row_selection_seed=config.data.row_selection_seed,
    )
    size = dataset.expected_rows
    effective_batch = int(batch_size or config.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        num_workers=config.data.num_workers,
        collate_fn=lambda rows: collate_recommendation_groups(rows, store),
        pin_memory=torch.cuda.is_available(),
    )
    total = math.ceil(size / effective_batch) if size is not None else None
    return loader, total


def run_recommendation_epoch(
    *,
    model: ExplicitFeedbackHPN,
    loader: DataLoader,
    total_batches: int | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    sampler: RecommendationCandidateSampler | None,
    token_loss_weight: float,
    candidate_loss_weight: float,
    sampled_negatives: int,
    gradient_clip_norm: float,
    description: str,
    log_every_steps: int,
    progress_path: Path | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    progress = tqdm(
        loader,
        total=total_batches,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    )
    for step, cpu_batch in enumerate(progress, start=1):
        batch = move_batch_to_device(cpu_batch, device)
        batch_size = int(batch["uid"].shape[0])
        with torch.set_grad_enabled(training):
            output = model(batch)
            token = multi_positive_hpn_loss(
                model,
                output,
                batch["positive_sids"],
                batch["positive_mask"],
            )
            combined = float(token_loss_weight) * token["loss"]
            values: dict[str, torch.Tensor] = {
                "token_loss": token["loss"],
                "predicted_path_hit": token["predicted_path_hit"],
                "unique_positive_paths": token["unique_positive_paths"],
            }
            for name, value in token.items():
                if name.startswith("token_hit_l"):
                    values[name] = value

            if candidate_loss_weight > 0:
                if sampler is None:
                    raise RuntimeError("candidate loss requires a candidate sampler")
                sid_logits = output["sid_logits"]
                if not isinstance(sid_logits, list):
                    raise TypeError("sid_logits must be a list")
                negative_sids = sampler.sample_training_batch(
                    batch["positive_sids"],
                    batch["positive_mask"],
                    int(sampled_negatives),
                    device,
                )
                candidate = multi_positive_candidate_loss(
                    model,
                    output,
                    batch["positive_sids"],
                    batch["positive_mask"],
                    negative_sids,
                )
                combined = combined + float(candidate_loss_weight) * candidate["loss"]
                values["candidate_loss"] = candidate["loss"]
                values["sampled_hit_at_1"] = candidate["sampled_hit_at_1"]
                values["sampled_mrr"] = candidate["sampled_mrr"]
            else:
                zero = combined.detach().new_tensor(0.0)
                values["candidate_loss"] = zero
                values["sampled_hit_at_1"] = zero
                values["sampled_mrr"] = zero
            values["loss"] = combined
            if not bool(torch.isfinite(combined)):
                raise FloatingPointError(f"non-finite recommendation loss in {description}")

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                combined.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(gradient_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError(f"non-finite model parameter after update: {name}")

        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        examples += batch_size
        if step == 1 or step % max(int(log_every_steps), 1) == 0:
            progress.set_postfix(
                loss=f"{float(combined.detach().cpu()):.4f}",
                token=f"{float(token['loss'].detach().cpu()):.4f}",
                examples=examples,
                refresh=False,
            )
            if progress_path is not None:
                atomic_write_json(
                    progress_path,
                    {
                        "phase": description,
                        "batch": int(step),
                        "total_batches": int(total_batches or 0),
                        "percent": (
                            round(100.0 * step / total_batches, 2)
                            if total_batches
                            else None
                        ),
                        "updated_unix_seconds": time.time(),
                    },
                )
    if examples == 0:
        raise RuntimeError(f"{description} produced no recommendation rows")
    return {
        **{name: value / examples for name, value in totals.items()},
        "examples": examples,
    }


def _checkpoint_payload(
    model: ExplicitFeedbackHPN,
    config: RecommendationExperimentConfig,
    store: RQCodebookStore,
    manifest: dict[str, Any],
    epoch: int,
    validation_metrics: dict[str, float],
    dataset_manifest_sha256: str,
    source_fingerprints: dict[str, str],
    method_name: str,
    row_selection: dict[str, Any],
    training_item_counts_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "method": method_name,
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "item_dim": store.item_dim,
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
        "dataset_source_sha256": manifest["created_from"]["source_sha256"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "codebook_catalog_fingerprints": manifest["mapping"]["codebook_catalog_fingerprints"],
        "source_fingerprints": source_fingerprints,
        "row_selection": row_selection,
        "training_item_counts_sha256": training_item_counts_sha256,
        "data_contract": manifest["data_contract"],
    }


def _row_selection_metadata(
    config: RecommendationExperimentConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        dataset = ExplicitRecommendationDataset(
            config.data.dataset_dir,
            split,
            seed=config.training.seed,
            epoch=0,
            shuffle=False,
            max_rows=_split_max_rows(config, split),
            limited_row_selection=config.data.limited_row_selection,
            row_selection_seed=config.data.row_selection_seed,
        )
        output[split] = dataset.selection_metadata
    return output


def _write_matched_training_item_counts(
    config: RecommendationExperimentConfig,
    run_dir: Path,
) -> tuple[Path, str]:
    source = config.data.dataset_dir / "train_item_counts.npz"
    if config.data.max_train_rows is None:
        return source, sha256_file(source)
    if config.data.limited_row_selection != "user_stratified_hash":
        # Legacy head-limited training changes its shard order each epoch, so no
        # single subset count file could honestly be called matched.
        return source, sha256_file(source)
    dataset = ExplicitRecommendationDataset(
        config.data.dataset_dir,
        "train",
        seed=config.training.seed,
        epoch=0,
        shuffle=False,
        max_rows=config.data.max_train_rows,
        limited_row_selection=config.data.limited_row_selection,
        row_selection_seed=config.data.row_selection_seed,
    )
    counts: Counter[int] = Counter()
    observed_rows = 0
    for row in dataset:
        counts.update(int(value) for value in row["positive_dense_item_ids"])
        observed_rows += 1
    if observed_rows != dataset.expected_rows:
        raise RuntimeError(
            f"matched popularity observed {observed_rows} rows, expected {dataset.expected_rows}"
        )
    dense_ids = np.asarray(sorted(counts), dtype=np.int64)
    values = np.asarray([counts[int(dense)] for dense in dense_ids], dtype=np.int64)
    destination = run_dir / "training_subset_item_counts.npz"
    atomic_save_npz(destination, dense_item_id=dense_ids, count=values)
    return destination, sha256_file(destination)


def train_recommendation_model(
    config: RecommendationExperimentConfig,
    *,
    overwrite_run: bool = False,
) -> dict[str, Any]:
    """Train with deterministic validation candidate samples and early stopping."""

    config.validate()
    seed_everything(
        config.training.seed,
        deterministic_algorithms=config.training.deterministic_algorithms,
    )
    device = resolve_device(config.training.device)
    store = RQCodebookStore(config.data.feature_store_dir)
    manifest = load_recommendation_manifest(config.data.dataset_dir)
    validate_codebook_store_against_manifest(store, manifest, config.data.dataset_dir)
    observed_vocabulary = int(manifest["mapping"]["sid_vocab_size_observed"])
    if observed_vocabulary > config.model.sid_vocab_size:
        raise ValueError(
            f"prepared codebook needs {observed_vocabulary} SID tokens, "
            f"model provides {config.model.sid_vocab_size}"
        )
    prepared_history = int(manifest.get("prepare_config", {}).get("history_len", -1))
    if prepared_history != config.data.history_len:
        raise ValueError(
            f"prepared history_len={prepared_history}, config history_len={config.data.history_len}"
        )
    model = build_recommendation_model(config, store).to(device)
    method_name = recommendation_method_name(config, store)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    run_dir = config.training.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = [
        path.name for path in (run_dir / "best.pt", run_dir / "last.pt") if path.exists()
    ]
    if existing_checkpoints and not overwrite_run:
        raise FileExistsError(
            f"Run directory already has checkpoints {existing_checkpoints}; "
            "pass --overwrite-run only after auditing that run"
        )
    manifest_path = config.data.dataset_dir / "manifest.json"
    dataset_manifest_sha256 = sha256_file(manifest_path)
    source_fingerprints = recommendation_source_fingerprints(PROJECT_ROOT)
    atomic_write_json(run_dir / "config.json", config.to_dict())
    atomic_write_json(run_dir / "environment.json", runtime_environment(device))
    atomic_write_json(run_dir / "prepared_data_manifest.json", manifest)
    atomic_write_json(run_dir / "source_fingerprints.json", source_fingerprints)
    row_selection = _row_selection_metadata(config)
    atomic_write_json(run_dir / "row_selection.json", row_selection)
    training_item_counts_path, training_item_counts_sha256 = _write_matched_training_item_counts(
        config, run_dir
    )
    atomic_write_json(
        run_dir / "training_item_counts.json",
        {
            "path": str(training_item_counts_path.resolve()),
            "sha256": training_item_counts_sha256,
            "matched_to_bounded_training_rows": (
                config.data.max_train_rows is None
                or config.data.limited_row_selection == "user_stratified_hash"
            ),
        },
    )

    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.training.epochs + 1):
        train_loader, train_total = make_recommendation_loader(
            config,
            store,
            "train",
            epoch=epoch,
            shuffle=True,
        )
        validation_loader, validation_total = make_recommendation_loader(
            config,
            store,
            "validation",
            epoch=0,
            shuffle=False,
        )
        train_sampler = RecommendationCandidateSampler(
            config.data.dataset_dir,
            store,
            seed=config.training.seed + epoch,
            mode=config.training.negative_mode,
            prefix_levels=config.training.semantic_prefix_levels,
            path_levels=config.model.sid_levels,
        )
        validation_sampler = RecommendationCandidateSampler(
            config.data.dataset_dir,
            store,
            seed=config.training.seed + 900_001,
            mode=config.training.negative_mode,
            prefix_levels=config.training.semantic_prefix_levels,
            path_levels=config.model.sid_levels,
        )
        train_metrics = run_recommendation_epoch(
            model=model,
            loader=train_loader,
            total_batches=train_total,
            device=device,
            optimizer=optimizer,
            sampler=train_sampler,
            token_loss_weight=config.training.token_loss_weight,
            candidate_loss_weight=config.training.candidate_loss_weight,
            sampled_negatives=config.training.sampled_negatives,
            gradient_clip_norm=config.training.gradient_clip_norm,
            description=f"[nolisten HPN {epoch}/{config.training.epochs} train]",
            log_every_steps=config.training.log_every_steps,
            progress_path=run_dir / "progress.json",
        )
        validation_metrics = run_recommendation_epoch(
            model=model,
            loader=validation_loader,
            total_batches=validation_total,
            device=device,
            optimizer=None,
            sampler=validation_sampler,
            token_loss_weight=config.training.token_loss_weight,
            candidate_loss_weight=config.training.candidate_loss_weight,
            sampled_negatives=config.training.sampled_negatives,
            gradient_clip_norm=config.training.gradient_clip_norm,
            description=f"[nolisten HPN {epoch}/{config.training.epochs} validation]",
            log_every_steps=config.training.log_every_steps,
            progress_path=run_dir / "progress.json",
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        atomic_write_json(run_dir / "metrics.json", history)
        atomic_torch_save(
            run_dir / "last.pt",
            _checkpoint_payload(
                model,
                config,
                store,
                manifest,
                epoch,
                validation_metrics,
                dataset_manifest_sha256,
                source_fingerprints,
                method_name,
                row_selection,
                training_item_counts_sha256,
            ),
        )
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_validation - config.training.min_improvement:
            best_validation = validation_loss
            stale_epochs = 0
            atomic_torch_save(
                run_dir / "best.pt",
                _checkpoint_payload(
                    model,
                    config,
                    store,
                    manifest,
                    epoch,
                    validation_metrics,
                    dataset_manifest_sha256,
                    source_fingerprints,
                    method_name,
                    row_selection,
                    training_item_counts_sha256,
                ),
            )
        else:
            stale_epochs += 1
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_loss,
                "best_validation_loss": best_validation,
                "stale_epochs": stale_epochs,
            }
        )
        if stale_epochs >= config.training.early_stopping_patience:
            break

    result = {
        "method": method_name,
        "device": str(device),
        "best_validation_loss": best_validation,
        "epochs_completed": len(history),
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
        "last_checkpoint": str((run_dir / "last.pt").resolve()),
        "history": history,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_fingerprints": source_fingerprints,
        "row_selection": row_selection,
        "training_item_counts_path": str(training_item_counts_path.resolve()),
        "training_item_counts_sha256": training_item_counts_sha256,
    }
    atomic_write_json(run_dir / "training_summary.json", result)
    return result


def load_recommendation_checkpoint(
    checkpoint_path: str | Path,
    config: RecommendationExperimentConfig,
    store: RQCodebookStore,
    device: torch.device,
    manifest: dict[str, Any] | None = None,
) -> tuple[ExplicitFeedbackHPN, dict[str, Any]]:
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format_version") != 2:
        raise ValueError("checkpoint must use recommendation format_version=2")
    if checkpoint.get("method") not in {
        "supervised_explicit_feedback_hpn",
        "supervised_explicit_feedback_hpn_legacy_joint_path",
        "supervised_explicit_feedback_semantic_hpn",
        "supervised_explicit_feedback_autoregressive_semantic_hpn",
        "supervised_listen_history_autoregressive_semantic_hpn",
        "supervised_listen_history_source_autoregressive_semantic_hpn",
    }:
        raise ValueError("checkpoint is not a supervised HPN")
    stored_model = checkpoint.get("config", {}).get("model")
    if isinstance(stored_model, dict) and "path_conditioning" not in stored_model:
        stored_model = {**stored_model, "path_conditioning": "expected_residual"}
    if stored_model != config.to_dict()["model"]:
        raise ValueError("checkpoint model configuration differs from evaluation config")
    stored_history = checkpoint.get("config", {}).get("data", {}).get("history_len")
    if stored_history != config.data.history_len:
        raise ValueError("checkpoint history length differs from evaluation config")
    if manifest is not None:
        expected_fingerprints = manifest["mapping"]["codebook_catalog_fingerprints"]
        if checkpoint.get("codebook_catalog_fingerprints") != expected_fingerprints:
            raise ValueError("checkpoint and evaluation codebook fingerprints differ")
        manifest_sha256 = sha256_file(config.data.dataset_dir / "manifest.json")
        if checkpoint.get("dataset_manifest_sha256") != manifest_sha256:
            raise ValueError("checkpoint and evaluation dataset manifests differ")
    model = build_recommendation_model(config, store).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
