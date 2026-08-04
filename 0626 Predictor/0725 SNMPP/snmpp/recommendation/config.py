"""Strict configuration for the no-listen supervised recommendation baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

T = TypeVar("T")


def _strict_construct(cls: type[T], values: Mapping[str, Any]) -> T:
    allowed = {field.name for field in fields(cast(Any, cls))}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(values))


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class RecommendationDataConfig:
    """Prepared grouped sequences and fixed RQ codebook resources."""

    dataset_dir: Path
    feature_store_dir: Path
    history_len: int = 50
    num_workers: int = 0
    shuffle_buffer_size: int = 8192
    max_train_rows: int | None = None
    max_validation_rows: int | None = None
    max_test_rows: int | None = None
    limited_row_selection: str = "head"
    row_selection_seed: int = 2026

    def validate(self) -> None:
        _positive("history_len", self.history_len)
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size cannot be negative")
        for name in ("max_train_rows", "max_validation_rows", "max_test_rows"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.limited_row_selection not in {"head", "user_stratified_hash"}:
            raise ValueError("limited_row_selection must be head or user_stratified_hash")
        if self.row_selection_seed < 0:
            raise ValueError("row_selection_seed cannot be negative")


@dataclass(frozen=True)
class RecommendationModelConfig:
    """Explicit-feedback history encoder and hierarchical policy network.

    ``sid_levels`` is the number of levels predicted by the network.  For the
    high-coverage catalog it must normally equal the four semantic RQ levels;
    the fifth identity suffix is then resolved separately within a collided
    semantic bucket.  A value equal to the store's full path length remains
    supported only for auditing the earlier joint-head pilot.
    """

    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    num_event_types: int = 4
    include_source: bool = False
    sid_levels: int = 4
    sid_vocab_size: int = 256
    sid_temperature: float = 1.0
    path_conditioning: str = "expected_residual"
    time_scale_seconds: float = 3600.0
    state_pooling: str = "last_mean"

    def validate(self) -> None:
        _positive("d_model", self.d_model)
        _positive("num_layers", self.num_layers)
        _positive("num_heads", self.num_heads)
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_event_types not in {4, 5}:
            raise ValueError(
                "num_event_types must be 4 (explicit marks only) or 5 (explicit + listen)"
            )
        _positive("sid_levels", self.sid_levels)
        _positive("sid_vocab_size", self.sid_vocab_size)
        _positive("sid_temperature", self.sid_temperature)
        if self.path_conditioning not in {
            "expected_residual",
            "prefix_autoregressive",
        }:
            raise ValueError("path_conditioning must be expected_residual or prefix_autoregressive")
        _positive("time_scale_seconds", self.time_scale_seconds)
        if self.state_pooling not in {"last", "mean", "last_mean"}:
            raise ValueError("state_pooling must be last, mean, or last_mean")


@dataclass(frozen=True)
class RecommendationTrainingConfig:
    """Optimization and sampled candidate-ranking settings."""

    run_dir: Path
    seed: int = 2026
    device: str = "auto"
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    token_loss_weight: float = 1.0
    candidate_loss_weight: float = 1.0
    sampled_negatives: int = 63
    negative_mode: str = "semantic_prefix"
    semantic_prefix_levels: tuple[int, ...] = (2, 1)
    early_stopping_patience: int = 3
    min_improvement: float = 1.0e-5
    log_every_steps: int = 100
    deterministic_algorithms: bool = True

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        _positive("epochs", self.epochs)
        _positive("batch_size", self.batch_size)
        _positive("learning_rate", self.learning_rate)
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        _positive("gradient_clip_norm", self.gradient_clip_norm)
        if self.token_loss_weight < 0 or self.candidate_loss_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.token_loss_weight == 0 and self.candidate_loss_weight == 0:
            raise ValueError("at least one supervised loss must be active")
        if self.candidate_loss_weight > 0:
            _positive("sampled_negatives", self.sampled_negatives)
            if self.negative_mode not in {"random", "semantic_prefix"}:
                raise ValueError("negative_mode must be random or semantic_prefix")
        if not self.semantic_prefix_levels:
            raise ValueError("semantic_prefix_levels cannot be empty")
        if any(value <= 0 for value in self.semantic_prefix_levels):
            raise ValueError("semantic_prefix_levels must contain positive integers")
        _positive("early_stopping_patience", self.early_stopping_patience)
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative")
        _positive("log_every_steps", self.log_every_steps)


@dataclass(frozen=True)
class RecommendationEvaluationConfig:
    """Group-aware ranking evaluation."""

    split: str = "test"
    candidate_mode: str = "sampled"
    sampled_negatives: int = 100
    top_k: tuple[int, ...] = (5, 10, 20)
    batch_size: int = 128
    seed: int = 2026
    max_rows: int | None = 10000
    exclude_history_items: bool = True

    def validate(self) -> None:
        if self.split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")
        if self.candidate_mode != "sampled":
            raise ValueError("only the explicitly labelled sampled protocol is currently supported")
        _positive("sampled_negatives", self.sampled_negatives)
        if not self.top_k or any(value <= 0 for value in self.top_k):
            raise ValueError("top_k must contain positive integers")
        _positive("batch_size", self.batch_size)
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.max_rows is not None:
            _positive("max_rows", self.max_rows)


@dataclass(frozen=True)
class RecommendationExperimentConfig:
    """Complete no-listen recommendation experiment."""

    data: RecommendationDataConfig
    model: RecommendationModelConfig
    training: RecommendationTrainingConfig
    evaluation: RecommendationEvaluationConfig

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        if self.data.num_workers > 0 and any(
            value is not None
            for value in (
                self.data.max_train_rows,
                self.data.max_validation_rows,
                self.data.max_test_rows,
            )
        ):
            raise ValueError(
                "exact max-row limits currently require num_workers=0; "
                "multiple iterable workers would each apply the limit"
            )
        if self.data.history_len <= 0:
            raise ValueError("history_len must be positive")
        if any(level > self.model.sid_levels for level in self.training.semantic_prefix_levels):
            raise ValueError("semantic prefix levels cannot exceed sid_levels")

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["data"]["dataset_dir"] = str(self.data.dataset_dir)
        output["data"]["feature_store_dir"] = str(self.data.feature_store_dir)
        output["training"]["run_dir"] = str(self.training.run_dir)
        return output


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def load_recommendation_config(path: str | Path) -> RecommendationExperimentConfig:
    """Load YAML and reject silent field misspellings."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("recommendation config root must be a mapping")
    expected = {"data", "model", "training", "evaluation"}
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")

    data_values = dict(raw["data"])
    data_values["dataset_dir"] = _resolve_path(data_values["dataset_dir"], config_path)
    data_values["feature_store_dir"] = _resolve_path(data_values["feature_store_dir"], config_path)

    model_values = dict(raw["model"])
    training_values = dict(raw["training"])
    training_values["run_dir"] = _resolve_path(training_values["run_dir"], config_path)
    if "semantic_prefix_levels" in training_values:
        training_values["semantic_prefix_levels"] = tuple(training_values["semantic_prefix_levels"])

    evaluation_values = dict(raw["evaluation"])
    if "top_k" in evaluation_values:
        evaluation_values["top_k"] = tuple(evaluation_values["top_k"])

    config = RecommendationExperimentConfig(
        data=_strict_construct(RecommendationDataConfig, data_values),
        model=_strict_construct(RecommendationModelConfig, model_values),
        training=_strict_construct(RecommendationTrainingConfig, training_values),
        evaluation=_strict_construct(
            RecommendationEvaluationConfig,
            evaluation_values,
        ),
    )
    config.validate()
    return config
