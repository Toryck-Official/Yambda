#!/usr/bin/env python3
"""Train simple learned embedding and autoregressive direct-SID predictors."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_ARRAYS = ROOT / "phase1_gate1b" / "work" / "learning_arrays"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "artifacts" / "predictor_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class EmbeddingPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = ContextEncoder(input_dim, hidden_dim, dropout)
        self.output = nn.Linear(hidden_dim, 128)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(self.encoder(context)), dim=-1)


class AutoregressiveSIDPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = ContextEncoder(input_dim, hidden_dim, dropout)
        token_dim = 32
        self.token_embeddings = nn.ModuleList([nn.Embedding(256, token_dim) for _ in range(3)])
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim + token_dim * level, 256) for level in range(4)]
        )

    def teacher_forcing(self, context: torch.Tensor, truth: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.encoder(context)
        prefixes: list[torch.Tensor] = []
        logits = []
        for level in range(4):
            features = torch.cat([hidden, *prefixes], dim=1) if prefixes else hidden
            logits.append(self.heads[level](features))
            if level < 3:
                prefixes.append(self.token_embeddings[level](truth[:, level]))
        return logits

    def autoregressive(self, context: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Inference has no truth-prefix argument by construction."""
        hidden = self.encoder(context)
        prefixes: list[torch.Tensor] = []
        predictions = []
        logits = []
        for level in range(4):
            features = torch.cat([hidden, *prefixes], dim=1) if prefixes else hidden
            level_logits = self.heads[level](features)
            level_prediction = level_logits.argmax(dim=1)
            logits.append(level_logits)
            predictions.append(level_prediction)
            if level < 3:
                prefixes.append(self.token_embeddings[level](level_prediction))
        return torch.stack(predictions, dim=1), logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    order = indices.copy()
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def tensor_rows(array: np.ndarray, indices: np.ndarray, device: torch.device, dtype=None):
    value = np.asarray(array[indices])
    tensor = torch.from_numpy(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device, non_blocking=False)


@torch.no_grad()
def embedding_validation(
    model: EmbeddingPredictor, context: np.ndarray, truth: np.ndarray,
    indices: np.ndarray, batch_size: int, device: torch.device,
) -> tuple[float, float]:
    model.eval()
    cosine_sum = 0.0
    mse_sum = 0.0
    count = 0
    for index in batches(indices, batch_size):
        x = tensor_rows(context, index, device, torch.float32)
        y = tensor_rows(truth, index, device, torch.float32)
        pred = model(x)
        cosine_sum += float((pred * y).sum(dim=1).sum())
        mse_sum += float(F.mse_loss(pred, y, reduction="sum"))
        count += len(index)
    return cosine_sum / count, mse_sum / (count * 128)


@torch.no_grad()
def sid_validation(
    model: AutoregressiveSIDPredictor, context: np.ndarray, codes: np.ndarray,
    indices: np.ndarray, batch_size: int, device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    predictions = []
    for index in batches(indices, batch_size):
        x = tensor_rows(context, index, device, torch.float32)
        y = tensor_rows(codes, index, device, torch.long)
        pred, logits = model.autoregressive(x)
        loss_sum += sum(float(F.cross_entropy(logit, y[:, level], reduction="sum")) for level, logit in enumerate(logits))
        predictions.append(pred.cpu().numpy().astype(np.uint16))
    return loss_sum / (len(indices) * 4), np.concatenate(predictions)


def train_embedding(
    seed: int, config: dict, context: np.ndarray, truth: np.ndarray,
    train: np.ndarray, validation: np.ndarray, device: torch.device,
) -> tuple[EmbeddingPredictor, list[dict]]:
    learning = config["learning"]
    model = EmbeddingPredictor(context.shape[1], int(learning["hidden_dim"]), float(learning["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning["learning_rate"]), weight_decay=float(learning["weight_decay"]))
    rng = np.random.default_rng(seed + 101)
    best = -np.inf
    best_state = None
    patience = 0
    trace = []
    for epoch in range(1, int(learning["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for index in batches(train, int(learning["batch_size"]), rng):
            x = tensor_rows(context, index, device, torch.float32)
            y = tensor_rows(truth, index, device, torch.float32)
            pred = model(x)
            loss = (1.0 - (pred * y).sum(dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            seen += len(index)
        val_cos, val_mse = embedding_validation(model, context, truth, validation, int(learning["batch_size"]), device)
        trace.append({"epoch": epoch, "train_cosine_loss": loss_sum / seen, "validation_cosine": val_cos, "validation_mse": val_mse})
        print(f"seed={seed} embedding epoch={epoch} val_cos={val_cos:.6f}", flush=True)
        if val_cos > best + 1e-6:
            best = val_cos
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(learning["patience"]):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, trace


def train_sid(
    seed: int, config: dict, context: np.ndarray, codes: np.ndarray,
    train: np.ndarray, validation: np.ndarray, device: torch.device,
) -> tuple[AutoregressiveSIDPredictor, list[dict]]:
    learning = config["learning"]
    model = AutoregressiveSIDPredictor(context.shape[1], int(learning["hidden_dim"]), float(learning["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning["learning_rate"]), weight_decay=float(learning["weight_decay"]))
    rng = np.random.default_rng(seed + 202)
    best = np.inf
    best_state = None
    patience = 0
    trace = []
    for epoch in range(1, int(learning["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for index in batches(train, int(learning["batch_size"]), rng):
            x = tensor_rows(context, index, device, torch.float32)
            y = tensor_rows(codes, index, device, torch.long)
            logits = model.teacher_forcing(x, y)
            loss = sum(F.cross_entropy(logit, y[:, level]) for level, logit in enumerate(logits)) / 4.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            seen += len(index)
        val_loss, val_pred = sid_validation(model, context, codes, validation, int(learning["batch_size"]), device)
        true = np.asarray(codes[validation])
        equal = val_pred == true
        prefix = np.logical_and.accumulate(equal, axis=1)
        trace.append({"epoch": epoch, "train_cross_entropy": loss_sum / seen, "validation_autoregressive_cross_entropy": val_loss,
                      "validation_prefix_at_1": float(prefix[:, 0].mean()), "validation_prefix_at_4": float(prefix[:, 3].mean())})
        print(f"seed={seed} direct_sid epoch={epoch} val_ce={val_loss:.6f} p1={prefix[:,0].mean():.4f} p4={prefix[:,3].mean():.4f}", flush=True)
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(learning["patience"]):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, trace


@torch.no_grad()
def predict_embedding(model, context, indices, batch_size, device):
    model.eval()
    output = []
    for index in batches(indices, batch_size):
        output.append(model(tensor_rows(context, index, device, torch.float32)).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    learning = config["learning"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = np.load(args.arrays / "context.float16.npy", mmap_mode="r")
    truth = np.load(args.arrays / "truth.float32.npy", mmap_mode="r")
    codes = np.load(args.arrays / "true_codes.uint16.npy", mmap_mode="r")
    split = np.load(args.arrays / "split.npy")
    train, validation, test = (np.flatnonzero(split == value) for value in (0, 1, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate 1B predictor training expects the currently enabled GPU")
    for raw_seed in learning["seeds"]:
        seed = int(raw_seed)
        seed_everything(seed)
        run_dir = args.output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        embed_model, embed_trace = train_embedding(seed, config, context, truth, train, validation, device)
        embed_prediction = predict_embedding(embed_model, context, test, int(learning["batch_size"]), device)
        torch.save({"state_dict": embed_model.state_dict(), "input_dim": context.shape[1], "hidden_dim": int(learning["hidden_dim"]), "dropout": float(learning["dropout"])}, run_dir / "embedding_predictor.pt")
        np.save(run_dir / "embedding_test_prediction.float32.npy", embed_prediction, allow_pickle=False)
        (run_dir / "embedding_trace.json").write_text(json.dumps(embed_trace, indent=2) + "\n")
        del embed_model, embed_prediction
        torch.cuda.empty_cache()
        seed_everything(seed)
        sid_model, sid_trace = train_sid(seed, config, context, codes, train, validation, device)
        _, sid_prediction = sid_validation(sid_model, context, codes, test, int(learning["batch_size"]), device)
        torch.save({"state_dict": sid_model.state_dict(), "input_dim": context.shape[1], "hidden_dim": int(learning["hidden_dim"]), "dropout": float(learning["dropout"])}, run_dir / "direct_sid_predictor.pt")
        np.save(run_dir / "direct_sid_test_prediction.uint16.npy", sid_prediction, allow_pickle=False)
        (run_dir / "direct_sid_trace.json").write_text(json.dumps(sid_trace, indent=2) + "\n")
        print(f"completed seed {seed}", flush=True)


if __name__ == "__main__":
    main()

