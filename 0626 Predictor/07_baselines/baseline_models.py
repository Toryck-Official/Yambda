from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecSIDPolicy(nn.Module):
    """Plain SASRec-style SID predictor.

    This is intentionally simpler than HPN: it encodes history once, then uses
    independent SID heads. It does not update the state with previous SID token
    expectations, so it is a clean sequence-model baseline.
    """

    def __init__(
        self,
        item_dim: int = 128,
        d_model: int = 128,
        max_seq_len: int = 50,
        n_layer: int = 2,
        n_head: int = 4,
        dropout: float = 0.1,
        sid_levels: int = 4,
        sid_vocab_size: int = 256,
        event_vocab_size: int = 8,
        state_pooling: str = "last_mean",
        use_history_feedback: bool = False,
        use_history_event_type: bool = False,
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.state_pooling = str(state_pooling)
        self.use_history_feedback = bool(use_history_feedback)
        self.use_history_event_type = bool(use_history_event_type)
        if self.state_pooling not in {"last", "mean", "last_mean"}:
            raise ValueError("state_pooling must be one of: last, mean, last_mean")

        self.item_map = nn.Linear(self.item_dim, self.d_model)
        self.feedback_map = nn.Linear(1, self.d_model, bias=False)
        self.event_emb = nn.Embedding(int(event_vocab_size), self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(float(dropout))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_head),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layer))
        self.register_buffer("pos_idx", torch.arange(self.max_seq_len, dtype=torch.long), persistent=False)
        causal = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.bool))
        self.register_buffer("attn_mask_full", ~causal, persistent=False)

        if self.state_pooling == "last_mean":
            self.state_fuse = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
            )

        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, self.sid_vocab_size) for _ in range(self.sid_levels)])

    def _masked_mean_state(self, seq: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch.get("history_mask")
        if mask is None:
            valid = torch.ones(seq.shape[:2], dtype=seq.dtype, device=seq.device)
        else:
            valid = mask.to(seq.device, dtype=seq.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (seq * valid.unsqueeze(-1)).sum(dim=1) / denom

    def _pool_state(self, seq: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        last = seq[:, -1, :]
        if self.state_pooling == "last":
            return last
        mean = self._masked_mean_state(seq, batch)
        if self.state_pooling == "mean":
            return mean
        return self.state_fuse(torch.cat([last, mean], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        hist = batch["history_features"]
        batch_size, hist_len, _ = hist.shape
        pos = self.pos_emb(self.pos_idx[:hist_len]).unsqueeze(0).expand(batch_size, hist_len, -1)
        x = self.item_map(hist) + pos

        if self.use_history_feedback and "history_feedbacks" in batch:
            x = x + self.feedback_map(batch["history_feedbacks"].to(hist.device, dtype=hist.dtype).unsqueeze(-1))
        if self.use_history_event_type and "history_event_type_ids" in batch:
            event_ids = batch["history_event_type_ids"].to(hist.device).long().clamp(min=0, max=self.event_emb.num_embeddings - 1)
            x = x + self.event_emb(event_ids)

        x = self.input_norm(self.drop(x))
        attn_mask = self.attn_mask_full[:hist_len, :hist_len]
        seq = self.encoder(x, mask=attn_mask)
        state = self._pool_state(seq, batch)
        sid_logits = [head(state) for head in self.sid_heads]
        return {"sid_logits": sid_logits, "state_emb": state, "seq_emb": seq}


def sid_ce_loss(sid_logits: list[torch.Tensor], target_sid: torch.Tensor) -> dict[str, torch.Tensor]:
    levels = min(len(sid_logits), int(target_sid.shape[1]))
    losses = []
    full = torch.ones(target_sid.shape[0], dtype=torch.bool, device=target_sid.device)
    out: dict[str, torch.Tensor] = {}
    for level in range(levels):
        logits = sid_logits[level]
        target_l = target_sid[:, level].long().clamp(min=0, max=logits.shape[-1] - 1)
        losses.append(F.cross_entropy(logits, target_l))
        pred_l = logits.argmax(dim=-1)
        correct = pred_l == target_l
        out[f"token_acc_l{level + 1}"] = correct.float().mean().detach()
        full &= correct
    if not losses:
        raise RuntimeError("target_sid has no usable semantic levels.")
    loss = sum(losses) / len(losses)
    out["loss"] = loss
    out["full_path_acc"] = full.float().mean().detach()
    return out


def score_sid_candidates(sid_logits: list[torch.Tensor], candidate_sid: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros(candidate_sid.shape[:2], dtype=torch.float32, device=candidate_sid.device)
    for level, logits in enumerate(sid_logits):
        log_probs = F.log_softmax(logits, dim=-1)
        idx = candidate_sid[:, :, level].long().clamp(min=0, max=logits.shape[-1] - 1)
        scores = scores + torch.gather(log_probs, 1, idx)
    return scores / max(len(sid_logits), 1)

