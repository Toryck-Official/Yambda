from __future__ import annotations

import torch
import torch.nn as nn


class FutureHPNPolicy(nn.Module):
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
        event_vocab_size: int = 6,
        sid_temp: float = 1.0,
        state_pooling: str = "last",
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.sid_temp = float(sid_temp)
        self.state_pooling = str(state_pooling)
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
        self.sid_token_embeds = nn.ModuleList([nn.Embedding(self.sid_vocab_size, self.d_model) for _ in range(self.sid_levels)])
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

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

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hist = batch["history_features"]
        batch_size, hist_len, _ = hist.shape
        pos = self.pos_emb(self.pos_idx[:hist_len]).unsqueeze(0).expand(batch_size, hist_len, -1)
        x = self.item_map(hist) + pos

        feedback = batch.get("history_feedbacks")
        if feedback is not None:
            x = x + self.feedback_map(feedback.to(hist.device, dtype=hist.dtype).unsqueeze(-1))

        event_ids = batch.get("history_event_type_ids")
        if event_ids is not None:
            event_ids = event_ids.to(hist.device).long().clamp(min=0, max=self.event_emb.num_embeddings - 1)
            x = x + self.event_emb(event_ids)

        x = self.input_norm(self.drop(x))
        attn_mask = self.attn_mask_full[:hist_len, :hist_len]
        seq = self.encoder(x, mask=attn_mask)
        return {"seq_emb": seq, "state_emb": self._pool_state(seq, batch)}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        enc = self.encode_history(batch)
        context = enc["state_emb"]
        sid_logits = []
        tau = max(self.sid_temp, 1e-6)
        for level in range(self.sid_levels):
            logits = self.sid_heads[level](context)
            sid_logits.append(logits)
            probs = torch.softmax(logits / tau, dim=-1)
            exp_emb = probs @ self.sid_token_embeds[level].weight
            context = self.sid_res_norms[level](context - exp_emb)
        return {"sid_logits": sid_logits, "state_emb": enc["state_emb"], "seq_emb": enc["seq_emb"]}


def hpn_loss(out: dict[str, torch.Tensor | list[torch.Tensor]], target_sid: torch.Tensor) -> dict[str, torch.Tensor]:
    logits_list = out["sid_logits"]
    if not isinstance(logits_list, list):
        raise TypeError("sid_logits must be a list of tensors.")
    levels = min(len(logits_list), int(target_sid.shape[1]))
    losses = []
    token_correct = []
    full = torch.ones(target_sid.shape[0], dtype=torch.bool, device=target_sid.device)
    for level in range(levels):
        logits = logits_list[level]
        target_l = target_sid[:, level].long().clamp(min=0, max=logits.shape[-1] - 1)
        losses.append(torch.nn.functional.cross_entropy(logits, target_l))
        pred_l = logits.argmax(dim=-1)
        correct = pred_l == target_l
        token_correct.append(correct.float().mean())
        full &= correct
    if not losses:
        raise RuntimeError("target_sid has no usable semantic levels.")
    loss = sum(losses) / max(len(losses), 1)
    metrics = {"loss": loss, "full_path_acc": full.float().mean().detach()}
    for idx, acc in enumerate(token_correct):
        metrics[f"token_acc_l{idx+1}"] = acc.detach()
    return metrics


def score_sid_candidates(sid_logits: list[torch.Tensor], candidate_sid: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros(candidate_sid.shape[:2], dtype=torch.float32, device=candidate_sid.device)
    for level, logits in enumerate(sid_logits):
        log_probs = torch.log_softmax(logits, dim=-1)
        idx = candidate_sid[:, :, level].long().clamp(min=0, max=logits.shape[-1] - 1)
        scores = scores + torch.gather(log_probs, 1, idx)
    return scores / max(len(sid_logits), 1)


def beam_decode_sid_paths(sid_logits: list[torch.Tensor], top_paths: int = 32, branch_k: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = sid_logits[0].shape[0]
    device = sid_logits[0].device
    all_paths = []
    all_scores = []
    for b in range(batch_size):
        beams: list[tuple[list[int], float]] = [([], 0.0)]
        for logits in sid_logits:
            log_probs = torch.log_softmax(logits[b], dim=-1)
            vals, idxs = torch.topk(log_probs, k=min(branch_k, log_probs.shape[-1]))
            new_beams = []
            for path, score in beams:
                for val, idx in zip(vals.tolist(), idxs.tolist()):
                    new_beams.append((path + [int(idx)], float(score + val)))
            new_beams.sort(key=lambda item: item[1], reverse=True)
            beams = new_beams[:top_paths]
        paths = [item[0] for item in beams]
        scores = [item[1] for item in beams]
        while len(paths) < top_paths:
            paths.append([0] * len(sid_logits))
            scores.append(float("-inf"))
        all_paths.append(paths[:top_paths])
        all_scores.append(scores[:top_paths])
    return (
        torch.tensor(all_paths, dtype=torch.long, device=device),
        torch.tensor(all_scores, dtype=torch.float32, device=device),
    )
