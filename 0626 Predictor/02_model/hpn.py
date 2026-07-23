from __future__ import annotations

import torch
import torch.nn as nn

from state_encoder import StateEncoder


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
        sid_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.sid_temp = float(sid_temp)

        self.state_encoder = StateEncoder(
            item_dim=self.item_dim,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            n_layer=int(n_layer),
            n_head=int(n_head),
            dropout=float(dropout),
            response_dim=5,
        )

        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, self.sid_vocab_size) for _ in range(self.sid_levels)])
        self.sid_token_embeds = nn.ModuleList([nn.Embedding(self.sid_vocab_size, self.d_model) for _ in range(self.sid_levels)])
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.state_encoder(batch)

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
    losses = []
    token_correct = []
    full = torch.ones(target_sid.shape[0], dtype=torch.bool, device=target_sid.device)
    for level, logits in enumerate(logits_list):
        target_l = target_sid[:, level].long()
        losses.append(torch.nn.functional.cross_entropy(logits, target_l))
        pred_l = logits.argmax(dim=-1)
        correct = pred_l == target_l
        token_correct.append(correct.float().mean())
        full &= correct
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
