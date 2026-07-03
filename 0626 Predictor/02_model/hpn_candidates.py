from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SidCandidatePool:
    """Score concrete candidate items by their SID paths.

    This follows the 0408/HSRL selection contract: the HPN produces a token
    distribution at each semantic level, then each concrete candidate item is
    scored by the probability of its full SID path.
    """

    def __init__(
        self,
        dense_item2sid: np.ndarray,
        max_candidate_items: int = 0,
        chunk_size: int = 65536,
    ) -> None:
        self.dense_item2sid = dense_item2sid
        self.chunk_size = max(int(chunk_size), 1024)
        sid_slice = dense_item2sid[1:]
        valid = np.all(sid_slice >= 0, axis=1)
        dense_ids = (np.nonzero(valid)[0] + 1).astype(np.int64, copy=False)
        max_candidate_items = int(max_candidate_items)
        if max_candidate_items > 0 and dense_ids.shape[0] > max_candidate_items:
            rng = np.random.default_rng(2026)
            dense_ids = np.sort(rng.choice(dense_ids, size=max_candidate_items, replace=False)).astype(np.int64, copy=False)
        self.candidate_dense_ids = dense_ids

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_dense_ids.shape[0])

    def score_dense_ids(
        self,
        sid_logits: list[torch.Tensor],
        dense_ids: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        if dense_ids.size == 0:
            return torch.empty((sid_logits[0].shape[0], 0), dtype=torch.float32, device=device)
        sid_np = np.asarray(self.dense_item2sid[dense_ids], dtype=np.int64)
        sid = torch.as_tensor(sid_np, dtype=torch.long, device=device)
        scores = torch.zeros((sid_logits[0].shape[0], sid.shape[0]), dtype=torch.float32, device=device)
        for level, logits in enumerate(sid_logits):
            log_probs = F.log_softmax(logits, dim=-1)
            token = sid[:, level].clamp(min=0, max=log_probs.shape[-1] - 1)
            scores = scores + log_probs.index_select(dim=1, index=token)
        return scores / max(len(sid_logits), 1)

    def score_row_dense_ids(
        self,
        sid_logits: list[torch.Tensor],
        dense_ids: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        sid_np = np.asarray(self.dense_item2sid[dense_ids], dtype=np.int64)
        sid = torch.as_tensor(sid_np, dtype=torch.long, device=device)
        scores = torch.zeros((sid.shape[0],), dtype=torch.float32, device=device)
        for level, logits in enumerate(sid_logits):
            log_probs = F.log_softmax(logits, dim=-1)
            token = sid[:, level].clamp(min=0, max=log_probs.shape[-1] - 1)
            scores = scores + log_probs.gather(1, token.unsqueeze(1)).squeeze(1)
        return scores / max(len(sid_logits), 1)

    def topk_from_logits(
        self,
        sid_logits: list[torch.Tensor],
        max_candidates: int,
        device: torch.device,
        fallback_dense: np.ndarray | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, int]:
        batch_size = int(sid_logits[0].shape[0])
        max_candidates = max(int(max_candidates), 1)
        if self.candidate_count == 0:
            dense = np.zeros((batch_size, max_candidates), dtype=np.int64)
            scores = torch.full((batch_size, max_candidates), -1e9, dtype=torch.float32, device=device)
            mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)
            return dense, scores, mask, batch_size

        top_scores = torch.full((batch_size, 0), -1e9, dtype=torch.float32, device=device)
        top_dense = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        for start in range(0, self.candidate_count, self.chunk_size):
            dense_ids = self.candidate_dense_ids[start : start + self.chunk_size]
            scores = self.score_dense_ids(sid_logits, dense_ids, device)
            chunk_k = min(max_candidates, scores.shape[1])
            chunk_scores, chunk_pos = torch.topk(scores, k=chunk_k, dim=1)
            dense_tensor = torch.as_tensor(dense_ids, dtype=torch.long, device=device)
            chunk_dense = dense_tensor.unsqueeze(0).expand(batch_size, -1).gather(1, chunk_pos)
            merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
            merged_dense = torch.cat([top_dense, chunk_dense], dim=1)
            keep_k = min(max_candidates, merged_scores.shape[1])
            top_scores, keep_pos = torch.topk(merged_scores, k=keep_k, dim=1)
            top_dense = merged_dense.gather(1, keep_pos)

        if top_dense.shape[1] < max_candidates:
            pad = max_candidates - top_dense.shape[1]
            top_dense = torch.cat([top_dense, torch.zeros((batch_size, pad), dtype=torch.long, device=device)], dim=1)
            top_scores = torch.cat(
                [top_scores, torch.full((batch_size, pad), -1e9, dtype=torch.float32, device=device)],
                dim=1,
            )

        if fallback_dense is not None and max_candidates > 0:
            fallback = np.asarray(fallback_dense, dtype=np.int64)
            valid = fallback > 0
            if np.any(valid):
                safe_fallback = fallback.clip(0, self.dense_item2sid.shape[0] - 1)
                row_scores = self.score_row_dense_ids(sid_logits, safe_fallback, device)
                top_dense_cpu = top_dense.detach().cpu().numpy()
                for row_idx, dense_id in enumerate(fallback.tolist()):
                    if dense_id <= 0:
                        continue
                    if dense_id in top_dense_cpu[row_idx].tolist():
                        continue
                    if max_candidates > 1:
                        top_dense[row_idx, 1:] = top_dense[row_idx, :-1].clone()
                        top_scores[row_idx, 1:] = top_scores[row_idx, :-1].clone()
                    top_dense[row_idx, 0] = int(dense_id)
                    top_scores[row_idx, 0] = row_scores[row_idx]

        mask = top_dense > 0
        dense_np = top_dense.detach().cpu().numpy().astype(np.int64, copy=False)
        empty = int((~mask.any(dim=1)).sum().detach().cpu())
        return dense_np, top_scores, mask, empty


class SidPathIndex(SidCandidatePool):
    """Backward-compatible wrapper for older CLI/code names."""

    def __init__(
        self,
        dense_item2sid: np.ndarray,
        max_items_per_sid: int = 4,
        max_index_items: int = 0,
        chunk_size: int = 65536,
    ) -> None:
        del max_items_per_sid
        super().__init__(dense_item2sid, max_candidate_items=max_index_items, chunk_size=chunk_size)


def build_hpn_candidate_batch(
    batch: dict[str, torch.Tensor],
    hpn,
    index: SidCandidatePool,
    dense2orig: np.ndarray,
    store,
    device: torch.device,
    max_candidates: int,
    top_sid_paths: int,
    branch_k: int,
    fallback_to_logged: bool = False,
) -> dict[str, torch.Tensor | np.ndarray | int]:
    del top_sid_paths, branch_k
    with torch.no_grad():
        hpn_out = hpn(batch)
        sid_logits = hpn_out["sid_logits"]
        if not isinstance(sid_logits, list):
            raise TypeError("HPN output sid_logits must be a list of tensors.")
        fallback_dense = None
        if fallback_to_logged and "target_dense_item_id" in batch:
            fallback_dense = batch["target_dense_item_id"].detach().cpu().numpy().astype(np.int64, copy=False)
        candidate_dense_np, hpn_scores, masks, empty = index.topk_from_logits(
            sid_logits=sid_logits,
            max_candidates=max_candidates,
            device=device,
            fallback_dense=fallback_dense,
        )

    candidate_orig = np.zeros_like(candidate_dense_np)
    valid_dense = (candidate_dense_np > 0) & (candidate_dense_np < len(dense2orig))
    candidate_orig[valid_dense] = dense2orig[candidate_dense_np[valid_dense]]
    lookup_ids = candidate_dense_np if getattr(store, "id_mode", "orig") == "dense" else candidate_orig.astype(np.uint32)
    candidate_features = torch.tensor(store.lookup(lookup_ids), dtype=torch.float32, device=device)
    return {
        "candidate_features": candidate_features,
        "candidate_dense": candidate_dense_np,
        "hpn_scores": hpn_scores.to(device),
        "candidate_mask": masks.to(device),
        "empty_rows": empty,
        "candidate_pool_size": index.candidate_count,
    }
