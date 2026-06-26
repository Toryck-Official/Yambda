from __future__ import annotations

import numpy as np
import torch

from hpn import beam_decode_sid_paths


class SidPathIndex:
    def __init__(self, dense_item2sid: np.ndarray, max_items_per_sid: int = 4, max_index_items: int = 0) -> None:
        self.path_to_dense: dict[tuple[int, ...], list[int]] = {}
        stop = dense_item2sid.shape[0] if max_index_items <= 0 else min(dense_item2sid.shape[0], max_index_items + 1)
        for dense_id in range(1, stop):
            sid = dense_item2sid[dense_id]
            if np.any(sid < 0):
                continue
            path = tuple(int(x) for x in sid.tolist())
            for prefix_len in range(0, len(path) + 1):
                key = path[:prefix_len]
                bucket = self.path_to_dense.setdefault(key, [])
                if len(bucket) < max_items_per_sid:
                    bucket.append(int(dense_id))

    def candidates_for_paths(
        self,
        paths: list[list[int]],
        scores: list[float],
        max_candidates: int,
    ) -> tuple[list[int], list[float]]:
        dense_ids: list[int] = []
        hpn_scores: list[float] = []
        used = set()
        for path, score in zip(paths, scores):
            clean_path = tuple(int(x) for x in path)
            for prefix_len in range(len(clean_path), -1, -1):
                bucket = self.path_to_dense.get(clean_path[:prefix_len], [])
                for dense_id in bucket:
                    if dense_id in used:
                        continue
                    dense_ids.append(dense_id)
                    hpn_scores.append(float(score))
                    used.add(dense_id)
                    if len(dense_ids) >= max_candidates:
                        return dense_ids, hpn_scores
                if bucket:
                    break
        return dense_ids, hpn_scores


def build_hpn_candidate_batch(
    batch: dict[str, torch.Tensor],
    hpn,
    index: SidPathIndex,
    dense2orig: np.ndarray,
    store,
    device: torch.device,
    max_candidates: int,
    top_sid_paths: int,
    branch_k: int,
    fallback_to_logged: bool = False,
) -> dict[str, torch.Tensor | np.ndarray | int]:
    hpn_out = hpn(batch)
    paths_t, path_scores_t = beam_decode_sid_paths(hpn_out["sid_logits"], top_sid_paths, branch_k)
    candidate_dense = []
    hpn_scores = []
    masks = []
    empty = 0

    for row_idx in range(paths_t.shape[0]):
        paths = paths_t[row_idx].detach().cpu().tolist()
        scores = path_scores_t[row_idx].detach().cpu().tolist()
        dense_ids, scores_out = index.candidates_for_paths(paths, scores, max_candidates)
        if fallback_to_logged:
            logged_dense = int(batch["target_dense_item_id"][row_idx].detach().cpu())
            if logged_dense > 0 and logged_dense not in dense_ids:
                dense_ids = [logged_dense, *dense_ids]
                scores_out = [scores[0] if scores else 0.0, *scores_out]
        if not dense_ids:
            empty += 1
            dense_ids = [0]
            scores_out = [-1e9]

        mask = [1 if dense_id > 0 else 0 for dense_id in dense_ids]
        while len(dense_ids) < max_candidates:
            dense_ids.append(0)
            scores_out.append(-1e9)
            mask.append(0)
        candidate_dense.append(dense_ids[:max_candidates])
        hpn_scores.append(scores_out[:max_candidates])
        masks.append(mask[:max_candidates])

    candidate_dense_np = np.asarray(candidate_dense, dtype=np.int64)
    candidate_orig = np.zeros_like(candidate_dense_np)
    valid_dense = (candidate_dense_np > 0) & (candidate_dense_np < len(dense2orig))
    candidate_orig[valid_dense] = dense2orig[candidate_dense_np[valid_dense]]
    candidate_features = torch.tensor(store.lookup(candidate_orig.astype(np.uint32)), dtype=torch.float32, device=device)
    return {
        "candidate_features": candidate_features,
        "candidate_dense": candidate_dense_np,
        "hpn_scores": torch.tensor(hpn_scores, dtype=torch.float32, device=device),
        "candidate_mask": torch.tensor(masks, dtype=torch.bool, device=device),
        "empty_rows": empty,
    }
