from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


RAW_EVENTS = ["listen", "like", "dislike", "unlike", "undislike"]
EVENT_TO_ID = {"pad": 0, **{name: idx + 1 for idx, name in enumerate(RAW_EVENTS)}}
ID_TO_EVENT = {idx: name for name, idx in EVENT_TO_ID.items()}
RESPONSE_TO_INDEX = {name: idx for idx, name in enumerate(RAW_EVENTS)}
INDEX_TO_RESPONSE = {idx: name for name, idx in RESPONSE_TO_INDEX.items()}

REGRET_TYPES = ["none", "low_play", "dislike", "unlike"]
# low_play is kept for checkpoint compatibility, but the current predictor
# reward policy treats any listen as non-negative positive feedback.
REGRET_TO_ID = {name: idx for idx, name in enumerate(REGRET_TYPES)}
ID_TO_REGRET = {idx: name for name, idx in REGRET_TO_ID.items()}


@dataclass(frozen=True)
class RewardConfig:
    version: str = "v2"
    positive_play_threshold: float = 0.8
    low_play_regret_threshold: float = 0.2
    undo_grace_seconds: int = 10
    timestamp_unit_seconds: float = 5.0
    v2_like: float = 0.8
    v2_dislike: float = 1.2
    v2_unlike: float = 0.6
    v2_undislike: float = 0.2
    v2_clip_min: float = -2.0
    v2_clip_max: float = 2.0


def history_signal(event_type: str, played_ratio_norm: float) -> float:
    if event_type == "listen":
        return float(played_ratio_norm)
    if event_type == "like":
        return 1.0
    if event_type == "dislike":
        return -1.0
    if event_type == "unlike":
        return -0.5
    if event_type == "undislike":
        return 0.5
    return 0.0


def play_bucket_id(played_ratio_norm: float) -> int:
    if played_ratio_norm <= 0.0:
        return 0
    if played_ratio_norm <= 0.2:
        return 1
    if played_ratio_norm <= 0.8:
        return 2
    if played_ratio_norm <= 1.0:
        return 3
    return 4


def effective_feedback(events: Iterable[dict], cfg: RewardConfig) -> dict[str, int]:
    pending_like_times: list[int] = []
    pending_dislike_times: list[int] = []
    effective_dislike_count = 0
    effective_unlike_count = 0
    effective_undislike_count = 0
    canceled_like_count = 0
    canceled_dislike_count = 0

    ordered = sorted(events, key=lambda item: (int(item["timestamp"]), int(item.get("raw_pos", 0))))
    for event in ordered:
        event_type = str(event["event_type"])
        timestamp = int(event["timestamp"])
        if event_type == "like":
            pending_like_times.append(timestamp)
        elif event_type == "unlike":
            if pending_like_times:
                like_time = pending_like_times.pop()
                gap = (timestamp - like_time) * float(cfg.timestamp_unit_seconds)
                if 0 <= gap <= cfg.undo_grace_seconds:
                    canceled_like_count += 1
                else:
                    effective_unlike_count += 1
            else:
                effective_unlike_count += 1
        elif event_type == "dislike":
            pending_dislike_times.append(timestamp)
        elif event_type == "undislike":
            if pending_dislike_times:
                dislike_time = pending_dislike_times.pop()
                gap = (timestamp - dislike_time) * float(cfg.timestamp_unit_seconds)
                if 0 <= gap <= cfg.undo_grace_seconds:
                    canceled_dislike_count += 1
                else:
                    effective_dislike_count += 1
                    effective_undislike_count += 1
            else:
                effective_undislike_count += 1

    effective_dislike_count += len(pending_dislike_times)
    return {
        "effective_like": int(len(pending_like_times) > 0),
        "effective_dislike": int(effective_dislike_count > 0),
        "effective_unlike": int(effective_unlike_count > 0),
        "effective_undislike": int(effective_undislike_count > 0),
        "canceled_like_count": int(canceled_like_count),
        "canceled_dislike_count": int(canceled_dislike_count),
    }


def summarize_events(events: list[dict], cfg: RewardConfig) -> dict:
    if cfg.version != "v2":
        raise ValueError("This predictor workspace uses reward v2 only.")

    event_types = [str(event["event_type"]) for event in events]
    play_ratios = [float(event["played_ratio_norm"]) for event in events if event["event_type"] == "listen"]
    max_play_ratio = max(play_ratios) if play_ratios else 0.0
    mean_play_ratio = float(np.mean(play_ratios)) if play_ratios else 0.0
    n_listen = int(sum(1 for item in event_types if item == "listen"))
    feedback = effective_feedback(events, cfg)

    play = float(np.clip(max_play_ratio, 0.0, 1.0))
    # Predictor-side reward policy: any observed play is positive evidence.
    # Low play may be weak positive feedback, but it is no longer a negative
    # reward or a regret label by itself.
    play_reward = play if n_listen > 0 else 0.0
    reward_raw = (
        play_reward
        + cfg.v2_like * feedback["effective_like"]
        - cfg.v2_dislike * feedback["effective_dislike"]
        - cfg.v2_unlike * feedback["effective_unlike"]
        + cfg.v2_undislike * feedback["effective_undislike"]
    )
    reward_scaled = float(np.clip(reward_raw, cfg.v2_clip_min, cfg.v2_clip_max))

    if feedback["effective_dislike"]:
        regret_type = "dislike"
        regret_strength = float(cfg.v2_dislike)
    elif feedback["effective_unlike"]:
        regret_type = "unlike"
        regret_strength = float(cfg.v2_unlike)
    else:
        regret_type = "none"
        regret_strength = 0.0

    feedback_label = int(
        (feedback["effective_like"] or n_listen > 0)
        and not feedback["effective_dislike"]
        and not feedback["effective_unlike"]
    )
    return {
        "n_events": int(len(events)),
        "n_listen": n_listen,
        "max_play_ratio": float(max_play_ratio),
        "mean_play_ratio": float(mean_play_ratio),
        **feedback,
        "reward_raw": float(reward_raw),
        "reward_scaled": reward_scaled,
        "feedback_label": feedback_label,
        "regret_type": regret_type,
        "regret_type_id": int(REGRET_TO_ID[regret_type]),
        "regret_strength": regret_strength,
    }
