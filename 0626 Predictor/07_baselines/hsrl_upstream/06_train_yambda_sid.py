#!/usr/bin/env python3
"""
最小 Yambda-HSRL SID 训练入口。

它把当前 baseline 的核心组件串起来：
- YambdaEnvironment_GPU_HAC: 用 05 的 UserResponse 给 reward
- SIDPolicy_credit: HPN / Actor
- Token_Critic: MLC / Critic
- SIDFacade_credit: semantic action -> candidate item
- DDPG: 当前仓库里已有的 actor-critic 更新器

它是正式 baseline 的单文件入口；推荐通过 run_stage.sh 读取集中配置来运行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


BASELINE_DIR = Path(__file__).resolve().parents[1]
PREDICTOR_ROOT = BASELINE_DIR.parent
WORKSPACE_ROOT = PREDICTOR_ROOT.parent
PROJECT_ROOT = Path(os.environ.get("HSRL_PROJECT_ROOT", str(WORKSPACE_ROOT / "HSRL")))
REGRET_ROOT = WORKSPACE_ROOT / "Regret"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(REGRET_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRET_ROOT))

from adapter.bootstrap import install_hsrl_adapter  # noqa: E402

install_hsrl_adapter()

from env.YambdaEnvironment_GPU_HAC import YambdaEnvironment_GPU_HAC  # type: ignore  # noqa: E402
from model.agents.DDPG import DDPG  # type: ignore  # noqa: E402
from model.critic.Token_Critic import Token_Critic  # type: ignore  # noqa: E402
from model.facade.SIDFacade_credit import SIDFacade_credit  # type: ignore  # noqa: E402
from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore  # noqa: E402
from regret_core.data.transition_dataset import TransitionIterableDataset  # noqa: E402
from utils import set_random_seed  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Train minimal Yambda SID HSRL baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--train_mode",
        type=str,
        default="online_env",
        choices=["online_env", "offline_transition"],
        help="online_env keeps the old simulator rollout path; offline_transition trains directly on transition parquet.",
    )
    parser.add_argument(
        "--transition_root",
        type=str,
        default=str(WORKSPACE_ROOT / "Regret/artifacts/transitions/raw_rqkmeans_v2_smoke_timefix"),
        help="Directory containing train/val/test transition parquet shards.",
    )
    parser.add_argument(
        "--item_features_npy",
        type=str,
        default=str(WORKSPACE_ROOT / "Regret/artifacts/mappings/raw_rqkmeans/dense_item_features.npy"),
        help="Dense item feature matrix used by transition parquet training.",
    )
    parser.add_argument(
        "--urm_log_path",
        type=str,
        default=str(WORKSPACE_ROOT / "artifacts/env/log/yambda_user_env.model.log"),
    )
    parser.add_argument(
        "--dense_item2sid_npy",
        type=str,
        default=str(WORKSPACE_ROOT / "Regret/artifacts/mappings/raw_rqkmeans/dense_item2sid.npy"),
    )
    parser.add_argument(
        "--hpn_checkpoint",
        type=str,
        default=str(WORKSPACE_ROOT / "artifacts/models/hpn_warmstart.pt"),
    )
    parser.add_argument("--slate_size", type=int, default=1)
    parser.add_argument("--max_candidate_items", type=int, default=50000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--start_timestamp", type=int, default=2000)
    parser.add_argument("--episode_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=10000)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--offline_epochs", type=int, default=3)
    parser.add_argument("--shuffle_train_files", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=0)
    parser.add_argument("--train_sample_across_files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offline_reward_threshold", type=float, default=0.0)
    parser.add_argument("--offline_reward_temperature", type=float, default=0.5)
    parser.add_argument("--offline_bc_weight", type=float, default=1.0)
    parser.add_argument("--offline_avoid_weight", type=float, default=0.25)
    parser.add_argument("--offline_critic_weight", type=float, default=0.5)
    parser.add_argument("--offline_entropy_weight", type=float, default=0.001)
    parser.add_argument("--offline_grad_clip", type=float, default=5.0)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--train_every_n_step", type=int, default=5)
    parser.add_argument("--check_episode", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--actor_decay", type=float, default=1e-5)
    parser.add_argument("--critic_decay", type=float, default=1e-5)
    parser.add_argument("--target_mitigate_coef", type=float, default=0.01)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--bc_coef", type=float, default=0.1)
    parser.add_argument("--initial_greedy_epsilon", type=float, default=0.0)
    parser.add_argument("--final_greedy_epsilon", type=float, default=0.0)
    parser.add_argument("--elbow_greedy", type=float, default=0.5)
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--sara_eta", type=float, default=0.5)
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    parser.add_argument("--enable_regret_intervention", action="store_true")
    parser.add_argument("--regret_memory_size", type=int, default=20)
    parser.add_argument("--regret_reward_threshold", type=float, default=0.0)
    parser.add_argument("--regret_gamma", type=float, default=0.9)
    parser.add_argument("--regret_phi_scale", type=float, default=1.0)
    parser.add_argument("--regret_phi_clip", type=float, default=1.5)
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--critic_dropout_rate", type=float, default=0.2)
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(WORKSPACE_ROOT / "artifacts/models/yambda_sid"),
    )
    parser.add_argument(
        "--save_meta",
        type=str,
        default=str(WORKSPACE_ROOT / "artifacts/models/yambda_sid.meta.json"),
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """输入：设备字符串。输出：torch.device。"""
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    """输入：dense_item2sid 数组。输出：SID 层数和每层 vocab size。"""
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def load_hpn_checkpoint(actor: SIDPolicy_credit, ckpt_path: Path, device: torch.device) -> bool:
    """输入：actor、checkpoint 路径、设备。输出：是否成功加载。"""
    if not ckpt_path.exists():
        print(f"[hpn] checkpoint not found, use random init: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    actor_state = actor.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in actor_state and tuple(actor_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    missing, unexpected = actor.load_state_dict(compatible_state, strict=False)
    print(
        f"[hpn] loaded {ckpt_path}, compatible={len(compatible_state)}, "
        f"skipped_shape={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return True


class OfflineTransitionEnvSpec:
    """Minimal environment spec needed by SIDPolicy_credit and Token_Critic."""

    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def make_transition_loader(args: argparse.Namespace, split_name: str) -> DataLoader:
    """Build a streaming transition dataloader for the requested split."""
    root = Path(args.transition_root)
    path = root / split_name
    max_rows = args.max_train_rows if split_name == "train" else args.max_val_rows
    is_train = split_name == "train"
    dataset = TransitionIterableDataset(
        path,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=max_rows,
        batch_size=args.read_batch_size,
        shuffle_files=is_train and args.shuffle_train_files,
        shuffle_buffer_size=args.shuffle_buffer_size if is_train else 0,
        seed=args.seed,
        sample_across_files=is_train and args.train_sample_across_files,
    )
    print(
        f"[data] {split_name}: files={len(dataset.files)} max_rows={max_rows} "
        f"shuffle_files={dataset.shuffle_files} shuffle_buffer={dataset.shuffle_buffer_size} "
        f"sample_across_files={dataset.sample_across_files}"
    )
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def target_sid_from_dense_ids(
    dense_item2sid: np.ndarray,
    item_ids: torch.Tensor,
    device: torch.device,
    sid_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map dense item ids to SID tokens and return a validity mask."""
    item_ids_np = item_ids.detach().cpu().numpy().astype(np.int64)
    sid_np = np.asarray(dense_item2sid[item_ids_np], dtype=np.int64)
    sid = torch.as_tensor(sid_np, dtype=torch.long, device=device)
    valid = (sid >= 0).all(dim=1) & (sid < int(sid_vocab_size)).all(dim=1)
    return sid, valid


def compute_offline_transition_loss(
    actor: SIDPolicy_credit,
    critic: Token_Critic,
    batch: dict[str, torch.Tensor],
    dense_item2sid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sid_vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward-gated offline loss for transition data.

    Positive reward transitions imitate the logged SID path. Negative reward
    transitions reduce probability on that logged SID path, preventing the
    policy from blindly cloning known bad actions.
    """
    history_features = batch["history_features"].to(device)
    reward = batch["reward"].to(device).float()
    target_sid, valid = target_sid_from_dense_ids(dense_item2sid, batch["target_dense_item_id"], device, sid_vocab_size)
    if not bool(valid.any()):
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"n": 0.0}

    history_features = history_features[valid]
    reward = reward[valid]
    target_sid = target_sid[valid]

    output = actor({"history_features": history_features})
    critic_output = critic({"context_list": output["context_list"]})
    q_pred = critic_output["q"]

    temp = max(float(args.offline_reward_temperature), 1e-6)
    threshold = float(args.offline_reward_threshold)
    pos_weight = torch.sigmoid((reward - threshold) / temp).detach()
    neg_weight = torch.sigmoid((threshold - reward) / temp).detach()

    nll = torch.zeros_like(reward)
    avoid = torch.zeros_like(reward)
    token_acc_sum = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    full_match = torch.ones_like(reward, dtype=torch.bool)

    for level, logits_l in enumerate(output["sid_logits"]):
        z_l = target_sid[:, level]
        log_probs_l = F.log_softmax(logits_l, dim=-1)
        logp_l = log_probs_l.gather(1, z_l.view(-1, 1)).squeeze(1)
        prob_l = logp_l.exp().clamp(max=1.0 - 1e-6)
        nll = nll - logp_l
        avoid = avoid - torch.log1p(-prob_l)
        pred_l = logits_l.argmax(dim=-1)
        token_acc_sum = token_acc_sum + (pred_l == z_l).float().mean()
        full_match = full_match & (pred_l == z_l)
        probs_l = log_probs_l.exp()
        entropy = entropy - (probs_l * log_probs_l).sum(dim=-1).mean()

    n_level = max(len(output["sid_logits"]), 1)
    token_acc = token_acc_sum / n_level
    entropy = entropy / n_level
    pos_loss = (pos_weight * nll).sum() / pos_weight.sum().clamp_min(1e-6)
    neg_loss = (neg_weight * avoid).sum() / neg_weight.sum().clamp_min(1e-6)
    critic_loss = F.mse_loss(q_pred, reward)
    actor_loss = (
        float(args.offline_bc_weight) * pos_loss
        + float(args.offline_avoid_weight) * neg_loss
        - float(args.offline_entropy_weight) * entropy
    )
    loss = actor_loss + float(args.offline_critic_weight) * critic_loss

    with torch.no_grad():
        pos_mask = reward >= threshold
        neg_mask = ~pos_mask
        pos_token_acc = token_acc
        neg_token_acc = token_acc
        if bool(pos_mask.any()):
            pos_level_acc = []
            for level, logits_l in enumerate(output["sid_logits"]):
                pos_level_acc.append((logits_l[pos_mask].argmax(dim=-1) == target_sid[pos_mask, level]).float().mean())
            pos_token_acc = torch.stack(pos_level_acc).mean()
        if bool(neg_mask.any()):
            neg_level_acc = []
            for level, logits_l in enumerate(output["sid_logits"]):
                neg_level_acc.append((logits_l[neg_mask].argmax(dim=-1) == target_sid[neg_mask, level]).float().mean())
            neg_token_acc = torch.stack(neg_level_acc).mean()
        metrics = {
            "n": float(reward.numel()),
            "loss": float(loss.detach().cpu()),
            "actor": float(actor_loss.detach().cpu()),
            "critic": float(critic_loss.detach().cpu()),
            "pos": float(pos_loss.detach().cpu()),
            "avoid": float(neg_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "q_mse": float(critic_loss.detach().cpu()),
            "reward_mean": float(reward.mean().detach().cpu()),
            "pos_share": float(pos_mask.float().mean().detach().cpu()),
            "token_acc": float(token_acc.detach().cpu()),
            "full_sid_acc": float(full_match.float().mean().detach().cpu()),
            "pos_token_acc": float(pos_token_acc.detach().cpu()),
            "neg_token_acc": float(neg_token_acc.detach().cpu()),
        }
    return loss, metrics


def run_offline_epoch(
    actor: SIDPolicy_credit,
    critic: Token_Critic,
    loader: DataLoader,
    dense_item2sid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sid_vocab_size: int,
    actor_optimizer: torch.optim.Optimizer | None,
    critic_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    split_name: str,
) -> dict[str, float]:
    is_train = actor_optimizer is not None and critic_optimizer is not None
    actor.train(is_train)
    critic.train(is_train)
    totals: dict[str, float] = {}
    n_rows = 0.0
    pbar = tqdm(loader, desc=f"[epoch {epoch}] {split_name}", ncols=120)
    for batch in pbar:
        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_offline_transition_loss(
                actor=actor,
                critic=critic,
                batch=batch,
                dense_item2sid=dense_item2sid,
                args=args,
                device=device,
                sid_vocab_size=sid_vocab_size,
            )
            if metrics.get("n", 0.0) <= 0:
                continue
            if is_train:
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                if args.offline_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.offline_grad_clip)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.offline_grad_clip)
                actor_optimizer.step()
                critic_optimizer.step()

        batch_n = metrics["n"]
        n_rows += batch_n
        for key, value in metrics.items():
            if key == "n":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_n
        if n_rows > 0:
            pbar.set_postfix(
                loss=totals.get("loss", 0.0) / n_rows,
                q_mse=totals.get("q_mse", 0.0) / n_rows,
                token_acc=totals.get("token_acc", 0.0) / n_rows,
                full=totals.get("full_sid_acc", 0.0) / n_rows,
                pos_share=totals.get("pos_share", 0.0) / n_rows,
            )
    pbar.close()
    if n_rows <= 0:
        return {"n": 0.0}
    return {"n": n_rows, **{key: value / n_rows for key, value in totals.items()}}


def run_offline_transition_training(
    args: argparse.Namespace,
    dense_item2sid: np.ndarray,
    sid_levels: int,
    sid_vocab_size: int,
    device: torch.device,
) -> None:
    """Train SID actor/critic directly from transition parquet shards."""
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    env = OfflineTransitionEnvSpec(
        n_item=int(dense_item2sid.shape[0] - 1),
        item_dim=int(feature_shape[1]),
        max_seq_len=args.max_seq_len,
    )
    policy_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
        sara_eta=args.sara_eta,
        sara_layer_weights=args.sara_layer_weights,
    )
    actor = SIDPolicy_credit(policy_args, env).to(device)
    hpn_loaded = load_hpn_checkpoint(actor, Path(args.hpn_checkpoint), device)
    critic_args = SimpleNamespace(
        critic_hidden_dims=args.critic_hidden_dims,
        critic_dropout_rate=args.critic_dropout_rate,
    )
    critic = Token_Critic(critic_args, env, actor).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)

    train_loader = make_transition_loader(args, "train")
    val_loader = make_transition_loader(args, "val")
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.offline_epochs) + 1):
        train_metrics = run_offline_epoch(
            actor,
            critic,
            train_loader,
            dense_item2sid,
            args,
            device,
            sid_vocab_size,
            actor_optimizer,
            critic_optimizer,
            epoch,
            "train",
        )
        val_metrics = run_offline_epoch(
            actor,
            critic,
            val_loader,
            dense_item2sid,
            args,
            device,
            sid_vocab_size,
            None,
            None,
            epoch,
            "val",
        )
        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics.get('loss', float('nan')):.5f} "
            f"train_token_acc={train_metrics.get('token_acc', float('nan')):.3f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.5f} "
            f"val_q_mse={val_metrics.get('q_mse', float('nan')):.5f} "
            f"val_token_acc={val_metrics.get('token_acc', float('nan')):.3f} "
            f"val_full_sid_acc={val_metrics.get('full_sid_acc', float('nan')):.3f} "
            f"val_pos_share={val_metrics.get('pos_share', float('nan')):.3f}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics.get("loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(actor.state_dict(), args.save_path + "_actor")
            torch.save(critic.state_dict(), args.save_path + "_critic")
            torch.save(actor_optimizer.state_dict(), args.save_path + "_actor_optimizer")
            torch.save(critic_optimizer.state_dict(), args.save_path + "_critic_optimizer")
            print(f"[save] best offline transition checkpoint at epoch={epoch} val_loss={val_loss:.5f}")

    meta = {
        "training_args": vars(args),
        "train_mode": "offline_transition",
        "device": str(device),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "item_features_npy": args.item_features_npy,
        "transition_root": args.transition_root,
        "hpn_checkpoint": args.hpn_checkpoint,
        "hpn_loaded": hpn_loaded,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "save_path": args.save_path,
        "actor_checkpoint": args.save_path + "_actor",
        "critic_checkpoint": args.save_path + "_critic",
    }
    Path(args.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] offline transition actor saved to {args.save_path}_actor")
    print(f"[done] offline transition critic saved to {args.save_path}_critic")
    print(f"[done] train meta saved to {args.save_meta}")


def main() -> None:
    """主入口：跑一个最小 Yambda SID-HSRL baseline 训练。"""
    args = parse_args()
    save_meta = Path(args.save_meta)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[device] using {device}")
    set_random_seed(args.seed)

    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    print(f"[sid] sid_levels={sid_levels}, sid_vocab_size={sid_vocab_size}")

    if args.train_mode == "offline_transition":
        print("[mode] offline_transition: train actor/critic directly from transition parquet")
        run_offline_transition_training(args, dense_item2sid, sid_levels, sid_vocab_size, device)
        return

    env_args = SimpleNamespace(
        env_path="",
        reward_func="direct_score",
        max_step_per_episode=10,
        initial_temper=5,
        urm_log_path=args.urm_log_path,
        temper_sweet_point=0.9,
        temper_prob_lag=100,
        device=str(device),
    )
    env = YambdaEnvironment_GPU_HAC(env_args)

    policy_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
        sara_eta=args.sara_eta,
        sara_layer_weights=args.sara_layer_weights,
    )
    actor = SIDPolicy_credit(policy_args, env).to(device)
    hpn_loaded = load_hpn_checkpoint(actor, Path(args.hpn_checkpoint), device)

    critic_args = SimpleNamespace(
        critic_hidden_dims=args.critic_hidden_dims,
        critic_dropout_rate=args.critic_dropout_rate,
    )
    critic = Token_Critic(critic_args, env, actor).to(device)

    facade_args = SimpleNamespace(
        device=str(device),
        slate_size=args.slate_size,
        buffer_size=args.buffer_size,
        start_timestamp=args.start_timestamp,
        noise_var=0.0,
        n_iter=[args.n_iter],
        q_laplace_smoothness=0.5,
        topk_rate=1.0,
        empty_start_rate=0.0,
        item2sid=args.dense_item2sid_npy,
        candidate_ids_npy="",
        max_candidate_items=args.max_candidate_items,
        candidate_seed=args.seed,
        enable_regret_intervention=args.enable_regret_intervention,
        regret_memory_size=args.regret_memory_size,
        regret_reward_threshold=args.regret_reward_threshold,
        regret_gamma=args.regret_gamma,
        regret_phi_scale=args.regret_phi_scale,
        regret_phi_clip=args.regret_phi_clip,
    )
    facade = SIDFacade_credit(facade_args, env, actor, critic)

    agent_args = SimpleNamespace(
        device=str(device),
        gamma=args.gamma,
        n_iter=[args.n_iter],
        train_every_n_step=args.train_every_n_step,
        initial_greedy_epsilon=args.initial_greedy_epsilon,
        final_greedy_epsilon=args.final_greedy_epsilon,
        elbow_greedy=args.elbow_greedy,
        check_episode=args.check_episode,
        with_eval=False,
        save_path=args.save_path,
        use_wandb=False,
        wandb_project="yambda_sid",
        wandb_name=None,
        episode_batch_size=args.episode_batch_size,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_decay=args.actor_decay,
        critic_decay=args.critic_decay,
        target_mitigate_coef=args.target_mitigate_coef,
        entropy_coef=args.entropy_coef,
        bc_coef=args.bc_coef,
    )
    agent = DDPG(agent_args, facade)
    agent.train()

    meta = {
        "training_args": vars(args),
        "device": str(device),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "hpn_checkpoint": args.hpn_checkpoint,
        "hpn_loaded": hpn_loaded,
        "candidate_count": int(facade.candidate_iids.numel()),
        "regret_intervention": {
            "enabled": bool(args.enable_regret_intervention),
            "memory_size": int(args.regret_memory_size),
            "reward_threshold": float(args.regret_reward_threshold),
            "gamma": float(args.regret_gamma),
            "phi_scale": float(args.regret_phi_scale),
            "phi_clip": float(args.regret_phi_clip),
            "sara_eta": float(args.sara_eta),
            "sara_layer_weights": args.sara_layer_weights,
        },
        "buffer_size": int(facade.current_buffer_size),
        "n_iter": int(args.n_iter),
        "save_path": args.save_path,
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] train meta saved to {save_meta}")


if __name__ == "__main__":
    main()
