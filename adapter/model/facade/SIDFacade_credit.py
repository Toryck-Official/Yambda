import pickle

import numpy as np
import torch
import utils

from _loader import load_hsrl_module

_orig = load_hsrl_module("model/facade/SIDFacade_credit.py", "_hsrl_original_sid_facade")


class SIDFacade_credit(_orig.SIDFacade_credit):
    """
    Yambda 版 SID facade 覆盖层。

    核心变化：
    - 支持 dense_item2sid.npy
    - 支持 feature-cache 候选池，避免全量 catalog 打分
    - replay buffer 显式存储 history_features，避免 padding id 索引错误
    """

    @staticmethod
    def parse_model_args(parser):
        parser = _orig.SIDFacade_credit.parse_model_args(parser)
        parser.add_argument("--candidate_ids_npy", type=str, default="", help="Optional dense item ids for candidates")
        parser.add_argument("--max_candidate_items", type=int, default=0, help="If >0, sample/truncate candidate pool")
        parser.add_argument("--candidate_seed", type=int, default=42, help="Random seed for candidate sampling")
        parser.add_argument("--enable_regret_intervention", action="store_true", help="Apply episode-local regret memory to SID logits")
        parser.add_argument("--regret_memory_size", type=int, default=20, help="Number of failed SID paths kept per active slot")
        parser.add_argument("--regret_reward_threshold", type=float, default=0.0, help="Reward below this value enters regret memory")
        parser.add_argument("--regret_gamma", type=float, default=0.9, help="Per-step decay for stored regret strengths")
        parser.add_argument("--regret_phi_scale", type=float, default=1.0, help="Scale for converting negative reward to regret strength")
        parser.add_argument("--regret_phi_clip", type=float, default=1.5, help="Upper clip for regret strength")
        return parser

    def __init__(self, args, environment, actor, critic):
        super(_orig.SIDFacade_credit, self).__init__()
        self.device = args.device
        self.env = environment
        self.actor = actor
        self.critic = critic
        self.slate_size = args.slate_size
        self.noise_var = args.noise_var
        n_iter = getattr(args, "n_iter", [1])
        self.noise_decay = args.noise_var / max(n_iter[-1], 1)
        self.q_laplace_smoothness = args.q_laplace_smoothness
        self.topk_rate = args.topk_rate
        self.empty_start_rate = args.empty_start_rate

        print(f"Note!! item2sid path is {args.item2sid}")
        self.item2sid = self._load_item2sid(args.item2sid)
        self.n_item = self.env.action_space["item_id"][1]
        self.candidate_iids, self.candidate_features = self._load_candidate_pool(args)
        self.buffer_size = args.buffer_size
        self.start_timestamp = args.start_timestamp
        self.enable_regret_intervention = bool(getattr(args, "enable_regret_intervention", False))
        self.regret_memory_size = int(getattr(args, "regret_memory_size", 20))
        self.regret_reward_threshold = float(getattr(args, "regret_reward_threshold", 0.0))
        self.regret_gamma = float(getattr(args, "regret_gamma", 0.9))
        self.regret_phi_scale = float(getattr(args, "regret_phi_scale", 1.0))
        self.regret_phi_clip = float(getattr(args, "regret_phi_clip", 1.5))
        self.regret_pool_tokens = None
        self.regret_pool_phis = None
        if self.enable_regret_intervention:
            print(
                "[SIDFacade] regret_intervention=on "
                f"memory={self.regret_memory_size} threshold={self.regret_reward_threshold:.3f} "
                f"gamma={self.regret_gamma:.3f}"
            )

    def _load_item2sid(self, item2sid_path):
        """输入：item2sid 路径。输出：dense item id -> SID 查表。"""
        if item2sid_path is None or item2sid_path == "":
            return None
        if str(item2sid_path).endswith(".npy"):
            return np.load(item2sid_path, mmap_mode="r")
        with open(item2sid_path, "rb") as f:
            return pickle.load(f)

    def _load_candidate_pool(self, args):
        """输入：训练参数。输出：候选 item ids 和候选 item features。"""
        candidate_ids_npy = getattr(args, "candidate_ids_npy", "")
        max_candidate_items = int(getattr(args, "max_candidate_items", 0))
        candidate_seed = int(getattr(args, "candidate_seed", 42))

        if candidate_ids_npy:
            candidate_iids = np.load(candidate_ids_npy).astype(np.int64)
            candidate_features = self.env.reader.get_item_list_meta(candidate_iids).astype(np.float32)
            source = candidate_ids_npy
        elif hasattr(self.env.reader, "dense_ids") and hasattr(self.env.reader, "features"):
            candidate_iids = self.env.reader.dense_ids.astype(np.int64)
            candidate_features = self.env.reader.features.astype(np.float32)
            source = "reader.feature_cache"
        else:
            if self.n_item > 1000000 and max_candidate_items <= 0:
                raise RuntimeError(
                    "Yambda candidate pool is too large for full-catalog scoring. "
                    "Pass --candidate_ids_npy / --max_candidate_items or use YambdaFeatureCacheReader."
                )
            candidate_iids = np.arange(1, self.n_item + 1, dtype=np.int64)
            source = "full_catalog"
            if max_candidate_items > 0:
                candidate_iids = candidate_iids[:max_candidate_items]
            candidate_features = self.env.reader.get_item_list_meta(candidate_iids).astype(np.float32)

        if max_candidate_items > 0 and len(candidate_iids) > max_candidate_items:
            rng = np.random.default_rng(candidate_seed)
            keep_idx = np.sort(rng.choice(len(candidate_iids), size=max_candidate_items, replace=False))
            candidate_iids = candidate_iids[keep_idx]
            candidate_features = candidate_features[keep_idx]

        print(f"[SIDFacade] candidate_source={source}, n_candidate={len(candidate_iids):,}")
        return (
            torch.tensor(candidate_iids, dtype=torch.long, device=self.device),
            torch.tensor(candidate_features, dtype=torch.float32, device=self.device),
        )

    def _lookup_sid_tokens(self, item_ids, n_levels):
        """输入：dense item ids tensor。输出：对应 SID token tensor。"""
        if self.item2sid is None:
            raise RuntimeError("SIDFacade requires item2sid for semantic action scoring.")

        if isinstance(self.item2sid, np.ndarray):
            item_ids_np = item_ids.detach().cpu().numpy()
            sid_np = self.item2sid[item_ids_np, :n_levels]
            return torch.as_tensor(sid_np, dtype=torch.long, device=self.device)

        flat_ids = item_ids.detach().cpu().view(-1).tolist()
        sid_arr = np.zeros((len(flat_ids), n_levels), dtype=np.int64)
        for idx, iid in enumerate(flat_ids):
            sid = self.item2sid.get(int(iid), [0] * n_levels)
            sid_arr[idx, :] = np.asarray(list(sid)[:n_levels], dtype=np.int64)
        sid_arr = sid_arr.reshape(*item_ids.shape, n_levels)
        return torch.as_tensor(sid_arr, dtype=torch.long, device=self.device)

    @staticmethod
    def _gather_candidate_features(cand_feats, indices):
        """输入 cand_feats=[B,N,D], indices=[B,K]。输出 action_features=[B,K,D]。"""
        gather_index = indices.unsqueeze(-1).expand(-1, -1, cand_feats.size(-1))
        return torch.gather(cand_feats, 1, gather_index).detach()

    def _reset_regret_memory(self, batch_size, n_level):
        """Reset episode-local regret memory for active rollout slots."""
        if not self.enable_regret_intervention:
            return
        self.regret_pool_tokens = torch.zeros(
            batch_size,
            self.regret_memory_size,
            n_level,
            dtype=torch.long,
            device=self.device,
        )
        self.regret_pool_phis = torch.zeros(
            batch_size,
            self.regret_memory_size,
            dtype=torch.float32,
            device=self.device,
        )

    def _ensure_regret_memory(self, batch_size, n_level):
        if not self.enable_regret_intervention:
            return
        if (
            self.regret_pool_tokens is None
            or self.regret_pool_phis is None
            or self.regret_pool_tokens.shape != (batch_size, self.regret_memory_size, n_level)
        ):
            self._reset_regret_memory(batch_size, n_level)

    def _can_apply_regret_intervention(self, observation, batch_size, n_level):
        if not self.enable_regret_intervention:
            return False
        # Replay-buffer samples do not carry stable online slots; only live
        # rollout observations have `history`, so intervention stays online.
        if "history" not in observation:
            return False
        self._ensure_regret_memory(batch_size, n_level)
        return self.regret_pool_tokens is not None and self.regret_pool_phis is not None

    def _update_regret_memory(self, policy_output, reward, done_mask):
        if not self.enable_regret_intervention or self.regret_memory_size <= 0:
            return
        if "sid_tokens" not in policy_output:
            return
        sid_tokens = policy_output["sid_tokens"].detach().to(self.device).long()
        if sid_tokens.dim() != 3:
            return
        batch_size, slate_size, n_level = sid_tokens.shape
        self._ensure_regret_memory(batch_size, n_level)
        if self.regret_pool_tokens is None or self.regret_pool_phis is None:
            return

        reward = reward.detach().to(self.device).float().view(-1)
        done_mask = done_mask.detach().to(self.device).bool().view(-1)
        self.regret_pool_phis.mul_(self.regret_gamma)
        self.regret_pool_phis.masked_fill_(self.regret_pool_phis < 1e-6, 0.0)

        fail_mask = reward < self.regret_reward_threshold
        if bool(fail_mask.any()):
            phi = ((self.regret_reward_threshold - reward) * self.regret_phi_scale).clamp(
                min=0.0,
                max=self.regret_phi_clip,
            )
            for row_idx in torch.nonzero(fail_mask, as_tuple=False).flatten().tolist():
                n_insert = min(int(slate_size), self.regret_memory_size)
                if n_insert <= 0:
                    continue
                old_tokens = self.regret_pool_tokens[row_idx].clone()
                old_phis = self.regret_pool_phis[row_idx].clone()
                if n_insert < self.regret_memory_size:
                    self.regret_pool_tokens[row_idx, n_insert:] = old_tokens[:-n_insert]
                    self.regret_pool_phis[row_idx, n_insert:] = old_phis[:-n_insert]
                self.regret_pool_tokens[row_idx, :n_insert] = sid_tokens[row_idx, :n_insert]
                self.regret_pool_phis[row_idx, :n_insert] = phi[row_idx]

        if bool(done_mask.any()):
            self.regret_pool_tokens[done_mask] = 0
            self.regret_pool_phis[done_mask] = 0.0

    def reset_env(self, initial_params={"batch_size": 1}):
        """Reset environment and clear episode-local regret memory."""
        initial_params["empty_history"] = True if np.random.rand() < self.empty_start_rate else False
        observation = self.env.reset(initial_params)
        if self.enable_regret_intervention:
            batch_size = int(observation["history_features"].shape[0])
            self._reset_regret_memory(batch_size, self.actor.sid_levels)
        return observation

    def initialize_train(self):
        """输入：无。输出：初始化 replay buffer。"""
        portrait_dim = max(int(getattr(self.env.reader, "portrait_len", 0)), 1)
        item_dim = int(getattr(self.env.reader, "item_vec_size", self.env.action_space["item_feature"][1]))
        self.buffer = {
            "user_profile": torch.zeros(self.buffer_size, portrait_dim),
            "history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "history_features": torch.zeros(self.buffer_size, self.env.reader.max_seq_len, item_dim),
            "next_history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "next_history_features": torch.zeros(self.buffer_size, self.env.reader.max_seq_len, item_dim),
            "context_list": torch.zeros(self.buffer_size, self.actor.sid_levels + 1, self.actor.state_dim, dtype=torch.float32),
            "action": torch.zeros(self.buffer_size, self.slate_size, dtype=torch.long),
            "reward": torch.zeros(self.buffer_size),
            "feedback": torch.zeros(self.buffer_size, self.slate_size),
            "done": torch.zeros(self.buffer_size, dtype=torch.bool),
            "sid_tokens": torch.zeros(self.buffer_size, self.slate_size, self.actor.sid_levels, dtype=torch.long),
        }
        for key, value in self.buffer.items():
            self.buffer[key] = value.to(self.device)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False

    def apply_critic(self, observation, policy_output, critic_model):
        """输入：observation/policy_output/critic。输出：critic q/value。"""
        if "context_list" not in policy_output:
            raise KeyError("Token_Critic requires policy_output['context_list'].")
        return critic_model({"context_list": policy_output["context_list"]})

    def apply_policy(self, observation, policy_model, epsilon=0.0, do_explore=False, do_softmax=True):
        """在候选集内用 SID token 概率打分并选择 item。"""
        feed_dict = observation
        batch_size_hint = int(feed_dict["history_features"].shape[0])
        n_level_hint = int(getattr(policy_model, "sid_levels", self.actor.sid_levels))
        if self._can_apply_regret_intervention(feed_dict, batch_size_hint, n_level_hint) and hasattr(policy_model, "get_sara_logits"):
            out_dict = policy_model.get_sara_logits(
                feed_dict,
                pool_tokens=self.regret_pool_tokens,
                pool_phis=self.regret_pool_phis,
            )
            out_dict["regret_intervention_applied"] = True
            out_dict["regret_memory_active"] = (self.regret_pool_phis > 0).float().sum(dim=1).detach()
        else:
            out_dict = policy_model(feed_dict)
            out_dict["regret_intervention_applied"] = False
        assert "sid_logits" in out_dict, "SIDPolicy 必须输出 'sid_logits'"
        sid_logits_list = out_dict["sid_logits"]
        batch_size = sid_logits_list[0].size(0)
        n_level = len(sid_logits_list)

        if "candidate_ids" in feed_dict:
            cand_ids = feed_dict["candidate_ids"]
            cand_feats = feed_dict["candidate_features"]
            if isinstance(cand_ids, torch.Tensor) and cand_ids.dim() == 1:
                cand_ids = cand_ids.unsqueeze(0).repeat(batch_size, 1)
            cand_ids = cand_ids.to(self.device)
            cand_feats = cand_feats.to(self.device)
        else:
            cand_ids = self.candidate_iids.unsqueeze(0).repeat(batch_size, 1)
            cand_feats = self.candidate_features.unsqueeze(0)
        if cand_feats.dim() == 2:
            cand_feats = cand_feats.unsqueeze(0)
        if cand_feats.size(0) == 1 and batch_size > 1:
            cand_feats = cand_feats.expand(batch_size, -1, -1)

        cand_sid = self._lookup_sid_tokens(cand_ids, n_level)
        level_probs = [torch.softmax(logits, dim=-1) for logits in sid_logits_list]
        candidate_prob = torch.ones_like(cand_ids, dtype=level_probs[0].dtype)

        for level in range(n_level):
            idx_l = cand_sid[..., level]
            pl = level_probs[level].gather(1, idx_l)
            candidate_prob = candidate_prob * pl
        candidate_prob = candidate_prob / (candidate_prob.sum(dim=1, keepdim=True) + 1e-12)

        if do_explore and epsilon > 0:
            candidate_prob = (1 - epsilon) * candidate_prob + epsilon * (1.0 / candidate_prob.size(1))

        if np.random.rand() >= self.topk_rate:
            action, indices = utils.sample_categorical_action(
                candidate_prob, cand_ids, self.slate_size, with_replacement=False, batch_wise=True, return_idx=True
            )
        else:
            _, indices = torch.topk(candidate_prob, k=self.slate_size, dim=1)
            action = torch.gather(cand_ids, 1, indices).detach()

        out_dict["action"] = action
        out_dict["action_features"] = self._gather_candidate_features(cand_feats, indices)
        out_dict["action_prob"] = torch.gather(candidate_prob, 1, indices)
        out_dict["candidate_prob"] = candidate_prob
        out_dict["sid_tokens"] = self._lookup_sid_tokens(action, n_level)
        return out_dict

    def sample_buffer(self, batch_size):
        """输入 batch_size。输出 DDPG 训练所需 transition batch。"""
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)
        user_profile, history, next_history, context, sid, action, reward, feedback, done = self.read_buffer(indices)
        observation = {"user_profile": user_profile, "history_features": history}
        policy_output = {"context_list": context, "action": action, "sid_tokens": sid}
        next_observation = {"user_profile": user_profile, "history_features": next_history, "previous_feedback": feedback}
        return observation, policy_output, reward, done, next_observation, feedback

    def extract_behavior_data(self, observation, policy_output, next_observation):
        """输入 transition。输出 BC 需要的 exposure / feedback。"""
        observation = {"user_profile": observation["user_profile"], "history_features": observation["history_features"]}
        exposed_items = policy_output["action"]
        exposure = {"ids": exposed_items, "features": policy_output.get("action_features")}
        if exposure["features"] is None:
            flat = exposed_items.detach().cpu().view(-1).tolist()
            features = self.env.reader.get_item_list_meta(flat)
            exposure["features"] = torch.tensor(features, dtype=torch.float32, device=self.device).view(*exposed_items.shape, -1)
        user_feedback = next_observation["previous_feedback"]
        return observation, exposure, user_feedback

    def update_buffer(self, observation, policy_output, reward, done_mask, next_observation, info):
        """输入环境 transition。输出：写入 replay buffer。"""
        if self.buffer_head + reward.shape[0] >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + [i for i in range(reward.shape[0] - tail)]
        else:
            indices = [self.buffer_head + i for i in range(reward.shape[0])]

        self.buffer["user_profile"][indices] = observation["user_profile"]
        self.buffer["history"][indices] = observation["history"]
        self.buffer["history_features"][indices] = observation["history_features"]
        self.buffer["next_history"][indices] = next_observation["history"]
        self.buffer["next_history_features"][indices] = next_observation["history_features"]
        self.buffer["context_list"][indices] = policy_output["context_list"]
        self.buffer["action"][indices] = policy_output["action"]
        self.buffer["sid_tokens"][indices] = policy_output["sid_tokens"]
        self.buffer["reward"][indices] = reward
        self.buffer["feedback"][indices] = info["response"]
        self.buffer["done"][indices] = done_mask
        self._update_regret_memory(policy_output, reward, done_mask)

        self.buffer_head = (self.buffer_head + reward.shape[0]) % self.buffer_size
        self.n_stream_record += reward.shape[0]
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        if self.n_stream_record >= self.start_timestamp:
            self.is_training_available = True

    def read_buffer(self, indices):
        """输入 buffer indices。输出 transition tuple。"""
        user_profile = self.buffer["user_profile"][indices]
        history = self.buffer["history_features"][indices]
        next_history = self.buffer["next_history_features"][indices]
        context = self.buffer["context_list"][indices]
        sid = self.buffer["sid_tokens"][indices]
        action = self.buffer["action"][indices]
        reward = self.buffer["reward"][indices]
        feedback = self.buffer["feedback"][indices]
        done = self.buffer["done"][indices]
        return user_profile, history, next_history, context, sid, action, reward, feedback, done
