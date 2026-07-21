"""
ExGRPO experience management (arXiv:2510.02245).

Implements:
  - Module 1: bucketed question sampling buffer
  - Module 2: low-entropy trajectory selection under current policy
  - Module 3: mixed-policy mini-batch construction helpers
  - Module 4: delayed start gate
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_types import Episode


def group_episodes_by_prefix(episodes: Sequence[Episode]) -> dict[tuple[str, ...], list[Episode]]:
    groups: dict[tuple[str, ...], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.prefix,)].append(episode)
    return groups


@dataclass
class ExGRPOConfig:
    enabled: bool = False
    rho: float = 0.5
    beta: float = 0.1
    mu: float = 0.5
    sigma: float = 1.0
    activation_threshold: float = 0.35
    K: int = 8
    mix_acc_threshold: float = 0.125

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ExGRPOConfig:
        raw = raw or {}
        k = int(raw.get("K", 8))
        default_mix_thr = 1.0 / max(k, 1)
        return cls(
            enabled=bool(raw.get("enabled", False)),
            rho=float(raw.get("rho", 0.5)),
            beta=float(raw.get("beta", 0.1)),
            mu=float(raw.get("mu", 0.5)),
            sigma=float(raw.get("sigma", 1.0)),
            activation_threshold=float(raw.get("activation_threshold", 0.35)),
            K=k,
            mix_acc_threshold=float(raw.get("mix_acc_threshold", default_mix_thr)),
        )


@dataclass
class StoredTrajectory:
    """Successful trajectory stored in the replay buffer."""

    generated_token_ids: List[int]
    past_token_log_probs: List[float]
    prefix: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    text: str
    reward_info: Dict[str, float] = field(default_factory=dict)


@dataclass
class BufferQuestionMeta:
    """Metadata needed to reconstruct rollouts / rewards for a buffered question."""

    prefix: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    ground_truth: str | None = None
    numbers: List[int] | None = None
    target: int | None = None


def question_id_from_prefix(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()


EXGRPO_STATE_VERSION = 1


def _stored_trajectory_to_dict(
    traj: StoredTrajectory,
    *,
    compact: bool,
) -> dict[str, Any]:
    payload = {
        "generated_token_ids": list(traj.generated_token_ids),
        "past_token_log_probs": list(traj.past_token_log_probs),
        "prefix": traj.prefix,
        "prefix_token_ids": list(traj.prefix_token_ids),
        "reward_info": dict(traj.reward_info),
    }
    if not compact:
        payload["prefix_tokens"] = list(traj.prefix_tokens)
        payload["text"] = traj.text
    return payload


def _stored_trajectory_from_dict(raw: dict[str, Any]) -> StoredTrajectory:
    return StoredTrajectory(
        generated_token_ids=list(raw["generated_token_ids"]),
        past_token_log_probs=list(raw["past_token_log_probs"]),
        prefix=str(raw["prefix"]),
        prefix_token_ids=list(raw["prefix_token_ids"]),
        prefix_tokens=list(raw.get("prefix_tokens", [])),
        text=str(raw.get("text", "")),
        reward_info=dict(raw.get("reward_info", {})),
    )


def _buffer_meta_to_dict(meta: BufferQuestionMeta) -> dict[str, Any]:
    return {
        "prefix": meta.prefix,
        "prefix_token_ids": list(meta.prefix_token_ids),
        "prefix_tokens": list(meta.prefix_tokens),
        "ground_truth": meta.ground_truth,
        "numbers": list(meta.numbers) if meta.numbers is not None else None,
        "target": meta.target,
    }


def _buffer_meta_from_dict(raw: dict[str, Any]) -> BufferQuestionMeta:
    numbers = raw.get("numbers")
    return BufferQuestionMeta(
        prefix=str(raw["prefix"]),
        prefix_token_ids=list(raw["prefix_token_ids"]),
        prefix_tokens=list(raw.get("prefix_tokens", [])),
        ground_truth=raw.get("ground_truth"),
        numbers=list(numbers) if numbers is not None else None,
        target=raw.get("target"),
    )


def estimate_exgrpo_state_bytes(
    state: dict[str, Any] | None = None,
    *,
    num_questions: int | None = None,
    avg_gen_tokens: int = 1500,
    avg_prefix_tokens: int = 400,
    avg_prefix_chars: int = 1200,
    compact: bool = True,
) -> int:
    """Estimate serialized ExGRPO state size in bytes.

    When ``state`` is provided, measures a pickle round-trip (accurate for the
  current payload). Otherwise uses a closed-form heuristic from question counts.
    """
    if state is not None:
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    if num_questions is None:
        raise ValueError("num_questions is required when state is None")

    per_traj = (
        avg_gen_tokens * 4  # generated_token_ids (int32-ish)
        + avg_gen_tokens * 8  # past_token_log_probs (float64)
        + avg_prefix_tokens * 4  # prefix_token_ids
        + avg_prefix_chars * 2  # prefix utf-8
        + 256  # reward_info + dict overhead
    )
    if not compact:
        per_traj += avg_gen_tokens * 6 + avg_prefix_tokens * 4
    per_question = per_traj + 128  # acc_tracker + meta overhead
    retired_overhead = max(num_questions // 8, 1) * 64
    return num_questions * per_question + retired_overhead + 4096


def is_successful_rollout(episode: Episode) -> bool:
    """Binary success check (reward == 1 semantics for RLVR)."""
    if episode.reward_info.get("accuracy_correct", 0.0) > 0.0:
        return True
    if episode.reward_info.get("answer_reward", 0.0) > 0.0:
        return True
    return float(episode.reward) >= 1.0 - 1e-6


def compute_pass_at_1(episodes: Sequence[Episode]) -> float:
    """Pass@1 = fraction of questions with at least one correct rollout."""
    groups = group_episodes_by_prefix(list(episodes))
    if not groups:
        return 0.0
    passes = [any(is_successful_rollout(ep) for ep in group) for group in groups.values()]
    return float(np.mean(passes))


def _gaussian_bucket_weight(bucket_index: int, K: int, mu: float, sigma: float) -> float:
    x = bucket_index / K
    z = (x - mu) / max(sigma, 1e-8)
    return math.exp(-0.5 * z * z)


def sequential_multinomial(n: int, probs: np.ndarray) -> np.ndarray:
    """Draw counts summing to n via sequential binomial (Algorithm 2)."""
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Multinomial probabilities must sum to a positive value.")
    probs = probs / total
    d = len(probs)
    counts = np.zeros(d, dtype=int)
    remaining = n
    for i in range(d - 1):
        if remaining <= 0:
            break
        cond = probs[i] / max(probs[i:].sum(), 1e-12)
        draw = int(np.random.binomial(remaining, cond))
        counts[i] = draw
        remaining -= draw
    counts[-1] = remaining
    return counts


def _acc_to_bucket(acc: float, K: int) -> int:
    acc = min(max(acc, 0.0), 1.0)
    if acc >= 1.0:
        return K - 1
    return min(int(acc * K), K - 1)


class ExperienceBuffer:
    """Bucketed experience replay buffer (Module 1)."""

    def __init__(
        self,
        K: int = 8,
        mu: float = 0.5,
        sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.K = K
        self.mu = mu
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()
        self.buffer: Dict[str, List[StoredTrajectory]] = {}
        self.acc_tracker: Dict[str, float] = {}
        self.retired_set: Set[str] = set()
        self._meta: Dict[str, BufferQuestionMeta] = {}

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def num_trajectories(self) -> int:
        return sum(len(v) for v in self.buffer.values())

    def _store_trajectory(self, q_id: str, episode: Episode) -> None:
        traj = StoredTrajectory(
            generated_token_ids=list(episode.generated_token_ids),
            past_token_log_probs=list(episode.generated_token_log_probs),
            prefix=episode.prefix,
            prefix_token_ids=list(episode.prefix_token_ids),
            prefix_tokens=list(episode.prefix_tokens),
            text=episode.text,
            reward_info=dict(episode.reward_info),
        )
        # Keep only one replay candidate per question.
        self.buffer[q_id] = [traj]
        if q_id not in self._meta:
            self._meta[q_id] = BufferQuestionMeta(
                prefix=episode.prefix,
                prefix_token_ids=list(episode.prefix_token_ids),
                prefix_tokens=list(episode.prefix_tokens),
            )

    def collect(
        self,
        episodes: Sequence[Episode],
        *,
        is_correct: Callable[[Episode], bool] | None = None,
    ) -> dict[str, float]:
        """Collect successful trajectories and update acc / retired set."""
        is_correct = is_correct or is_successful_rollout
        groups = group_episodes_by_prefix(list(episodes))
        stats = {
            "collected_success": 0.0,
            "retired_questions": 0.0,
            "updated_questions": 0.0,
        }
        for group in groups.values():
            if not group:
                continue
            q_id = question_id_from_prefix(group[0].prefix)
            if q_id in self.retired_set:
                continue
            rewards = [1.0 if is_correct(ep) else 0.0 for ep in group]
            acc = float(sum(rewards) / self.K)
            self.acc_tracker[q_id] = acc
            stats["updated_questions"] += 1.0
            if acc >= 1.0 - 1e-8:
                self.retired_set.add(q_id)
                self.buffer.pop(q_id, None)
                self._meta.pop(q_id, None)
                stats["retired_questions"] += 1.0
                continue
            successful = [
                ep
                for ep, ok in zip(group, rewards)
                if ok > 0.0 and not ep.is_replay
            ]
            if successful:
                # Keep exactly one trajectory: latest batch's lowest-entropy success.
                best_ep = min(
                    successful,
                    key=lambda ep: _trajectory_mean_nll_from_rollout(ep.generated_token_log_probs),
                )
                self._store_trajectory(q_id, best_ep)
                stats["collected_success"] += 1.0
        return stats

    def _partition_buckets(self) -> Dict[int, List[str]]:
        buckets: Dict[int, List[str]] = {k: [] for k in range(self.K)}
        for q_id in self.buffer:
            if q_id in self.retired_set:
                continue
            acc = self.acc_tracker.get(q_id, 0.0)
            bucket = _acc_to_bucket(acc, self.K)
            buckets[bucket].append(q_id)
        return buckets

    def count_eligible_questions(self, *, min_acc: float = 0.0) -> int:
        eligible = 0
        for q_id, trajectories in self.buffer.items():
            if q_id in self.retired_set or not trajectories:
                continue
            if self.acc_tracker.get(q_id, 0.0) >= min_acc:
                eligible += 1
        return eligible

    def sample(self, n: int, *, min_acc: float = 0.0) -> List[Tuple[str, StoredTrajectory]]:
        """Bucketed multinomial sampling; returns (question_id, trajectory) pairs."""
        if n <= 0 or not self.buffer:
            return []
        buckets = self._partition_buckets()
        nonempty = {}
        for k, ids in buckets.items():
            filtered = [q_id for q_id in ids if self.acc_tracker.get(q_id, 0.0) >= min_acc]
            if filtered:
                nonempty[k] = filtered
        if not nonempty:
            return []

        bucket_ids = sorted(nonempty.keys())
        weights = np.array(
            [_gaussian_bucket_weight(k, self.K, self.mu, self.sigma) for k in bucket_ids],
            dtype=np.float64,
        )
        counts = sequential_multinomial(n, weights)
        bucket_map = {bucket_ids[i]: counts[i] for i in range(len(bucket_ids))}

        # Clip counts to bucket size and redistribute deficit.
        deficit = 0
        for k in bucket_ids:
            avail = len(nonempty[k])
            if bucket_map[k] > avail:
                deficit += bucket_map[k] - avail
                bucket_map[k] = avail
        if deficit > 0:
            for k in reversed(bucket_ids):
                room = len(nonempty[k]) - bucket_map[k]
                if room <= 0:
                    continue
                add = min(room, deficit)
                bucket_map[k] += add
                deficit -= add
                if deficit <= 0:
                    break

        sampled: List[Tuple[str, StoredTrajectory]] = []
        for k in bucket_ids:
            c_k = bucket_map[k]
            if c_k <= 0:
                continue
            q_ids = nonempty[k]
            chosen = self.rng.choice(q_ids, size=c_k, replace=False).tolist()
            for q_id in chosen:
                trajectories = self.buffer[q_id]
                traj = trajectories[int(self.rng.integers(0, len(trajectories)))]
                sampled.append((q_id, traj))
        return sampled

    def get_meta(self, q_id: str) -> BufferQuestionMeta | None:
        return self._meta.get(q_id)

    def state_dict(self, *, compact: bool = True) -> dict[str, Any]:
        return {
            "version": EXGRPO_STATE_VERSION,
            "compact": compact,
            "K": self.K,
            "mu": self.mu,
            "sigma": self.sigma,
            "rng_state": self.rng.bit_generator.state,
            "buffer": {
                q_id: [_stored_trajectory_to_dict(traj, compact=compact) for traj in trajs]
                for q_id, trajs in self.buffer.items()
            },
            "acc_tracker": dict(self.acc_tracker),
            "retired_set": sorted(self.retired_set),
            "meta": {q_id: _buffer_meta_to_dict(meta) for q_id, meta in self._meta.items()},
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        version = int(state.get("version", 0))
        if version != EXGRPO_STATE_VERSION:
            raise ValueError(
                f"Unsupported ExGRPO buffer state version {version}, "
                f"expected {EXGRPO_STATE_VERSION}"
            )
        for key in ("K", "mu", "sigma"):
            expected = getattr(self, key)
            actual = state.get(key, expected)
            if strict and actual != expected:
                raise ValueError(
                    f"ExGRPO buffer {key} mismatch on resume: "
                    f"checkpoint={actual}, current={expected}"
                )
        self.K = int(state.get("K", self.K))
        self.mu = float(state.get("mu", self.mu))
        self.sigma = float(state.get("sigma", self.sigma))
        self.rng.bit_generator.state = state["rng_state"]
        self.buffer = {
            q_id: [_stored_trajectory_from_dict(raw) for raw in trajs]
            for q_id, trajs in state.get("buffer", {}).items()
        }
        self.acc_tracker = {str(k): float(v) for k, v in state.get("acc_tracker", {}).items()}
        self.retired_set = set(state.get("retired_set", []))
        self._meta = {
            str(q_id): _buffer_meta_from_dict(raw)
            for q_id, raw in state.get("meta", {}).items()
        }


@torch.no_grad()
def compute_trajectory_mean_nll(
    model,
    prefix_token_ids: List[int],
    generated_token_ids: List[int],
    device: torch.device,
    dtype: torch.dtype,
    pad_token_id: int,
) -> float:
    """Mean per-token NLL under current policy: H(o) in ExGRPO paper."""
    if not generated_token_ids:
        return float("inf")
    token_ids = prefix_token_ids + generated_token_ids
    input_ids = torch.tensor([token_ids[:-1]], device=device, dtype=torch.long)
    targets = torch.tensor([token_ids[1:]], device=device, dtype=torch.long)
    prefix_len = len(prefix_token_ids)
    gen_start = prefix_len - 1
    gen_end = gen_start + len(generated_token_ids)
    with torch.autocast(device_type=device.type, dtype=dtype):
        logits = model.forward(input_ids).float()
    log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).reshape(1, -1)
    gen_log_probs = log_probs[0, gen_start:gen_end]
    return float(-gen_log_probs.mean().item())


def _trajectory_mean_nll_from_rollout(log_probs: Sequence[float]) -> float:
    if not log_probs:
        return float("inf")
    return float(-np.mean(np.asarray(log_probs, dtype=np.float64)))


@torch.no_grad()
def select_lowest_entropy_trajectory(
    question: str,
    candidate_trajectories: Sequence[StoredTrajectory],
    current_policy_model,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pad_token_id: int,
) -> StoredTrajectory:
    """Select trajectory with minimum mean token NLL under current policy."""
    del question  # prefix is embedded in trajectories
    if not candidate_trajectories:
        raise ValueError("candidate_trajectories must be non-empty")
    if len(candidate_trajectories) == 1:
        return candidate_trajectories[0]
    best = candidate_trajectories[0]
    best_h = float("inf")
    for traj in candidate_trajectories:
        h = compute_trajectory_mean_nll(
            current_policy_model,
            traj.prefix_token_ids,
            traj.generated_token_ids,
            device=device,
            dtype=dtype,
            pad_token_id=pad_token_id,
        )
        if h < best_h:
            best_h = h
            best = traj
    return best


def episode_from_stored_trajectory(stored: StoredTrajectory) -> Episode:
    """Build a replay Episode; past log-probs stored separately."""
    reward_info = dict(stored.reward_info)
    reward_info["is_replay"] = 1.0
    past_lps = list(stored.past_token_log_probs)
    return Episode(
        prefix=stored.prefix,
        text=stored.text,
        prefix_token_ids=list(stored.prefix_token_ids),
        prefix_tokens=list(stored.prefix_tokens),
        generated_token_ids=list(stored.generated_token_ids),
        # Keep rollout-time log-probs for metrics / fallback; IS denominator uses past_token_log_probs.
        generated_token_log_probs=list(past_lps),
        is_finished=True,
        reward=float(reward_info.get("accuracy_reward", 1.0)),
        reward_info=reward_info,
        is_replay=True,
        is_exp_group=True,
        past_token_log_probs=past_lps,
    )


def build_minibatch_from_meta(
    metas: Sequence[BufferQuestionMeta],
    template_batch: Any,
) -> Any:
    """Construct a MiniBatch-like object for buffered questions."""
    fields: dict[str, Any] = {}
    for key in dataclasses.fields(template_batch):
        if key.name == "prefix":
            fields["prefix"] = [m.prefix for m in metas]
        elif key.name == "prefix_tokens":
            fields["prefix_tokens"] = [m.prefix_tokens for m in metas]
        elif key.name == "prefix_token_ids":
            fields["prefix_token_ids"] = [m.prefix_token_ids for m in metas]
        elif key.name == "ground_truth":
            fields["ground_truth"] = [m.ground_truth or "" for m in metas]
        elif key.name == "numbers":
            fields["numbers"] = [m.numbers or [] for m in metas]
        elif key.name == "target":
            fields["target"] = [m.target or 0 for m in metas]
        elif key.name == "problem":
            fields["problem"] = [m.prefix for m in metas]
        elif key.name == "prompt":
            fields["prompt"] = [m.prefix for m in metas]
        else:
            value = getattr(template_batch, key.name)
            if isinstance(value, list):
                fields[key.name] = value[: len(metas)]
            else:
                fields[key.name] = value
    return dataclasses.replace(template_batch, **fields)


class ExGRPOManager:
    """Orchestrates mixed batch construction and delayed activation (Modules 3 & 4)."""

    def __init__(self, config: ExGRPOConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.buffer = ExperienceBuffer(
            K=config.K,
            mu=config.mu,
            sigma=config.sigma,
            rng=rng,
        )
        self._activated = False

    @property
    def activated(self) -> bool:
        return self._activated and self.config.enabled

    def should_activate_exgrpo(
        self,
        current_batch_pass_at_1: float,
        threshold: float | None = None,
    ) -> bool:
        if not self.config.enabled:
            return False
        if self._activated:
            return True
        thr = self.config.activation_threshold if threshold is None else threshold
        if current_batch_pass_at_1 >= thr:
            self._activated = True
        return self._activated

    def build_mixed_batch_plan(self, batch_size: int) -> Tuple[int, int]:
        """Return (n_exp, n_on) question counts."""
        if not self.activated or len(self.buffer) == 0:
            return 0, batch_size
        n_exp = min(int(math.floor(self.config.rho * batch_size)), batch_size)
        return n_exp, batch_size - n_exp

    def select_experience_candidates_from_batch(
        self,
        batch_prefixes: Sequence[str],
        n_exp: int,
    ) -> List[Tuple[int, str, float]]:
        """Sample candidate exp questions from *current batch* by historical acc.

        Returns tuples:
          (batch_question_index, question_id, historical_acc)
        """
        if n_exp <= 0 or not batch_prefixes:
            return []
        candidates: list[tuple[int, str, float, float]] = []
        for idx, prefix in enumerate(batch_prefixes):
            q_id = question_id_from_prefix(prefix)
            acc = float(self.buffer.acc_tracker.get(q_id, 0.0))
            bucket = _acc_to_bucket(acc, self.config.K)
            weight = _gaussian_bucket_weight(
                bucket,
                self.config.K,
                self.config.mu,
                self.config.sigma,
            )
            candidates.append((idx, q_id, acc, weight))
        pick_n = min(n_exp, len(candidates))
        probs = np.asarray([item[3] for item in candidates], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total
        chosen = self.buffer.rng.choice(
            len(candidates), size=pick_n, replace=False, p=probs
        ).tolist()
        selected = [
            (candidates[i][0], candidates[i][1], candidates[i][2])
            for i in chosen
        ]
        selected.sort(key=lambda x: x[0])
        return selected

    def build_replay_for_question_ids(
        self,
        question_ids: Sequence[str],
        model,
        *,
        device: torch.device,
        dtype: torch.dtype,
        pad_token_id: int,
    ) -> Dict[str, Tuple[StoredTrajectory, Episode]]:
        """Resolve one replay trajectory per question id."""
        out: Dict[str, Tuple[StoredTrajectory, Episode]] = {}
        for q_id in question_ids:
            candidates = self.buffer.buffer.get(q_id, [])
            if not candidates:
                continue
            best = select_lowest_entropy_trajectory(
                candidates[0].prefix,
                candidates,
                model,
                device=device,
                dtype=dtype,
                pad_token_id=pad_token_id,
            )
            out[q_id] = (best, episode_from_stored_trajectory(best))
        return out

    def sample_experience_questions(
        self,
        n_exp: int,
        model,
        *,
        device: torch.device,
        dtype: torch.dtype,
        pad_token_id: int,
    ) -> List[Tuple[str, StoredTrajectory, Episode]]:
        """Sample questions and apply low-entropy trajectory selection."""
        if n_exp <= 0:
            return []
        raw = self.buffer.sample(n_exp, min_acc=self.config.mix_acc_threshold)
        selected: List[Tuple[str, StoredTrajectory, Episode]] = []
        for q_id, _ in raw:
            candidates = self.buffer.buffer.get(q_id, [])
            if not candidates:
                continue
            best = select_lowest_entropy_trajectory(
                candidates[0].prefix,
                candidates,
                model,
                device=device,
                dtype=dtype,
                pad_token_id=pad_token_id,
            )
            selected.append((q_id, best, episode_from_stored_trajectory(best)))
        return selected

    def enrich_meta_from_batch(self, batch: Any) -> None:
        """Attach dataset fields (ground_truth, etc.) to buffered questions."""
        if not hasattr(batch, "prefix"):
            return
        for i, prefix in enumerate(batch.prefix):
            q_id = question_id_from_prefix(prefix)
            if q_id not in self.buffer.buffer:
                continue
            meta = self.buffer._meta.setdefault(
                q_id,
                BufferQuestionMeta(
                    prefix=prefix,
                    prefix_token_ids=list(batch.prefix_token_ids[i]),
                    prefix_tokens=list(batch.prefix_tokens[i]),
                ),
            )
            if hasattr(batch, "ground_truth"):
                meta.ground_truth = batch.ground_truth[i]
            if hasattr(batch, "numbers"):
                meta.numbers = list(batch.numbers[i])
            if hasattr(batch, "target"):
                meta.target = batch.target[i]

    def collect_from_rollouts(self, episodes: Sequence[Episode]) -> dict[str, float]:
        return self.buffer.collect(episodes)

    def state_dict(self, *, compact: bool = True) -> dict[str, Any]:
        return {
            "version": EXGRPO_STATE_VERSION,
            "activated": self._activated,
            "config": dataclasses.asdict(self.config),
            "buffer": self.buffer.state_dict(compact=compact),
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        version = int(state.get("version", 0))
        if version != EXGRPO_STATE_VERSION:
            raise ValueError(
                f"Unsupported ExGRPO manager state version {version}, "
                f"expected {EXGRPO_STATE_VERSION}"
            )
        saved_cfg = state.get("config", {})
        for key in ("K", "rho", "beta", "mix_acc_threshold", "activation_threshold"):
            if key not in saved_cfg:
                continue
            expected = getattr(self.config, key)
            actual = saved_cfg[key]
            if strict and actual != expected:
                raise ValueError(
                    f"ExGRPO config.{key} mismatch on resume: "
                    f"checkpoint={actual}, current={expected}"
                )
        self._activated = bool(state.get("activated", False))
        self.buffer.load_state_dict(state["buffer"], strict=strict)


def format_exgrpo_storage_summary(
    manager: ExGRPOManager | None = None,
    *,
    state: dict[str, Any] | None = None,
) -> str:
    """Human-readable storage estimate for logging."""
    if state is None:
        if manager is None:
            raise ValueError("manager or state must be provided")
        state = manager.state_dict()
    nbytes = estimate_exgrpo_state_bytes(state)
    buf = state.get("buffer", {})
    num_q = len(buf.get("buffer", {}))
    num_traj = sum(len(v) for v in buf.get("buffer", {}).values())
    retired = len(buf.get("retired_set", []))
    activated = bool(state.get("activated", False))
    mib = nbytes / (1024 * 1024)
    return (
        f"ExGRPO state ~{mib:.1f} MiB "
        f"(questions={num_q}, trajectories={num_traj}, retired={retired}, "
        f"activated={activated}, compact={buf.get('compact', True)})"
    )


def merge_replay_and_fresh_rollouts(
    replay_episodes: Sequence[Episode],
    fresh_episodes: Sequence[Episode],
) -> List[Episode]:
    """Form mixed advantage groups G_q* = {o*} ∪ {o_1, ..., o_{K-1}}."""
    replay_map = {ep.prefix: ep for ep in replay_episodes}
    groups = group_episodes_by_prefix(list(fresh_episodes))
    merged: List[Episode] = []
    for group in groups.values():
        if not group:
            continue
        replay = replay_map.get(group[0].prefix)
        group_eps: list[Episode] = []
        if replay is not None:
            group_eps.append(replay)
        group_eps.extend(group)
        for ep in group_eps:
            merged.append(dataclasses.replace(ep, is_exp_group=True))
    return merged


def compute_exgrpo_episode_weights(
    episodes: Sequence[Episode],
    rho: float,
) -> tuple[list[float], dict[str, float]]:
    """Per-episode weights for ExGRPO Eq. 4 (Dr.GRPO token sum + group/question reweighting).

    J = (1-ρ) E_{q~B_on}[1/K Σ CLIP] + ρ E_{q*~B_exp}[1/K (f(w*) + Σ CLIP)]

    Each trajectory in an on-policy group gets weight (1-ρ) / (n_on · |G|).
    Each trajectory in an experiential group gets weight ρ / (n_exp · |G|).
    Summed over all trajectories the weights total 1.0 when both splits are non-empty.
    """
    groups_dict = group_episodes_by_prefix(episodes)
    on_groups: list[list[Episode]] = []
    exp_groups: list[list[Episode]] = []
    for group in groups_dict.values():
        if any(ep.is_exp_group for ep in group):
            exp_groups.append(group)
        else:
            on_groups.append(group)

    n_on = len(on_groups)
    n_exp = len(exp_groups)
    weight_by_prefix: dict[str, float] = {}

    for group in on_groups:
        gsize = max(len(group), 1)
        w = (1.0 - rho) / max(n_on, 1) / gsize
        for ep in group:
            weight_by_prefix[ep.prefix] = w

    for group in exp_groups:
        gsize = max(len(group), 1)
        w = rho / max(n_exp, 1) / gsize
        for ep in group:
            weight_by_prefix[ep.prefix] = w

    weights = [weight_by_prefix[ep.prefix] for ep in episodes]
    on_weight_sum = (1.0 - rho) if n_on > 0 else 0.0
    exp_weight_sum = rho if n_exp > 0 else 0.0
    stats = {
        "exgrpo_n_on_groups": float(n_on),
        "exgrpo_n_exp_groups": float(n_exp),
        "exgrpo_on_weight_sum": on_weight_sum,
        "exgrpo_exp_weight_sum": exp_weight_sum,
        "exgrpo_weight_total": on_weight_sum + exp_weight_sum,
    }
    return weights, stats


def build_group_preserving_micro_batches(
    episodes: Sequence[Episode],
    micro_batch_size: int,
) -> list[list[Episode]]:
    """Pack whole advantage groups into micro-batches (required for ExGRPO weighting)."""
    groups_dict = group_episodes_by_prefix(episodes)
    sorted_groups = sorted(
        groups_dict.values(),
        key=lambda g: max(
            len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in g
        ),
    )
    batches: list[list[Episode]] = []
    current: list[Episode] = []
    for group in sorted_groups:
        if current and len(current) + len(group) > micro_batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches
