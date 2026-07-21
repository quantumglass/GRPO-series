"""Unit tests for ExGRPO experience buffer and sampling."""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from checkpoint import build_lora_checkpoint_payload
from grpo import compute_exgrpo_gspo_loss, compute_gspo_loss

from data_types import Episode
from exgrpo import (
    ExperienceBuffer,
    ExGRPOConfig,
    ExGRPOManager,
    StoredTrajectory,
    _acc_to_bucket,
    compute_exgrpo_episode_weights,
    episode_from_stored_trajectory,
    estimate_exgrpo_state_bytes,
    format_exgrpo_storage_summary,
    is_successful_rollout,
    merge_replay_and_fresh_rollouts,
    question_id_from_prefix,
    sequential_multinomial,
)


def _make_episode(prefix: str, reward: float, *, is_replay: bool = False) -> Episode:
    return Episode(
        prefix=prefix,
        text=prefix + " answer",
        prefix_token_ids=[1, 2, 3],
        prefix_tokens=["a", "b", "c"],
        generated_token_ids=[10, 11],
        generated_token_log_probs=[-0.5, -0.6],
        is_finished=True,
        reward=reward,
        reward_info={"accuracy_correct": float(reward >= 1.0)},
        is_replay=is_replay,
        past_token_log_probs=[-0.4, -0.5] if is_replay else None,
    )


class TestExperienceBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.buffer = ExperienceBuffer(K=8, mu=0.5, sigma=1.0, rng=self.rng)

    def test_collect_stores_only_successful(self) -> None:
        prefix = "question-a"
        episodes = [
            _make_episode(prefix, 1.0),
            _make_episode(prefix, 0.0),
            dataclasses.replace(
                _make_episode(prefix, 1.0),
                generated_token_ids=[12, 13],
                generated_token_log_probs=[-0.1, -0.1],
            ),
        ]
        # Pad to K=8 semantics via repeated failures in acc denominator
        while len(episodes) < 8:
            episodes.append(_make_episode(prefix, 0.0))
        stats = self.buffer.collect(episodes[:8])
        q_id = question_id_from_prefix(prefix)
        self.assertEqual(stats["collected_success"], 1.0)
        self.assertEqual(len(self.buffer.buffer[q_id]), 1)
        self.assertEqual(self.buffer.buffer[q_id][0].generated_token_ids, [12, 13])
        self.assertAlmostEqual(self.buffer.acc_tracker[q_id], 2 / 8)

    def test_retire_fully_correct_questions(self) -> None:
        prefix = "question-retire"
        episodes = [_make_episode(prefix, 1.0) for _ in range(8)]
        self.buffer.collect(episodes)
        q_id = question_id_from_prefix(prefix)
        self.assertIn(q_id, self.buffer.retired_set)
        self.assertNotIn(q_id, self.buffer.buffer)

    def test_retired_never_re_enter(self) -> None:
        prefix = "question-retire-2"
        episodes = [_make_episode(prefix, 1.0) for _ in range(8)]
        self.buffer.collect(episodes)
        q_id = question_id_from_prefix(prefix)
        partial = [_make_episode(prefix, 1.0)] + [_make_episode(prefix, 0.0) for _ in range(7)]
        self.buffer.collect(partial)
        self.assertNotIn(q_id, self.buffer.buffer)

    def test_bucket_partition(self) -> None:
        for acc, expected in [(0.0, 0), (0.125, 1), (0.875, 7), (0.99, 7)]:
            self.assertEqual(_acc_to_bucket(acc, 8), expected)

    def test_sample_respects_bucket_counts(self) -> None:
        for i in range(8):
            prefix = f"q-{i}"
            acc = (i + 0.5) / 8
            q_id = question_id_from_prefix(prefix)
            self.buffer.buffer[q_id] = [
                dataclasses.replace(
                    _make_episode(prefix, 1.0),
                    past_token_log_probs=[-0.1],
                )
            ]
            self.buffer.acc_tracker[q_id] = acc
        samples = self.buffer.sample(16)
        self.assertEqual(len(samples), 8)

    def test_sequential_multinomial_sums_to_n(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        for n in [0, 1, 5, 20]:
            counts = sequential_multinomial(n, probs)
            self.assertEqual(int(counts.sum()), n)


class TestExGRPOManager(unittest.TestCase):
    def test_delayed_activation(self) -> None:
        cfg = ExGRPOConfig(enabled=True, activation_threshold=0.35)
        mgr = ExGRPOManager(cfg)
        self.assertFalse(mgr.should_activate_exgrpo(0.2))
        self.assertFalse(mgr.activated)
        self.assertTrue(mgr.should_activate_exgrpo(0.4))
        self.assertTrue(mgr.activated)

    def test_build_mixed_batch_plan(self) -> None:
        cfg = ExGRPOConfig(enabled=True, rho=0.5)
        mgr = ExGRPOManager(cfg)
        mgr._activated = True
        n_exp, n_on = mgr.build_mixed_batch_plan(64)
        self.assertEqual(n_exp, 0)
        self.assertEqual(n_on, 64)
        for i in range(40):
            p = f"q-buffer-{i}"
            q_id = question_id_from_prefix(p)
            mgr.buffer.buffer[q_id] = [_make_episode(p, 1.0)]
            mgr.buffer.acc_tracker[q_id] = 0.5
        n_exp, n_on = mgr.build_mixed_batch_plan(64)
        self.assertEqual(n_exp, 32)
        self.assertEqual(n_on, 32)

    def test_build_mixed_batch_plan_respects_mix_acc_threshold(self) -> None:
        cfg = ExGRPOConfig(enabled=True, rho=0.5, mix_acc_threshold=0.9)
        mgr = ExGRPOManager(cfg)
        mgr._activated = True
        low_q = question_id_from_prefix("q-low")
        high_q = question_id_from_prefix("q-high")
        mgr.buffer.buffer[low_q] = [_make_episode("q-low", 1.0)]
        mgr.buffer.buffer[high_q] = [_make_episode("q-high", 1.0)]
        mgr.buffer.acc_tracker[low_q] = 0.25
        mgr.buffer.acc_tracker[high_q] = 0.75
        n_exp, n_on = mgr.build_mixed_batch_plan(64)
        self.assertEqual(n_exp, 32)
        self.assertEqual(n_on, 32)

    def test_select_experience_candidates_from_batch(self) -> None:
        cfg = ExGRPOConfig(enabled=True, rho=0.5)
        mgr = ExGRPOManager(cfg, rng=np.random.default_rng(0))
        prefixes = ["q-a", "q-b", "q-c", "q-d"]
        for i, p in enumerate(prefixes):
            q_id = question_id_from_prefix(p)
            mgr.buffer.acc_tracker[q_id] = i / 4
        picked = mgr.select_experience_candidates_from_batch(prefixes, n_exp=2)
        self.assertEqual(len(picked), 2)
        for idx, q_id, _ in picked:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(prefixes))
            self.assertEqual(q_id, question_id_from_prefix(prefixes[idx]))

class TestExGRPOEpisodeWeights(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        on_prefix = "on-q"
        exp_prefix = "exp-q"
        episodes = [_make_episode(on_prefix, 1.0) for _ in range(8)]
        replay = _make_episode(exp_prefix, 1.0, is_replay=True)
        fresh = [_make_episode(exp_prefix, 0.0) for _ in range(7)]
        merged = merge_replay_and_fresh_rollouts([replay], fresh)
        episodes.extend(merged)
        weights, stats = compute_exgrpo_episode_weights(episodes, rho=0.5)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertAlmostEqual(stats["exgrpo_on_weight_sum"], 0.5, places=6)
        self.assertAlmostEqual(stats["exgrpo_exp_weight_sum"], 0.5, places=6)
        self.assertEqual(stats["exgrpo_n_on_groups"], 1.0)
        self.assertEqual(stats["exgrpo_n_exp_groups"], 1.0)
        # Each on-policy trajectory: (1-ρ)/(n_on·K) = 0.5/8 = 0.0625
        self.assertAlmostEqual(weights[0], 0.0625, places=6)
        # Each exp trajectory (incl. replay): ρ/(n_exp·K) = 0.5/8 = 0.0625
        self.assertAlmostEqual(weights[8], 0.0625, places=6)


class TestMergeReplay(unittest.TestCase):
    def test_merge_replay_and_fresh(self) -> None:
        prefix = "shared-prefix"
        replay = _make_episode(prefix, 1.0, is_replay=True)
        fresh = [_make_episode(prefix, 0.0) for _ in range(7)]
        merged = merge_replay_and_fresh_rollouts([replay], fresh)
        self.assertEqual(len(merged), 8)
        self.assertTrue(merged[0].is_replay)
        self.assertTrue(all(ep.is_exp_group for ep in merged))
        self.assertFalse(merged[1].is_replay)


class TestReplayImportanceSampling(unittest.TestCase):
    def test_replay_not_recollected_with_zero_logprobs(self) -> None:
        buf = ExperienceBuffer(K=8)
        prefix = "replay-q"
        live = _make_episode(prefix, 1.0)
        live.generated_token_log_probs = [-1.0, -1.2]
        buf.collect([live] + [_make_episode(prefix, 0.0) for _ in range(7)])
        q_id = question_id_from_prefix(prefix)
        self.assertEqual(len(buf.buffer[q_id]), 1)
        stored_past = buf.buffer[q_id][0].past_token_log_probs

        replay = episode_from_stored_trajectory(buf.buffer[q_id][0])
        replay.generated_token_log_probs = [0.0, 0.0]
        group = [replay] + [_make_episode(prefix, 0.0) for _ in range(7)]
        buf.collect(group)
        self.assertEqual(buf.buffer[q_id][0].past_token_log_probs, stored_past)

    def test_episode_from_stored_copies_past_logprobs(self) -> None:
        stored = StoredTrajectory(
            generated_token_ids=[1, 2],
            past_token_log_probs=[-0.5, -0.7],
            prefix="p",
            prefix_token_ids=[1, 2, 3],
            prefix_tokens=["a"],
            text="p ans",
        )
        ep = episode_from_stored_trajectory(stored)
        self.assertEqual(ep.generated_token_log_probs, [-0.5, -0.7])
        self.assertEqual(ep.past_token_log_probs, [-0.5, -0.7])


class TestSuccessDetection(unittest.TestCase):
    def test_is_successful_rollout(self) -> None:
        ep = _make_episode("q", 0.0)
        ep.reward_info = {"answer_reward": 1.0}
        self.assertTrue(is_successful_rollout(ep))


class TestSequenceClipModes(unittest.TestCase):
    def test_gspo_none_uses_mean_log_ratio(self) -> None:
        """GSPO: ratio = exp(mean log r), clip in ratio space."""
        target_masks = torch.tensor([[True, True, True, True]])
        new_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        old_log_probs = torch.tensor([[-1.1, -1.1, -1.1, -1.1]])
        advantages = torch.tensor([1.0])

        loss, metrics = compute_gspo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            target_masks=target_masks,
            loss_denominator=1.0,
            clip_eps=0.2,
            clip_ratio_low=3.0e-4,
            clip_ratio_high=4.0e-4,
            use_ppo_clip=True,
            gspo_clip_len_scaling="none",
        )
        # mean log-ratio = 0.1 -> seq_ratio = exp(0.1) ~ 1.105 > 1.0004 => clipped
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["clip_responses"], 1.0)

    def test_fspo_sqrt_uses_sum_log_ratio_and_log_band(self) -> None:
        """FSPO: ratio = exp(sum log r), clip S to [-c*sqrt(L), c*sqrt(L)]."""
        l_len = 4
        target_masks = torch.ones(1, l_len, dtype=torch.bool)
        # per-token log-ratio = 0.01 -> S = 0.04, sqrt(L)=2, c=0.03 -> band=0.06, not clipped
        new_log_probs = torch.full((1, l_len), -0.99)
        old_log_probs = torch.full((1, l_len), -1.0)
        advantages = torch.tensor([1.0])

        loss_open, metrics_open = compute_gspo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            target_masks=target_masks,
            loss_denominator=1.0,
            clip_eps=0.03,
            clip_ratio_low=0.03,
            clip_ratio_high=0.03,
            use_ppo_clip=True,
            gspo_clip_len_scaling="sqrt",
        )
        self.assertEqual(metrics_open["clip_responses"], 0.0)

        # S = 1.2 > c*sqrt(4)=0.06 => clipped
        new_log_probs = torch.full((1, l_len), 0.2)
        old_log_probs = torch.full((1, l_len), -0.1)
        loss_clip, metrics_clip = compute_gspo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            target_masks=target_masks,
            loss_denominator=1.0,
            clip_eps=0.03,
            clip_ratio_low=0.03,
            clip_ratio_high=0.03,
            use_ppo_clip=True,
            gspo_clip_len_scaling="sqrt",
        )
        self.assertTrue(torch.isfinite(loss_open))
        self.assertTrue(torch.isfinite(loss_clip))
        self.assertEqual(metrics_clip["clip_responses"], 1.0)

    def test_linear_scaling_removed(self) -> None:
        with self.assertRaises(ValueError):
            compute_gspo_loss(
                new_log_probs=torch.zeros(1, 2),
                old_log_probs=torch.zeros(1, 2),
                advantages=torch.tensor([1.0]),
                target_masks=torch.ones(1, 2, dtype=torch.bool),
                loss_denominator=1.0,
                clip_eps=0.2,
                gspo_clip_len_scaling="linear",
            )


class TestExGRPOGSPOLoss(unittest.TestCase):
    def test_gspo_exgrpo_loss_weighted_sum(self) -> None:
        """ExGRPO+GSPO loss uses episode weights and mixes on/replay surrogates."""
        batch = 4
        seq_len = 3
        new_log_probs = torch.zeros(batch, seq_len)
        old_log_probs = torch.full((batch, seq_len), -0.5)
        past_log_probs = torch.full((batch, seq_len), -0.4)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        target_masks = torch.ones(batch, seq_len, dtype=torch.bool)
        replay_mask = torch.tensor([False, True, False, True])
        episode_weights = torch.tensor([0.125, 0.125, 0.125, 0.125])

        loss, metrics = compute_exgrpo_gspo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            past_log_probs=past_log_probs,
            advantages=advantages,
            target_masks=target_masks,
            replay_mask=replay_mask,
            episode_weights=episode_weights,
            shaping_beta=0.1,
            clip_eps=0.03,
            clip_ratio_low=0.03,
            clip_ratio_high=0.03,
            use_ppo_clip=True,
            gspo_clip_len_scaling="sqrt",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["exgrpo_replay_episode_weight_sum"], 0.25, places=4)
        self.assertAlmostEqual(metrics["exgrpo_on_episode_weight_sum"], 0.25, places=4)
        self.assertEqual(metrics["num_responses"], 4.0)


class TestExGRPOCheckpointState(unittest.TestCase):
    def _populate_manager(self, mgr: ExGRPOManager) -> None:
        for i in range(12):
            prefix = f"buffer-q-{i}"
            episodes = [_make_episode(prefix, 1.0 if j == 0 else 0.0) for j in range(8)]
            episodes[0].generated_token_ids = list(range(100 + i, 120 + i))
            episodes[0].generated_token_log_probs = [-0.2 - i * 0.01] * len(
                episodes[0].generated_token_ids
            )
            mgr.buffer.collect(episodes)
        mgr._activated = True

    def test_experience_buffer_state_roundtrip(self) -> None:
        buf = ExperienceBuffer(K=8, mu=0.5, sigma=1.0, rng=np.random.default_rng(7))
        prefix = "roundtrip-q"
        episodes = [_make_episode(prefix, 1.0)] + [_make_episode(prefix, 0.0) for _ in range(7)]
        buf.collect(episodes)
        q_id = question_id_from_prefix(prefix)
        buf.retired_set.add("retired-id")
        state = buf.state_dict(compact=True)
        restored = ExperienceBuffer(K=8, mu=0.5, sigma=1.0, rng=np.random.default_rng(0))
        restored.load_state_dict(state)
        self.assertEqual(len(restored.buffer[q_id]), 1)
        self.assertAlmostEqual(restored.acc_tracker[q_id], 0.125)
        self.assertIn("retired-id", restored.retired_set)

    def test_manager_state_roundtrip_preserves_activation(self) -> None:
        cfg = ExGRPOConfig(enabled=True, K=8, rho=0.5, mix_acc_threshold=0.125)
        mgr = ExGRPOManager(cfg, rng=np.random.default_rng(3))
        self._populate_manager(mgr)
        state = mgr.state_dict(compact=True)
        restored = ExGRPOManager(cfg, rng=np.random.default_rng(99))
        restored.load_state_dict(state)
        self.assertTrue(restored.activated)
        self.assertEqual(len(restored.buffer), len(mgr.buffer))
        self.assertEqual(restored.buffer.num_trajectories, mgr.buffer.num_trajectories)

    def test_replay_episode_works_after_restore(self) -> None:
        cfg = ExGRPOConfig(enabled=True, K=8)
        mgr = ExGRPOManager(cfg)
        prefix = "replay-restore"
        episodes = [_make_episode(prefix, 1.0)] + [_make_episode(prefix, 0.0) for _ in range(7)]
        mgr.buffer.collect(episodes)
        state = mgr.state_dict(compact=True)
        restored = ExGRPOManager(cfg)
        restored.load_state_dict(state)
        q_id = question_id_from_prefix(prefix)
        ep = episode_from_stored_trajectory(restored.buffer.buffer[q_id][0])
        self.assertTrue(ep.is_replay)
        self.assertEqual(len(ep.past_token_log_probs), 2)

    def test_storage_estimate_matches_state_dict(self) -> None:
        cfg = ExGRPOConfig(enabled=True, K=8)
        mgr = ExGRPOManager(cfg)
        self._populate_manager(mgr)
        state = mgr.state_dict(compact=True)
        measured = estimate_exgrpo_state_bytes(state)
        heuristic = estimate_exgrpo_state_bytes(
            num_questions=len(state["buffer"]["buffer"]),
            avg_gen_tokens=20,
            avg_prefix_tokens=3,
            avg_prefix_chars=12,
            compact=True,
        )
        self.assertGreater(measured, 0)
        self.assertIn("MiB", format_exgrpo_storage_summary(state=state))
        self.assertGreater(heuristic, 0)

    def test_config_mismatch_raises_on_strict_load(self) -> None:
        cfg = ExGRPOConfig(enabled=True, K=8, rho=0.5)
        mgr = ExGRPOManager(cfg)
        state = mgr.state_dict()
        other = ExGRPOConfig(enabled=True, K=8, rho=0.25)
        restored = ExGRPOManager(other)
        with self.assertRaises(ValueError):
            restored.load_state_dict(state, strict=True)

    def test_checkpoint_payload_roundtrip(self) -> None:
        class _TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

        cfg = ExGRPOConfig(enabled=True, K=8)
        mgr = ExGRPOManager(cfg)
        self._populate_manager(mgr)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        payload = build_lora_checkpoint_payload(
            7,
            model,
            optimizer,
            base_model_path="base",
            lora_config={"r": 4},
            exgrpo_state=mgr.state_dict(compact=True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test.pt"
            torch.save(payload, ckpt_path)
            loaded = torch.load(ckpt_path, map_location="cpu")
            restored_mgr = ExGRPOManager(cfg)
            restored_mgr.load_state_dict(loaded["exgrpo_state"])
        self.assertEqual(loaded["step"], 7)
        self.assertIn("exgrpo_state", loaded)
        self.assertTrue(restored_mgr.activated)
        self.assertEqual(
            restored_mgr.buffer.num_trajectories,
            mgr.buffer.num_trajectories,
        )


def _find_local_model_dir(*candidates: str):
    """Return first existing model directory under PROJECT_ROOT, else None."""
    from model_registry import PROJECT_ROOT

    for rel in candidates:
        path = PROJECT_ROOT / rel
        if path.is_dir() and (path / "config.json").is_file():
            return path
    return None


class TestModelRegistry(unittest.TestCase):
    def test_resolve_math7b_base_preset(self) -> None:
        from model_registry import resolve_model_config

        if _find_local_model_dir("models/Qwen2.5-Math-7B", "Qwen2.5-Math-7B") is None:
            self.skipTest("Qwen2.5-Math-7B not present locally")

        resolved = resolve_model_config({"preset": "qwen2.5-math-7b"})
        self.assertEqual(resolved.path.name, "Qwen2.5-Math-7B")
        self.assertEqual(resolved.run_tag, "exgrpo-math7b-base")
        self.assertEqual(resolved.max_position_embeddings, 4096)

    def test_infer_math7b_base_from_path(self) -> None:
        from model_registry import PROJECT_ROOT, resolve_model_config

        model_dir = _find_local_model_dir("models/Qwen2.5-Math-7B", "Qwen2.5-Math-7B")
        if model_dir is None:
            self.skipTest("Qwen2.5-Math-7B not present locally")

        rel = str(model_dir.relative_to(PROJECT_ROOT))
        resolved = resolve_model_config({"pretrained_model_path": rel})
        self.assertEqual(resolved.preset, "qwen2.5-math-7b")
        self.assertEqual(resolved.path.name, "Qwen2.5-Math-7B")

    def test_validate_seq_limits_with_fake_config(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from model_registry import (
            ResolvedModelConfig,
            validate_training_seq_limits,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            resolved = ResolvedModelConfig(
                preset="qwen2.5-math-7b",
                path=Path(tmp),
                run_tag="exgrpo-math7b-base",
                max_position_embeddings=4096,
            )
            with self.assertRaises(ValueError):
                validate_training_seq_limits(
                    resolved,
                    max_prompt_len=2048,
                    max_gen_len=3072,
                )

    def test_resolve_7b_instruct_preset(self) -> None:
        from model_registry import resolve_model_config

        if (
            _find_local_model_dir(
                "models/Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-7B-Instruct"
            )
            is None
        ):
            self.skipTest("Qwen2.5-Math-7B-Instruct not present locally")

        resolved = resolve_model_config(
            {"preset": "qwen2.5-math-7b-instruct"}
        )
        self.assertEqual(resolved.path.name, "Qwen2.5-Math-7B-Instruct")
        self.assertEqual(resolved.run_tag, "exgrpo-math7b")
        self.assertEqual(resolved.max_position_embeddings, 4096)

    def test_validate_seq_limits_raises_for_7b(self) -> None:
        from model_registry import resolve_model_config, validate_training_seq_limits

        if _find_local_model_dir("models/Qwen2.5-Math-7B", "Qwen2.5-Math-7B") is None:
            self.skipTest("Qwen2.5-Math-7B not present locally")

        resolved = resolve_model_config({"preset": "qwen2.5-math-7b"})
        with self.assertRaises(ValueError):
            validate_training_seq_limits(
                resolved,
                max_prompt_len=2048,
                max_gen_len=3072,
            )

    def test_preset_paths_use_models_prefix(self) -> None:
        from model_registry import MODEL_PRESETS

        for rel in MODEL_PRESETS.values():
            self.assertTrue(
                rel.startswith("models/"),
                f"open-source layout expects models/ prefix, got {rel}",
            )


class TestMathBasePrompt(unittest.TestCase):
    def test_math_prefix_uses_boxed_system_not_r1_thinking(self) -> None:
        from math_base_prompt import (
            MATH_SYSTEM_MESSAGE,
            build_math_chat_prefix,
            resolve_math_rollout_stop,
            resolve_math_rollout_stop_ids,
        )
        from tokenizer import Tokenizer

        model_dir = _find_local_model_dir("models/Qwen2.5-Math-7B", "Qwen2.5-Math-7B")
        if model_dir is None:
            self.skipTest("Qwen2.5-Math-7B tokenizer not present locally")

        tok = Tokenizer(str(model_dir / "tokenizer.json"))
        prefix = build_math_chat_prefix(tok, "What is 2+2?")
        self.assertIn("\\boxed{}", prefix)
        self.assertIn(MATH_SYSTEM_MESSAGE, prefix)
        self.assertNotIn("<think>", prefix)
        stop_token, stop_id = resolve_math_rollout_stop(tok)
        self.assertEqual(stop_id, 151645)
        stop_ids = resolve_math_rollout_stop_ids(tok)
        self.assertIn(tok.eos_token_id, stop_ids)
        self.assertIn(151645, stop_ids)

    def test_generated_token_slice_aligns_with_log_probs_when_pad_is_eos(self) -> None:
        """Qwen pad==eos: must slice by log-prob count, not first pad index."""
        prefix_len = 10
        eos_id = 151643
        tokens = list(range(prefix_len)) + [100, 101, eos_id, eos_id, eos_id]
        log_probs = [-0.1, -0.2, -0.3]
        gen_len = len(log_probs)
        generated = tokens[prefix_len : prefix_len + gen_len]
        self.assertEqual(generated, [100, 101, eos_id])
        self.assertEqual(len(generated), len(log_probs))


class TestMathBaseReward(unittest.TestCase):
    def test_format_reward_prefers_boxed(self) -> None:
        from exgrpo_reward import ExGRPORewardConfig
        from math_base_reward import compute_math_base_reward

        cfg = ExGRPORewardConfig(w_accuracy=1.0, w_format=0.2)
        boxed = compute_math_base_reward(
            "Step 1: add. \\boxed{4}",
            dataset_kind="math",
            cfg=cfg,
            ground_truth="4",
        )
        plain = compute_math_base_reward(
            "The answer is 4",
            dataset_kind="math",
            cfg=cfg,
            ground_truth="4",
        )
        self.assertGreater(boxed["reward_info"]["format_reward"], 0.9)
        self.assertLess(plain["reward_info"]["format_reward"], boxed["reward_info"]["format_reward"])


if __name__ == "__main__":
    unittest.main()
