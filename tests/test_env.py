"""Tests for ToolUseEnv. Numpy-only; no gymnasium required.

Covers the Gymnasium reset/step contracts and the core reward logic: that a
correct tool-then-submit yields positive reward, and that shortcuts (blind
submit, wrong tool, wrong value) are punished.
"""

from __future__ import annotations

import numpy as np
import pytest

from toolgym.env import (
    CALCULATOR,
    LOOKUP,
    N_ACTIONS,
    SUBMIT,
    ToolUseEnv,
)


# -- reset / step contracts --------------------------------------------------

def test_reset_returns_obs_and_info():
    env = ToolUseEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert "true_answer" in info
    assert "task_type" in info


def test_reset_is_deterministic_given_seed():
    env = ToolUseEnv()
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_step_returns_five_tuple():
    env = ToolUseEnv()
    env.reset(seed=1)
    result = env.step(SUBMIT)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space_size():
    env = ToolUseEnv()
    assert env.action_space.n == N_ACTIONS == 3


def test_invalid_action_raises():
    env = ToolUseEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_unknown_task_variant_raises():
    with pytest.raises(ValueError):
        ToolUseEnv(task="nonsense")


# -- core reward logic -------------------------------------------------------

def test_correct_tool_then_submit_is_positive_arithmetic():
    """The headline contract: right tool, then submit -> positive reward."""
    env = ToolUseEnv(task="arithmetic")
    env.reset(seed=0)
    obs, r_tool, term, trunc, info = env.step(CALCULATOR)
    assert info["answer_known"] is True
    assert r_tool == pytest.approx(-env.tool_cost)
    assert term is False

    obs, r_submit, term, trunc, info = env.step(SUBMIT)
    assert r_submit == pytest.approx(1.0)
    assert term is True

    # Net return is positive: +1.0 minus a single tool cost.
    assert (r_tool + r_submit) > 0.0


def test_correct_tool_then_submit_is_positive_lookup():
    env = ToolUseEnv(task="lookup")
    env.reset(seed=2)
    _, r_tool, _, _, info = env.step(LOOKUP)
    assert info["answer_known"] is True
    _, r_submit, term, _, _ = env.step(SUBMIT)
    assert r_submit == pytest.approx(1.0)
    assert term is True
    assert (r_tool + r_submit) > 0.0


def test_blind_submit_is_punished():
    env = ToolUseEnv(task="arithmetic")
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(SUBMIT)
    assert reward == pytest.approx(-1.0)
    assert terminated is True


def test_wrong_tool_does_not_reveal_answer_arithmetic():
    """Using LOOKUP on an arithmetic task must not mark the answer known."""
    env = ToolUseEnv(task="arithmetic")
    env.reset(seed=0)
    _, _, _, _, info = env.step(LOOKUP)  # wrong tool for arithmetic
    assert info["answer_known"] is False
    _, reward, terminated, _, _ = env.step(SUBMIT)
    assert reward == pytest.approx(-1.0)
    assert terminated is True


def test_wrong_tool_does_not_reveal_answer_lookup():
    env = ToolUseEnv(task="lookup")
    env.reset(seed=0)
    _, _, _, _, info = env.step(CALCULATOR)  # wrong tool for lookup
    assert info["answer_known"] is False
    _, reward, terminated, _, _ = env.step(SUBMIT)
    assert reward == pytest.approx(-1.0)


def test_episode_truncates_without_submit():
    env = ToolUseEnv(task="arithmetic", max_steps=3)
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(CALCULATOR)  # never submit
        assert terminated is False
    assert truncated is True


def test_correct_tool_reveals_true_answer():
    """The correct tool's revealed answer must equal the true answer."""
    env = ToolUseEnv(task="arithmetic")
    _, info = env.reset(seed=7)
    obs, _, _, _, _ = env.step(CALCULATOR)
    # obs[6] is known_answer; obs[5] is answer_known.
    assert obs[5] == pytest.approx(1.0)
    assert int(obs[6]) == info["true_answer"]


def test_observation_within_space_throughout_episode():
    env = ToolUseEnv(task="mixed")
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    for action in (CALCULATOR, LOOKUP, SUBMIT):
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_mixed_task_samples_both_variants():
    """Over many resets, mixed mode should produce both task types."""
    env = ToolUseEnv(task="mixed")
    seen = set()
    for s in range(50):
        _, info = env.reset(seed=s)
        seen.add(info["task_type"])
    assert seen == {"arithmetic", "lookup"}


# -- baseline smoke test -----------------------------------------------------

def test_qlearning_baseline_beats_random():
    """A short training run should already beat the random policy."""
    import examples.train_qlearning as tq

    q = tq.train(num_episodes=1500)
    learned = tq.evaluate_policy(q, num_episodes=500)
    random_reward = tq.evaluate_random(num_episodes=500)
    assert learned > random_reward
    assert learned > 0.0  # a competent tool-user earns positive return
