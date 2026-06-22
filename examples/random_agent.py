"""Run ToolUseEnv with a uniform-random policy and report episode rewards.

Usage:
    python examples/random_agent.py [num_episodes]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the package importable when run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolgym.env import ToolUseEnv  # noqa: E402


def run(num_episodes: int = 10, seed: int = 0) -> None:
    env = ToolUseEnv()
    rng = np.random.default_rng(seed)
    rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
        print(f"episode {ep:2d} | reward {total:+.2f} | true_answer {info['true_answer']}")

    rewards = np.array(rewards, dtype=np.float64)
    print("-" * 40)
    print(f"episodes: {len(rewards)}")
    print(f"mean reward: {rewards.mean():+.3f}")
    print(f"min/max:     {rewards.min():+.2f} / {rewards.max():+.2f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(n)
