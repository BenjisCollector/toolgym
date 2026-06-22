"""Tabular Q-learning baseline for ToolUseEnv.

This trains a tiny Q-table to solve the tool-use MDP and shows that it learns a
*policy of tool use* that beats a random policy: read the task type, call the
matching tool, then SUBMIT.

The full observation is continuous, but the decision-relevant structure is
small. We discretize each observation to the two features that determine the
optimal action:

    state = (task_type, answer_known)   ->  4 discrete states

The learned policy must map:
    (arithmetic, not-known) -> CALCULATOR
    (lookup,     not-known) -> LOOKUP
    (*,          known)     -> SUBMIT

That is the genuine tool-selection skill: the agent cannot win by always
pressing one button — the right first action depends on the task.

Runs with numpy only (the env shims Gymnasium when it is absent).

Usage:
    python examples/train_qlearning.py [num_train_episodes]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the package importable when run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolgym.env import N_ACTIONS, SUBMIT, ToolUseEnv  # noqa: E402

N_STATES = 4  # (task_type in {0,1}) x (answer_known in {0,1})


def encode_state(obs: np.ndarray) -> int:
    """Map a continuous observation to a discrete Q-table index.

    Uses obs[0] = task_type and obs[5] = answer_known.
    """
    task_type = int(round(float(obs[0])))   # 0 arithmetic, 1 lookup
    answer_known = int(round(float(obs[5])))  # 0 / 1
    return task_type * 2 + answer_known


def greedy_action(q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    """Argmax with random tie-breaking, so an all-zero row isn't biased."""
    row = q[state]
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best))


def train(
    num_episodes: int = 4000,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Train a Q-table with epsilon-greedy exploration; return the table."""
    env = ToolUseEnv(task="mixed")
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    for ep in range(num_episodes):
        # Linearly anneal exploration from start to end over training.
        frac = ep / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        obs, _ = env.reset(seed=seed + ep)
        state = encode_state(obs)
        done = False

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                action = greedy_action(q, state, rng)

            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(obs)
            done = terminated or truncated

            # Standard Q-learning update. No bootstrap from terminal states.
            target = reward
            if not terminated:
                target += gamma * q[next_state].max()
            q[state, action] += alpha * (target - q[state, action])

            state = next_state

    return q


def evaluate_policy(q: np.ndarray, num_episodes: int = 2000, seed: int = 10000) -> float:
    """Mean episode return of the greedy policy induced by ``q``."""
    env = ToolUseEnv(task="mixed")
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = encode_state(obs)
        total = 0.0
        done = False
        while not done:
            action = greedy_action(q, state, rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            state = encode_state(obs)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def evaluate_random(num_episodes: int = 2000, seed: int = 10000) -> float:
    """Mean episode return of a uniform-random policy (the baseline-to-beat)."""
    env = ToolUseEnv(task="mixed")
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            action = int(rng.integers(0, N_ACTIONS))
            _, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


_ACTION_NAMES = {0: "CALCULATOR", 1: "LOOKUP", 2: "SUBMIT"}
_STATE_NAMES = {
    0: "arithmetic, answer unknown",
    1: "arithmetic, answer known ",
    2: "lookup,     answer unknown",
    3: "lookup,     answer known  ",
}


def print_policy(q: np.ndarray) -> None:
    print("learned greedy policy (state -> action):")
    for s in range(N_STATES):
        a = int(np.argmax(q[s]))
        print(f"  [{_STATE_NAMES[s]}] -> {_ACTION_NAMES[a]:<10} (Q={q[s]})")


def main(num_episodes: int = 4000) -> None:
    print(f"training tabular Q-learning for {num_episodes} episodes...")
    q = train(num_episodes=num_episodes)

    random_reward = evaluate_random()
    learned_reward = evaluate_policy(q)
    optimal = 1.0 - ToolUseEnv().tool_cost  # one correct tool call, then submit

    print("-" * 52)
    print_policy(q)
    print("-" * 52)
    print(f"random  policy mean reward: {random_reward:+.3f}")
    print(f"learned policy mean reward: {learned_reward:+.3f}")
    print(f"theoretical optimum:        {optimal:+.3f}")
    print(f"improvement over random:    {learned_reward - random_reward:+.3f}")
    if learned_reward > random_reward:
        print("RESULT: learned policy beats random.")
    else:  # pragma: no cover - should not happen with these hyperparameters
        print("RESULT: learned policy did NOT beat random (check hyperparameters).")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    main(n)
