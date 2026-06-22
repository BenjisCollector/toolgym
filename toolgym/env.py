"""A minimal Gymnasium-compatible tool-use environment.

The agent is shown an arithmetic query (a op b) and must decide which *tool*
to invoke. The point is not the arithmetic itself but learning the *policy of
tool use*: when to gather information (CALCULATOR / LOOKUP) and when to commit
(SUBMIT). This is a toy stand-in for "agentic models via tool use".

The env follows the Gymnasium API (reset / step / observation_space /
action_space). It imports Gymnasium when available; otherwise it falls back to
a tiny local shim so the environment and the random agent run with numpy alone.
"""

from __future__ import annotations

import numpy as np

try:  # Prefer the real Gymnasium API when installed.
    import gymnasium as gym
    from gymnasium import spaces

    _HAVE_GYM = True
except Exception:  # pragma: no cover - fallback shim, exercised when gym absent.
    _HAVE_GYM = False

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return (self.np_random_uniform() * (self.high - self.low) + self.low).astype(
                self.dtype
            )

        def np_random_uniform(self):
            return np.random.random(self.shape)

        def contains(self, x):
            x = np.asarray(x)
            return bool(
                x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

        def sample(self):
            return int(np.random.randint(self.n))

        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class spaces:  # type: ignore  # noqa: N801 - mimic gymnasium.spaces namespace
        Box = _Box
        Discrete = _Discrete

    class gym:  # type: ignore  # noqa: N801 - mimic gymnasium.Env base class
        class Env:
            metadata: dict = {}

            def reset(self, *, seed=None, options=None):
                raise NotImplementedError

            def step(self, action):
                raise NotImplementedError

            def close(self):
                pass


# Action ids -----------------------------------------------------------------
CALCULATOR = 0  # compute the answer (reveals it in the observation)
LOOKUP = 1      # consult a "memory" tool (reveals it in the observation)
SUBMIT = 2      # commit the currently-known answer
N_ACTIONS = 3

# Supported binary operators, encoded as ints in the observation.
_OPS = {0: ("+", lambda a, b: a + b), 1: ("-", lambda a, b: a - b), 2: ("*", lambda a, b: a * b)}


class ToolUseEnv(gym.Env):
    """Tool-use MDP over single-step arithmetic queries.

    Observation (float32 vector of length 5):
        [operand_a, operand_b, op_id, answer_known, known_answer]
      - operand_a, operand_b: the query operands (0..max_operand)
      - op_id:                which operator (0=+, 1=-, 2=*)
      - answer_known:         1.0 once a CALCULATOR or LOOKUP tool has been used
      - known_answer:         the tool's answer, or 0.0 before any tool is used

    Actions (Discrete(3)): CALCULATOR, LOOKUP, SUBMIT.

    Reward:
      - CALCULATOR / LOOKUP: -tool_cost (small penalty; info has a price)
      - SUBMIT with a correct, tool-derived answer: +1.0, episode ends
      - SUBMIT without ever using a tool, or with a wrong answer: -1.0, ends
      - Exceeding max_steps tools without submitting: episode truncates, no bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_operand: int = 20, tool_cost: float = 0.1, max_steps: int = 5):
        super().__init__()
        self.max_operand = int(max_operand)
        self.tool_cost = float(tool_cost)
        self.max_steps = int(max_steps)

        high = float(max_operand * max_operand)  # bound large enough for products
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng()
        self._a = 0
        self._b = 0
        self._op_id = 0
        self._true_answer = 0
        self._known_answer = 0.0
        self._answer_known = False
        self._steps = 0

    # -- Gymnasium API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._a = int(self._rng.integers(0, self.max_operand + 1))
        self._b = int(self._rng.integers(0, self.max_operand + 1))
        self._op_id = int(self._rng.integers(0, len(_OPS)))
        _, fn = _OPS[self._op_id]
        self._true_answer = int(fn(self._a, self._b))
        self._known_answer = 0.0
        self._answer_known = False
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        self._steps += 1
        terminated = False
        truncated = False
        reward = 0.0

        if action in (CALCULATOR, LOOKUP):
            # Both tools return the correct answer here; a richer env could make
            # them noisy / differ in cost. Using a tool reveals the answer.
            self._known_answer = float(self._true_answer)
            self._answer_known = True
            reward = -self.tool_cost
        elif action == SUBMIT:
            terminated = True
            if self._answer_known and int(self._known_answer) == self._true_answer:
                reward = 1.0
            else:
                # Guessing without using a tool, or submitting a wrong value.
                reward = -1.0

        if not terminated and self._steps >= self.max_steps:
            truncated = True  # ran out of patience without committing

        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        op = _OPS[self._op_id][0]
        known = int(self._known_answer) if self._answer_known else "?"
        print(f"query: {self._a} {op} {self._b} = ?  | tool answer so far: {known}")

    # -- helpers -------------------------------------------------------------
    def _obs(self):
        return np.array(
            [
                float(self._a),
                float(self._b),
                float(self._op_id),
                1.0 if self._answer_known else 0.0,
                float(self._known_answer),
            ],
            dtype=np.float32,
        )

    def _info(self):
        return {
            "true_answer": self._true_answer,
            "answer_known": self._answer_known,
            "steps": self._steps,
        }


# Optional Gymnasium registration so `gym.make("ToolUse-v0")` works when the
# real library is installed. Harmless no-op otherwise.
if _HAVE_GYM:  # pragma: no cover - depends on optional dependency
    try:
        gym.register(id="ToolUse-v0", entry_point="toolgym.env:ToolUseEnv")
    except Exception:
        pass
