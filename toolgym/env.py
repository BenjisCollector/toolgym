"""A minimal Gymnasium-compatible tool-use environment.

The agent is shown a *query* and must decide which **tool** to invoke before it
commits an answer. The point is not the underlying task (arithmetic / lookup)
but learning the *policy of tool use*: pick the right tool to gather the answer,
then SUBMIT. This is a toy stand-in for "agentic models via tool use".

There are three task variants, chosen per episode:

* ``arithmetic`` — a query ``a op b``. The **CALCULATOR** tool returns the
  correct value; the LOOKUP tool returns garbage for this task.
* ``lookup`` — a query "what is the value of key ``k``?" backed by a small
  key/value table. The **LOOKUP** tool returns the correct value; the
  CALCULATOR tool returns garbage for this task.
* ``mixed`` — the env samples either of the above uniformly; this is the
  default and forces the agent to *read the task type from the observation*
  and choose the matching tool.

So the learnable skill is genuinely "select the correct tool conditioned on the
task, then submit" — not just "always press the same button".

The env follows the Gymnasium API (reset / step / observation_space /
action_space). It imports Gymnasium when available; otherwise it falls back to a
tiny local shim so the environment, tests, and baselines run with numpy alone.
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
            return (np.random.random(self.shape) * (self.high - self.low) + self.low).astype(
                self.dtype
            )

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
CALCULATOR = 0  # correct for arithmetic tasks; wrong for lookup tasks
LOOKUP = 1      # correct for lookup tasks; wrong for arithmetic tasks
SUBMIT = 2      # commit the currently-known answer
N_ACTIONS = 3

# Task-type ids (also encoded into the observation).
TASK_ARITHMETIC = 0
TASK_LOOKUP = 1

# Supported binary operators for arithmetic tasks, encoded as ints.
_OPS = {
    0: ("+", lambda a, b: a + b),
    1: ("-", lambda a, b: a - b),
    2: ("*", lambda a, b: a * b),
}


class ToolUseEnv(gym.Env):
    """Tool-use MDP over single-step queries with task-dependent tools.

    Observation (float32 vector of length 7):

        [task_type, operand_a, operand_b, op_id, lookup_key,
         answer_known, known_answer]

      - task_type:     0 = arithmetic, 1 = lookup
      - operand_a/b:   operands for arithmetic tasks (0 for lookup tasks)
      - op_id:         operator for arithmetic tasks (0=+, 1=-, 2=*)
      - lookup_key:    key id for lookup tasks (0 for arithmetic tasks)
      - answer_known:  1.0 once the *correct* tool has produced the answer
      - known_answer:  the answer currently held, or 0.0 before any tool

    Actions (Discrete(3)): CALCULATOR, LOOKUP, SUBMIT.

    Reward:
      - Using the *correct* tool for the task: reveals the true answer,
        costs ``-tool_cost``.
      - Using the *wrong* tool: returns a garbage value (answer NOT known),
        costs ``-tool_cost``.
      - SUBMIT with the correct, tool-derived answer: ``+1.0``, episode ends.
      - SUBMIT otherwise (no tool used, wrong tool, wrong value): ``-1.0``, ends.
      - Reaching ``max_steps`` without submitting: episode truncates, no bonus.

    The optimal policy reads ``task_type`` from the observation, calls the
    matching tool once, then submits — expected return ``+1.0 - tool_cost``.

    Parameters
    ----------
    task : {"mixed", "arithmetic", "lookup"}
        Which task variant(s) to sample each episode. Default ``"mixed"``.
    max_operand : int
        Operands are drawn from ``0..max_operand`` (arithmetic tasks).
    n_keys : int
        Size of the lookup table (lookup tasks).
    tool_cost : float
        Per-tool-call penalty. Information has a price.
    max_steps : int
        Patience: episode truncates after this many steps without SUBMIT.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task: str = "mixed",
        max_operand: int = 20,
        n_keys: int = 10,
        tool_cost: float = 0.1,
        max_steps: int = 5,
    ):
        super().__init__()
        if task not in ("mixed", "arithmetic", "lookup"):
            raise ValueError(f"unknown task variant: {task!r}")
        self.task = task
        self.max_operand = int(max_operand)
        self.n_keys = int(n_keys)
        self.tool_cost = float(tool_cost)
        self.max_steps = int(max_steps)

        # A fixed lookup "database": key id -> value. Deterministic so the
        # LOOKUP tool is a genuine information source, not a re-roll each call.
        self._table = {k: int((k * 7 + 3) % 50) for k in range(self.n_keys)}

        high = float(max(max_operand * max_operand, 50))  # bound for products/values
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng()
        self._reset_episode_state()

    # -- Gymnasium API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.task == "mixed":
            self._task_type = int(self._rng.integers(0, 2))
        elif self.task == "arithmetic":
            self._task_type = TASK_ARITHMETIC
        else:
            self._task_type = TASK_LOOKUP

        self._reset_episode_state()

        if self._task_type == TASK_ARITHMETIC:
            self._a = int(self._rng.integers(0, self.max_operand + 1))
            self._b = int(self._rng.integers(0, self.max_operand + 1))
            self._op_id = int(self._rng.integers(0, len(_OPS)))
            _, fn = _OPS[self._op_id]
            self._true_answer = int(fn(self._a, self._b))
            self._correct_tool = CALCULATOR
        else:
            self._key = int(self._rng.integers(0, self.n_keys))
            self._true_answer = int(self._table[self._key])
            self._correct_tool = LOOKUP

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
            reward = -self.tool_cost
            if action == self._correct_tool:
                # The matching tool reveals the genuine answer.
                self._known_answer = float(self._true_answer)
                self._answer_known = True
            else:
                # Wrong tool for this task: returns a deterministic but useless
                # value, and crucially does NOT mark the answer as known.
                self._known_answer = float((self._true_answer + 13) % 50)
                self._answer_known = False
        elif action == SUBMIT:
            terminated = True
            if self._answer_known and int(self._known_answer) == self._true_answer:
                reward = 1.0
            else:
                # Guessing with no tool, the wrong tool, or a wrong value.
                reward = -1.0

        if not terminated and self._steps >= self.max_steps:
            truncated = True  # ran out of patience without committing

        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        known = int(self._known_answer) if self._answer_known else "?"
        if self._task_type == TASK_ARITHMETIC:
            op = _OPS[self._op_id][0]
            query = f"{self._a} {op} {self._b} = ?"
        else:
            query = f"lookup(key={self._key}) = ?"
        print(f"task={self._task_name()} | query: {query} | tool answer so far: {known}")

    # -- helpers -------------------------------------------------------------
    def _reset_episode_state(self):
        self._a = 0
        self._b = 0
        self._op_id = 0
        self._key = 0
        self._true_answer = 0
        self._correct_tool = CALCULATOR
        self._known_answer = 0.0
        self._answer_known = False
        self._steps = 0
        if not hasattr(self, "_task_type"):
            self._task_type = TASK_ARITHMETIC

    def _task_name(self):
        return "arithmetic" if self._task_type == TASK_ARITHMETIC else "lookup"

    def _obs(self):
        return np.array(
            [
                float(self._task_type),
                float(self._a),
                float(self._b),
                float(self._op_id),
                float(self._key),
                1.0 if self._answer_known else 0.0,
                float(self._known_answer),
            ],
            dtype=np.float32,
        )

    def _info(self):
        return {
            "task_type": self._task_name(),
            "true_answer": self._true_answer,
            "correct_tool": self._correct_tool,
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
