# toolgym

A small Gymnasium-style reinforcement-learning environment where an agent must
learn to **use the right tool** to solve a task, rather than answer directly.

**Demonstrates RL environment design for agentic tool use.**

## Why

Modern agentic models are trained to decide *when and which tool to call* — to
gather information before committing to an answer. `toolgym` is a deliberately
tiny sandbox for exactly that decision. The agent sees a query but is **not**
rewarded for "knowing" the answer. It is rewarded for invoking the *correct*
tool to obtain the answer and only then submitting it.

The twist that makes this a real learning problem: there are two task types and
two tools, and **the right tool depends on the task**. The agent must read the
task type from its observation and pick the matching tool. It cannot win by
always pressing the same button.

## The environment

`ToolUseEnv` (in [`toolgym/env.py`](toolgym/env.py)) implements the Gymnasium
API: `reset`, `step`, `observation_space`, and `action_space`. If `gymnasium` is
installed it subclasses `gymnasium.Env` and registers as `ToolUse-v0`; if not,
it falls back to a tiny built-in shim so the env, tests, and baselines run with
**numpy alone**.

### Task variants

| variant        | query                          | correct tool  |
|----------------|--------------------------------|---------------|
| `arithmetic`   | `a op b = ?` (`+`, `-`, `*`)   | `CALCULATOR`  |
| `lookup`       | `value of key k?` (k/v table)  | `LOOKUP`      |
| `mixed` (default) | either of the above, sampled | task-dependent |

Calling the **wrong** tool for a task returns a deterministic garbage value and
does **not** reveal the answer — so submitting after the wrong tool is punished.

### MDP

**State / observation** — `float32` vector of length 7:

| index | field          | meaning                                              |
|-------|----------------|------------------------------------------------------|
| 0     | `task_type`    | `0` = arithmetic, `1` = lookup                       |
| 1     | `operand_a`    | first operand (arithmetic tasks; `0` otherwise)      |
| 2     | `operand_b`    | second operand (arithmetic tasks; `0` otherwise)     |
| 3     | `op_id`        | operator: `0` = `+`, `1` = `-`, `2` = `*`            |
| 4     | `lookup_key`   | key id (lookup tasks; `0` otherwise)                 |
| 5     | `answer_known` | `1.0` once the *correct* tool revealed the answer    |
| 6     | `known_answer` | the answer currently held, or `0.0` before any tool  |

**Actions** — `Discrete(3)`:

| id | action       | effect                                                          |
|----|--------------|----------------------------------------------------------------|
| 0  | `CALCULATOR` | correct for arithmetic; wrong (garbage) for lookup; `-tool_cost` |
| 1  | `LOOKUP`     | correct for lookup; wrong (garbage) for arithmetic; `-tool_cost` |
| 2  | `SUBMIT`     | commits the currently-known answer; ends the episode           |

**Reward**

- Any tool call: `-tool_cost` (default `-0.1`) — information has a price.
- `SUBMIT` with the correct, tool-derived answer: `+1.0`, episode terminates.
- `SUBMIT` after no tool, the wrong tool, or with a wrong value: `-1.0`, ends.
- Reaching `max_steps` (default `5`) without submitting: the episode truncates
  with no terminal bonus.

**The optimal policy** reads `task_type`, calls the matching tool once, then
submits — expected return `+1.0 - tool_cost = +0.9`. Submitting blind, or after
the wrong tool, is punished, so the agent must learn to *select the right tool
first*.

## Install

```bash
pip install -r requirements.txt   # gymnasium, numpy
```

`numpy` alone is enough to run everything (the env shims Gymnasium when absent).
The tests and the Q-learning baseline need only `numpy` (+ `pytest` for tests).

## Run

### Random baseline

```bash
python examples/random_agent.py        # 10 episodes
python examples/random_agent.py 50     # custom episode count
```

### Tabular Q-learning baseline

A working baseline that **learns the tool-use policy and beats random**:

```bash
python examples/train_qlearning.py        # 4000 training episodes
python examples/train_qlearning.py 1500   # custom episode count
```

It discretizes the observation to the decision-relevant features
`(task_type, answer_known)` — 4 states — and learns the Q-table. Sample output:

```
learned greedy policy (state -> action):
  [arithmetic, answer unknown] -> CALCULATOR
  [arithmetic, answer known  ] -> SUBMIT
  [lookup,     answer unknown] -> LOOKUP
  [lookup,     answer known  ] -> SUBMIT
----------------------------------------------------
random  policy mean reward: -0.508
learned policy mean reward: +0.900
theoretical optimum:        +0.900
improvement over random:    +1.408
RESULT: learned policy beats random.
```

The learned policy recovers the optimal tool-selection rule and reaches the
theoretical optimum, a large improvement over the random policy's negative
return.

## Test

```bash
pytest -q
```

Tests are numpy-only (no gymnasium required) and cover the reset/step
contracts, the reward logic (correct-tool-then-submit is positive; blind submit
and wrong-tool submit are punished), and a smoke test that the Q-learning
baseline beats random.

## Layout

```
toolgym/
  toolgym/
    __init__.py
    env.py                # ToolUseEnv (Gymnasium API, 3 task variants)
  examples/
    random_agent.py       # uniform-random rollout
    train_qlearning.py    # tabular Q-learning baseline that beats random
  tests/
    test_env.py           # pytest, numpy-only
  .github/workflows/ci.yml
  conftest.py
  requirements.txt
  README.md
  LICENSE
  .gitignore
```

## License

MIT — see [LICENSE](LICENSE).
