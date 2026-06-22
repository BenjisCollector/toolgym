# toolgym

A small Gymnasium-style reinforcement-learning environment where an agent must
learn to **use tools** to solve a task, rather than answer directly.

Demonstrates: RL environment design for LLM tool use (agentic RL).

**Status: early scaffold**

## Why

Modern agentic models are trained to decide *when and which tool to call* — to
gather information before committing to an answer. `toolgym` is a deliberately
tiny sandbox for that decision: the agent sees an arithmetic query but is not
rewarded for "knowing" the answer. It is rewarded for invoking a tool to obtain
the answer and only then submitting it. The arithmetic is incidental; the
learnable skill is the **policy of tool use**.

## The environment

`ToolUseEnv` (in `toolgym/env.py`) implements the Gymnasium API: `reset`,
`step`, `observation_space`, and `action_space`. If `gymnasium` is installed it
subclasses `gymnasium.Env` and registers as `ToolUse-v0`; if not, it falls back
to a tiny built-in shim so the env and the random agent still run with only
`numpy`.

### MDP

**State / observation** — `float32` vector of length 5:

| index | field           | meaning                                            |
|-------|-----------------|----------------------------------------------------|
| 0     | `operand_a`     | first operand of the query                         |
| 1     | `operand_b`     | second operand                                     |
| 2     | `op_id`         | operator: `0` = `+`, `1` = `-`, `2` = `*`          |
| 3     | `answer_known`  | `1.0` once any tool has been used, else `0.0`      |
| 4     | `known_answer`  | the tool-provided answer, or `0.0` before any tool |

**Actions** — `Discrete(3)`:

| id | action       | effect                                                        |
|----|--------------|--------------------------------------------------------------|
| 0  | `CALCULATOR` | reveals the answer in the observation; costs `-tool_cost`    |
| 1  | `LOOKUP`     | reveals the answer in the observation; costs `-tool_cost`    |
| 2  | `SUBMIT`     | commits the currently-known answer; ends the episode         |

**Reward**

- `CALCULATOR` / `LOOKUP`: `-tool_cost` (default `-0.1`) — information has a price.
- `SUBMIT` with a correct, tool-derived answer: `+1.0`, episode terminates.
- `SUBMIT` without ever using a tool, or with a wrong answer: `-1.0`, terminates.
- More than `max_steps` (default `5`) tool calls without submitting: the episode
  truncates with no terminal bonus.

The optimal policy is therefore: call one tool, then submit — expected return
`+1.0 - tool_cost`. Submitting blind is punished, so the agent must learn to
*use a tool first*.

## Install

```bash
pip install -r requirements.txt   # gymnasium, numpy
```

`numpy` alone is enough to run the example (the env shims Gymnasium when absent).

## Run

```bash
python examples/random_agent.py        # 10 episodes
python examples/random_agent.py 50     # custom episode count
```

This rolls out a uniform-random policy and prints per-episode and mean reward.
A random policy lands well below the optimal `~+0.9`, which is the point: the
gap is what a learning agent would close.

## Layout

```
toolgym/
  toolgym/
    __init__.py
    env.py            # ToolUseEnv (Gymnasium API)
  examples/
    random_agent.py   # runnable random-policy rollout
  requirements.txt
  README.md
  LICENSE
  .gitignore
```

## License

MIT — see [LICENSE](LICENSE).
