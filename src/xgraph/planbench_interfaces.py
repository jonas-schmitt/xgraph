"""Planning interfaces sharing PDDL information, execution and resource budgets.

The model chooses termination. No primary interface receives a goal-test result,
gold plan or optimal length. Budget and context exhaustion remain incomplete.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

from xgraph.planbench_pddl import PlanSyntaxError, check_plan, load_problem, parse_plan, sexpr

ARMS = ("free", "resume", "feedback", "menu", "mask", "llm_planner", "random")


@dataclass(frozen=True)
class Budget:
    max_actions: int = 64
    max_calls: int = 65
    generated_tokens: int = 2048
    prompt_tokens: int = 262144
    context_tokens: int = 16384

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("every planning budget must be a positive integer")


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str = "stop"
    token_ids: tuple[int, ...] = ()


def _base(problem, budget):
    return [
        {
            "role": "user",
            "content": (
                "Solve this classical planning problem using only its PDDL semantics. "
                "Write only lower-case ground actions, one per line, "
                "separated by a single newline. "
                "End the plan by ending your response, including an empty response if desired. "
                "Do not write numbering, prose, code fences, or STOP. "
                f"The action limit is {budget.max_actions}.\n[DOMAIN]\n{problem.domain_text}"
                f"\n[PROBLEM]\n{problem.problem_text}"
            ),
        }
    ]


def _observation(problem, state, prefix, error=None):
    # Complete, unranked support; no gold, optimal length, goal status, pruning or pages.
    return {
        "executed_prefix": list(prefix),
        "state": [list(f) for f in sorted(state)],
        "applicable_actions": [a.text for a in problem.applicable(state)],
        "error": error,
    }


def _replace_goal(problem, text):
    goal = sexpr(text)
    if len(goal) != 2 or goal[0] != ":goal":
        raise ValueError("goal writer must return exactly (:goal <condition>)")
    root = sexpr(problem.problem_text)
    root = [
        goal if isinstance(node, list) and node and node[0] == ":goal" else node for node in root
    ]

    def render(node):
        return "(" + " ".join(render(x) for x in node) + ")" if isinstance(node, list) else node

    return load_problem(problem.domain_text, render(root))


def run_episode(
    problem, arm, backend, budget=None, *, seed=16, mask_arguments=None, planner_fn=None
):
    budget = budget or Budget()
    if arm not in (*ARMS, "random-goal-reached-control"):
        raise ValueError(f"unknown planning arm: {arm}")
    if arm == "mask" and mask_arguments is None:
        raise ValueError("mask arm requires explicit planning request arguments")
    started, cpu = time.perf_counter(), time.process_time()
    messages = _base(problem, budget)
    if arm in ("menu", "feedback"):
        messages.append(
            {
                "role": "user",
                "content": (
                    "For this interactive interface, write exactly one offered action or STOP."
                    if arm == "menu"
                    else "For this feedback interface, write a chunk of actions to execute, "
                    "or STOP. "
                    "Each chunk receives execution feedback."
                ),
            }
        )
    if arm == "llm_planner":
        messages.append(
            {
                "role": "user",
                "content": (
                    "For the LLM-to-planner reference, return only (:goal <condition>) "
                    "expressing the "
                    "problem's stated goal. A classical planner will solve the resulting PDDL."
                ),
            }
        )
    prefix, turns, menus = [], [], []
    state = problem.init
    prompt_tokens = generated_tokens = invalid = 0
    complete, termination, error = False, "call_budget", None
    final_text, planner = "", None
    if arm.startswith("random"):
        rng = random.Random(seed)
        while len(prefix) < budget.max_actions:
            if arm == "random-goal-reached-control" and problem.goal(state):
                complete, termination = True, "goal-reached-control"
                break
            action = rng.choice([*problem.applicable(state), None])
            if action is None:
                complete, termination = True, "policy_stop"
                break
            prefix.append(action.text)
            state = action.apply(state)
        else:
            termination = "action_budget"
        final_text = "\n".join(prefix)
    else:
        while len(turns) < budget.max_calls:
            if len(prefix) >= budget.max_actions:
                termination = "action_budget"
                break
            if arm in ("menu", "feedback"):
                observation = _observation(problem, state, prefix, error)
                menus.append(observation["applicable_actions"])
                messages.append(
                    {"role": "user", "content": "Execution observation: " + json.dumps(observation)}
                )
            expected = backend.count_prompt(messages)
            if expected + prompt_tokens > budget.prompt_tokens:
                termination = "prompt_budget"
                break
            allowance = min(
                budget.generated_tokens - generated_tokens, budget.context_tokens - expected
            )
            if allowance <= 0:
                termination = (
                    "context_budget" if expected >= budget.context_tokens else "token_budget"
                )
                break
            before = time.perf_counter()
            reply = backend.complete(
                messages, allowance, mask_arguments=(mask_arguments if arm == "mask" else None)
            )
            if (
                reply.prompt_tokens != expected
                or type(reply.generated_tokens) is not int
                or not (0 <= reply.generated_tokens <= allowance)
            ):
                raise ValueError("backend token usage disagrees with admission budget")
            prompt_tokens += reply.prompt_tokens
            generated_tokens += reply.generated_tokens
            turns.append(
                {
                    "messages": list(messages),
                    "text": reply.text,
                    "prompt_tokens": reply.prompt_tokens,
                    "generated_tokens": reply.generated_tokens,
                    "finish_reason": reply.finish_reason,
                    "token_ids": list(reply.token_ids),
                    "max_tokens": allowance,
                    "seconds": time.perf_counter() - before,
                    "masked": arm == "mask",
                }
            )
            messages.append({"role": "assistant", "content": reply.text})
            if reply.finish_reason != "stop":
                final_text = "\n".join([*prefix, reply.text])
                termination = "token_budget" if reply.finish_reason == "length" else "backend_abort"
                break
            if arm == "llm_planner":
                from xgraph.planbench_reference import optimal_plan

                try:
                    written = _replace_goal(problem, reply.text)
                except ValueError as exc:
                    error, termination = str(exc), "invalid_goal"
                    break
                planner = (planner_fn or optimal_plan)(written)
                final_text = "\n".join(planner["plan"])
                complete = planner["status"] == "optimal"
                termination = "model_and_planner" if complete else "planner_" + planner["status"]
                break
            if arm in ("menu", "feedback") and reply.text.strip().upper() in ("", "STOP"):
                complete, termination = True, "model"
                final_text = "\n".join(prefix)
                break
            if arm == "menu":
                try:
                    choice = parse_plan(reply.text)
                except PlanSyntaxError:
                    choice = ()
                if len(choice) != 1 or choice[0] not in menus[-1]:
                    invalid += 1
                    error = "Choose one action from the complete applicable menu, or STOP."
                    continue
            candidate = "\n".join([*prefix, reply.text])
            try:
                parsed = parse_plan(candidate)
            except PlanSyntaxError as exc:
                parsed = exc.prefix
            if len(parsed) >= budget.max_actions:
                candidate = "\n".join(parsed[: budget.max_actions])
                termination = "action_budget"
                final_text = candidate
                break
            checked = check_plan(problem, candidate)
            prefix, state = list(checked.prefix), checked.state
            final_text = candidate
            if not checked.valid:
                invalid += 1
                error = checked.error
            else:
                error = None
            if arm in ("free", "mask") or (arm == "resume" and checked.valid):
                complete, termination = True, "model"
                break
            if arm == "resume":
                observation = _observation(problem, state, prefix, error)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Execution stopped at the first unavailable action. "
                            "Freeze the executed prefix. Write only its remaining suffix; "
                            "an empty response ends at that prefix.\n" + json.dumps(observation)
                        ),
                    }
                )
        else:
            termination = "call_budget"
    checked = check_plan(problem, final_text)
    if arm == "llm_planner" and error:
        complete = False
    if checked.length >= budget.max_actions:
        complete, termination = False, "action_budget"
    return {
        "arm": arm,
        "plan": list(checked.prefix) if checked.valid else final_text.splitlines(),
        "executed_prefix": list(checked.prefix),
        "plan_valid": checked.valid,
        "goal_success": complete and checked.goal_reached,
        "complete": complete,
        "termination": termination,
        "length": checked.length,
        "error": error or checked.error,
        "failed_step": checked.failed_step,
        "model_calls": len(turns),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "invalid_attempts": invalid,
        "menus": menus,
        "turns": turns,
        "planner": planner,
        "budget": asdict(budget),
        "seed": seed,
        "seconds": time.perf_counter() - started,
        "client_cpu_seconds": time.process_time() - cpu,
    }
