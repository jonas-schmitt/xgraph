"""Independent VAL validation and bounded optimal classical planning references.

Both planners use their own PDDL parser, grounding and A*/admissible LM-cut with
unit costs. Timeouts have no optimum; gold lengths are never substituted. Search
runs in memory-limited children; a timeout kills the entire planner process group.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from xgraph.planbench_pddl import Problem, parse_plan


def optimal_plan(
    problem: Problem,
    *,
    seconds: float = 10,
    memory_mib: int = 2048,
    fast_downward: Path | None = None,
) -> dict:
    if seconds <= 0 or memory_mib < 64:
        raise ValueError("planner needs positive seconds and at least 64 MiB")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="planbench-planner-") as folder:
        root = Path(folder)
        (root / "domain.pddl").write_text(problem.domain_text)
        (root / "problem.pddl").write_text(problem.problem_text)
        command = [
            sys.executable,
            "-m",
            __name__,
            "--worker",
            str(root / "domain.pddl"),
            str(root / "problem.pddl"),
            "--memory-mib",
            str(memory_mib),
        ]
        if fast_downward is not None:
            command.extend(["--fast-downward", str(fast_downward.resolve())])
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
        )
        run = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = run.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(run.pid, signal.SIGKILL)
            run.communicate()
            result = {"status": "timeout", "optimal_length": None, "plan": []}
        else:
            result = (
                json.loads(stdout)
                if run.returncode == 0
                else {
                    "status": "error",
                    "optimal_length": None,
                    "plan": [],
                    "error": stderr[-2000:],
                }
            )
    return {
        **result,
        "seconds": time.monotonic() - started,
        "planner": ("fast-downward-24.06.1" if fast_downward else "pyperplan-2.1")
        + "/astar/lmcut/unit-cost",
        "time_limit": seconds,
        "memory_mib": memory_mib,
    }


def validate_with_val(binary: Path, domain: Path, problem: Path, plan: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="planbench-val-") as folder:
        path = Path(folder) / "plan.txt"
        path.write_text(plan + "\n")
        run = subprocess.run(
            [str(binary.resolve()), str(domain), str(problem), str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    output = run.stdout + run.stderr
    # VAL returns zero for some invalid plans; process status alone proves nothing.
    if any(s in output for s in ("Plan invalid", "Plan failed to execute", "Bad operator in plan")):
        return False
    if run.returncode != 0 or "Plan valid" not in output:
        raise RuntimeError(f"VAL failed to validate: {output[-2000:]}")
    return True


def _worker(domain: str, problem: str, memory_mib: int, fast_downward=None) -> dict:
    resource.setrlimit(resource.RLIMIT_AS, (memory_mib * 1024**2, memory_mib * 1024**2))
    if fast_downward:
        folder = Path(problem).parent
        run = subprocess.run(
            [
                sys.executable,
                fast_downward,
                "--overall-memory-limit",
                str(memory_mib) + "M",
                "--plan-file",
                "sas_plan",
                domain,
                problem,
                "--search",
                "astar(lmcut())",
            ],
            cwd=folder,
            capture_output=True,
            text=True,
        )
        path = folder / "sas_plan"
        if run.returncode == 0 and path.exists():
            plan = list(parse_plan(path.read_text()))
            return {
                "status": "optimal",
                "optimal_length": len(plan),
                "plan": plan,
                "worker_peak_rss_mib": max(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                )
                / 1024,
            }
        return {
            "status": "unsolvable" if run.returncode == 20 else "error",
            "optimal_length": None,
            "plan": [],
            "exit_code": run.returncode,
            "error": (run.stdout + run.stderr)[-3000:],
        }
    from pyperplan.heuristics.lm_cut import LmCutHeuristic
    from pyperplan.planner import search_plan
    from pyperplan.search import astar_search

    plan = search_plan(domain, problem, astar_search, LmCutHeuristic)
    return {
        "status": "optimal" if plan is not None else "unsolvable",
        "optimal_length": len(plan) if plan is not None else None,
        "plan": [a.name for a in plan] if plan is not None else [],
        "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, required=True)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--fast-downward")
    args = parser.parse_args()
    print(json.dumps(_worker(*args.worker, args.memory_mib, args.fast_downward)))
