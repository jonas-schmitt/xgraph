"""Identify executed GPU admission scripts, without matching readers or fixtures."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

INTERPRETERS = {"ash", "bash", "dash", "ksh", "sh", "zsh", "perl", "ruby"}


def repository_roots(repo):
    """The repository and its registered worktrees, including external worktrees."""
    repo = Path(repo).resolve()
    common = repo / ".git"
    if common.is_file():
        gitdir = common.read_text().removeprefix("gitdir: ").strip()
        metadata = (repo / gitdir).resolve()
        common = (metadata / (metadata / "commondir").read_text().strip()).resolve()
    roots = {repo, common.parent}
    for gitdir in (common / "worktrees").glob("*/gitdir"):
        roots.add(Path(gitdir.read_text().strip()).parent.resolve())
    return roots


def executing_pids(repo, *, proc_root=Path("/proc"), include_scripts=()):
    """PIDs executing an exact repository script; inventory errors reach the caller.

    Interpreters execute argv[1]; direct executions use argv[0]. A path in any later
    argument, including a shell command string, is never an executing script.
    Relative paths resolve against that process's cwd, then through symlinks.
    """
    scripts = {(root / "scripts/gpu-headroom.sh").resolve() for root in repository_roots(repo)}
    scripts.update(Path(script).resolve() for script in include_scripts)
    script_names = {"gpu-headroom.sh", *(script.name for script in scripts)}
    pids = []
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
            if not args[0]:
                continue
            executable = Path(os.fsdecode(args[0])).name
            interpreted = executable in INTERPRETERS or re.fullmatch(
                r"(?:python|pypy)[\d.]*", executable
            )
            position = 1 if interpreted else 0
            if len(args) <= position or not args[position]:
                continue
            if interpreted and args[position].startswith(b"-"):
                continue  # options such as -c, -m and -n are not a script path
            script = Path(os.fsdecode(args[position]))
            try:
                if not script.is_absolute():
                    script = (process / "cwd").resolve(strict=True) / script
                if script.resolve() in scripts:
                    pids.append(int(process.name))
            except PermissionError:
                if script.name not in script_names:
                    # Protected service paths do not identify an admission script.
                    # Known script filenames retain fail-closed handling.
                    continue
                raise
        except FileNotFoundError:  # a process exited during the snapshot
            continue
    return sorted(pids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--include-script", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    try:
        pids = executing_pids(
            args.repo, proc_root=args.proc_root, include_scripts=args.include_script
        )
    except (OSError, RuntimeError) as error:
        parser.exit(2, f"cannot inspect GPU waiters: {error}\n")
    for pid in pids:
        print(pid)
    return 0 if pids else 1


if __name__ == "__main__":
    raise SystemExit(main())
