"""Run expert-stage child commands with a complete log and visible heartbeat."""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from grond.paths import PACKAGE_ROOT


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _manual_stream_line(line: str, *, verbose: bool) -> bool:
    """Return whether a child line belongs on the concise controller stream."""

    if verbose:
        return True
    stripped = line.lstrip()
    return stripped.startswith(("STEP", "DONE", "FAILED", "WARN", "WAIT", "RERUN", "ERROR"))


def _run_stage_command(command: list[str], *, verbose: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    env = dict(os.environ)
    package_src = str(PACKAGE_ROOT.parent)
    inherited = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = package_src + (f"{os.pathsep}{inherited}" if inherited else "")
    env["PYTHONUNBUFFERED"] = "1"
    heartbeat_seconds = max(
        0.01,
        float(env.get("GROND_MANUAL_HEARTBEAT_SECONDS", "30") or 30.0),
    )
    out_dir: Path | None = None
    if "--out-dir" in command:
        out_dir = Path(command[command.index("--out-dir") + 1]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        env["GROND_CACHE_DIR"] = str(out_dir.parent / "cache")
    if len(command) > 3 and command[1:3] == ["-m", "grond.cli"]:
        child_name = command[3]
    elif len(command) > 1:
        child_name = Path(command[1]).stem
    else:
        child_name = Path(command[0]).name
    log_path = (out_dir or Path.cwd()) / f"manual_{child_name.replace('-', '_')}.log"
    print(f"RUN  {child_name} -> {out_dir or Path.cwd()}", flush=True)
    continuing_log = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a" if continuing_log else "w") as log:
        if continuing_log:
            log.write("\n[manual_resume]\n")
        log.write(f"$ {_command_text(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        while True:
            try:
                line = output_queue.get(timeout=heartbeat_seconds)
            except queue.Empty:
                running_seconds = time.perf_counter() - started
                print(
                    f"WAIT {child_name} still running ({running_seconds:.0f}s); controller log {log_path}",
                    flush=True,
                )
                continue
            if line is None:
                break
            if _manual_stream_line(line, verbose=verbose):
                print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        reader.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    record = {
        "command": command,
        "command_text": _command_text(command),
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed),
        "log": str(log_path.resolve()),
    }
    if returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(
            f"Manual stage command failed with return code {returncode}: "
            f"{record['command_text']}\nLog: {log_path}\nLast log lines:\n{tail}"
        )
    print(f"DONE {child_name} in {elapsed:.1f}s; log {log_path}", flush=True)
    return record


def _command_elapsed(command: Mapping[str, object]) -> float | None:
    elapsed_values: list[float] = []

    def collect(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("elapsed_seconds") is not None:
            elapsed_values.append(float(value["elapsed_seconds"]))
            return
        for nested in value.values():
            collect(nested)

    collect(command)
    return sum(elapsed_values) if elapsed_values else None


def _child_base(args) -> list[str]:
    python = str(getattr(args, "python", "") or sys.executable)
    return [python, "-m", "grond.cli"]


def _tool_args(args) -> list[str]:
    out: list[str] = []
    config = str(getattr(args, "config", "") or "").strip()
    if config:
        out += ["--config", str(Path(config).expanduser().resolve())]
    if bool(getattr(args, "allow_missing_tools", False)):
        out.append("--allow-missing-tools")
    return out


def _verbose_child_output(args) -> bool:
    return bool(getattr(args, "verbose_child_output", False))
