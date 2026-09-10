"""Paper-only process supervisor: respawn polywang and flush health on death.

QUANT-20260909-01. Additive — an already-running ``uv run polywang`` paper
process does not need to be stopped to deploy this wrapper.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import time
from typing import Callable, List, Optional, Sequence

from .monitor import (
    DEFAULT_EXCEPTIONS_LOG,
    DEFAULT_SUPERVISOR_STOP_FILE,
    mark_health_stopped,
    next_backoff_seconds,
    record_process_exit,
    stop_flag_is_set,
)

LOG = logging.getLogger("paper-supervisor")
DEFAULT_COMMAND = ("uv", "run", "polywang")


def refuse_live_command(command: Sequence[str]) -> List[str]:
    """Paper supervisor must never start live trading."""
    argv = [str(part) for part in command]
    if "--live" in argv:
        raise ValueError("paper supervisor refuses --live; do not enable live trading")
    return argv


def build_command(extra: Optional[Sequence[str]] = None, environ: Optional[dict] = None) -> List[str]:
    env = os.environ if environ is None else environ
    raw = str(env.get("PAPER_SUPERVISOR_CMD") or "").strip()
    command = raw.split() if raw else list(DEFAULT_COMMAND)
    command.extend(str(part) for part in (extra or ()))
    return refuse_live_command(command)


def describe_child_exit(returncode: Optional[int]) -> tuple[Optional[int], str, str]:
    """Map wait() status onto exit_code + halt_reason + last_error."""
    if returncode is None:
        return None, "child exited (unknown code)", "child process disappeared"
    code = int(returncode)
    if code < 0:
        sig = -code
        return code, f"child killed by signal {sig}", f"signal {sig}"
    return code, f"child exited with code {code}", f"exit {code}"


class PaperSupervisor:
    """Run one paper child at a time; flush stopped health when it dies."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        health_path: str = "live-health.json",
        stop_file: str = DEFAULT_SUPERVISOR_STOP_FILE,
        exceptions_path: str = DEFAULT_EXCEPTIONS_LOG,
        backoff_start: float = 1.0,
        backoff_max: float = 30.0,
        stable_seconds: float = 60.0,
        poll_seconds: float = 1.0,
        kill_grace_seconds: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
        popen: Callable[..., object] = subprocess.Popen,
        environ: Optional[dict] = None,
        max_restarts: Optional[int] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.command = refuse_live_command(command)
        self.health_path = health_path
        self.stop_file = stop_file
        self.exceptions_path = exceptions_path
        self.backoff_start = float(backoff_start)
        self.backoff_max = float(backoff_max)
        self.stable_seconds = float(stable_seconds)
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.kill_grace_seconds = max(0.0, float(kill_grace_seconds))
        self.sleep = sleep
        self.popen = popen
        self.environ = os.environ if environ is None else environ
        self.max_restarts = max_restarts
        self.clock = clock
        self._stop = False
        self.child = None
        self.starts = 0

    def request_stop(self, *_args) -> None:
        self._stop = True
        child = self.child
        if child is None:
            return
        try:
            child.send_signal(signal.SIGTERM)
        except (OSError, ProcessLookupError, AttributeError):
            return

    def stop_requested(self) -> bool:
        return self._stop or stop_flag_is_set(self.stop_file, self.environ)

    def note_child_death(
        self,
        returncode: Optional[int],
        pid: Optional[int],
        *,
        now: Optional[float] = None,
    ) -> dict:
        code, halt_reason, last_error = describe_child_exit(returncode)
        payload = mark_health_stopped(
            self.health_path,
            exit_code=code,
            last_error=last_error,
            halt_reason=halt_reason,
            now=now,
            extra={"pid": pid} if pid is not None else None,
        )
        record_process_exit(
            halt_reason,
            path=self.exceptions_path,
            source="paper-supervisor",
            extra={"exit_code": code, "pid": pid, "halt_reason": halt_reason},
            now=now,
        )
        return payload

    def _spawn(self):
        kwargs = {"stdin": subprocess.DEVNULL, "start_new_session": True}
        try:
            return self.popen(self.command, **kwargs)
        except TypeError:
            return self.popen(self.command)

    def _wait_child(self, child) -> Optional[int]:
        poll = getattr(child, "poll", None)
        if not callable(poll):
            return child.wait()
        sent_term = False
        term_at = None
        while True:
            code = poll()
            if code is not None:
                return code
            if self.stop_requested():
                if not sent_term:
                    self.request_stop()
                    sent_term = True
                    term_at = self.clock()
                elif self.clock() - float(term_at or 0.0) >= self.kill_grace_seconds:
                    killer = getattr(child, "kill", None)
                    if callable(killer):
                        try:
                            killer()
                        except (OSError, ProcessLookupError):
                            pass
            self.sleep(self.poll_seconds)

    def run(self) -> int:
        backoff = 0.0
        last_code = 0
        while True:
            if self.stop_requested():
                if self.starts == 0:
                    mark_health_stopped(
                        self.health_path,
                        exit_code=0,
                        last_error="supervisor stop",
                        halt_reason="supervisor stop flag",
                    )
                return last_code
            if self.max_restarts is not None and self.starts >= self.max_restarts:
                return last_code
            self.starts += 1
            started = self.clock()
            LOG.info("starting paper child (%s): %s", self.starts, " ".join(self.command))
            child = self._spawn()
            self.child = child
            try:
                returncode = self._wait_child(child)
            except KeyboardInterrupt:
                self.request_stop()
                try:
                    returncode = self._wait_child(child)
                except Exception:
                    returncode = getattr(child, "returncode", None)
            self.child = None
            last_code = 0 if returncode is None else int(returncode)
            pid = getattr(child, "pid", None)
            self.note_child_death(returncode, pid)
            if self.stop_requested():
                LOG.info("stop flag set; not respawning paper child")
                return last_code
            lived = self.clock() - started
            if lived >= self.stable_seconds:
                backoff = 0.0
            delay = next_backoff_seconds(
                backoff, start=self.backoff_start, maximum=self.backoff_max,
            )
            backoff = delay
            LOG.warning("paper child exited %s; respawn in %.1fs", returncode, delay)
            self.sleep(delay)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Respawn paper polywang and flush live-health on unexpected death",
    )
    parser.add_argument(
        "child_args",
        nargs="*",
        help="Extra args forwarded to `uv run polywang` (never pass --live)",
    )
    parser.add_argument(
        "--health-path",
        default=os.getenv("LIVE_HEALTH_PATH", "live-health.json"),
    )
    parser.add_argument(
        "--stop-file",
        default=os.getenv("PAPER_SUPERVISOR_STOP_FILE", DEFAULT_SUPERVISOR_STOP_FILE),
    )
    parser.add_argument(
        "--exceptions-log",
        default=os.getenv("MONITOR_EXCEPTIONS_LOG", DEFAULT_EXCEPTIONS_LOG),
    )
    parser.add_argument(
        "--backoff-start",
        type=float,
        default=float(os.getenv("PAPER_SUPERVISOR_BACKOFF_START", "1")),
    )
    parser.add_argument(
        "--backoff-max",
        type=float,
        default=float(os.getenv("PAPER_SUPERVISOR_BACKOFF_MAX", "30")),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        command = build_command(args.child_args)
    except ValueError as error:
        LOG.error("%s", error)
        return 2
    supervisor = PaperSupervisor(
        command,
        health_path=args.health_path,
        stop_file=args.stop_file,
        exceptions_path=args.exceptions_log,
        backoff_start=args.backoff_start,
        backoff_max=args.backoff_max,
    )
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    for sig in signals:
        signal.signal(sig, supervisor.request_stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
