#!/usr/bin/env python3
"""QUANT-20260909-01: paper supervisor respawn / stop-flag / health flush."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from polywang.arbitrage_bot import run_health_loop
from polywang.arbitrage_core import BinaryArbitrageScanner
from polywang.monitor import HealthPublisher
from polywang.paper_supervisor import (
    PaperSupervisor,
    build_command,
    describe_child_exit,
    refuse_live_command,
)


class FakeChild:
    def __init__(self, returncode=3, pid=4242, delay_polls=0):
        self.returncode = returncode
        self.pid = pid
        self.delay_polls = delay_polls
        self.polls = 0
        self.signals = []
        self.killed = False

    def poll(self):
        self.polls += 1
        if self.killed:
            self.returncode = -9
            return self.returncode
        if self.delay_polls > 0:
            self.delay_polls -= 1
            return None
        return self.returncode

    def wait(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.delay_polls = 0

    def kill(self):
        self.killed = True


class PaperSupervisorTests(unittest.TestCase):
    def test_refuses_live_flag(self):
        with self.assertRaisesRegex(ValueError, "--live"):
            refuse_live_command(["uv", "run", "polywang", "--live"])
        with self.assertRaisesRegex(ValueError, "--live"):
            build_command(["--live"])
        command = build_command(["--markets", "20"], environ={})
        self.assertEqual(command[:3], ["uv", "run", "polywang"])
        self.assertNotIn("--live", command)

    def test_describe_signal_exit(self):
        code, reason, error = describe_child_exit(-9)
        self.assertEqual(code, -9)
        self.assertIn("signal 9", reason)
        self.assertEqual(error, "signal 9")
        code, reason, error = describe_child_exit(1)
        self.assertEqual(code, 1)
        self.assertIn("code 1", reason)

    def test_unexpected_exit_respawns_until_stop_file(self):
        with tempfile.TemporaryDirectory() as directory:
            health_path = os.path.join(directory, "live-health.json")
            exceptions_path = os.path.join(directory, "monitor-exceptions.jsonl")
            stop_file = os.path.join(directory, "paper-supervisor.stop")
            with open(health_path, "w", encoding="utf-8") as handle:
                json.dump({"status": "running", "open_negrisk": 2, "pid": 7}, handle)
            children = []

            def popen(command, **_kwargs):
                child = FakeChild(returncode=3, pid=1000 + len(children))
                children.append(child)
                return child

            def sleep(_seconds):
                if len(children) >= 2:
                    with open(stop_file, "w", encoding="utf-8") as handle:
                        handle.write("stop\n")

            supervisor = PaperSupervisor(
                [sys.executable, "-c", "pass"],
                health_path=health_path,
                stop_file=stop_file,
                exceptions_path=exceptions_path,
                backoff_start=0.01,
                backoff_max=0.05,
                poll_seconds=0.01,
                sleep=sleep,
                popen=popen,
                environ={},
                max_restarts=5,
            )
            code = supervisor.run()
            self.assertEqual(code, 3)
            self.assertEqual(len(children), 2)
            with open(health_path, encoding="utf-8") as handle:
                health = json.load(handle)
            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["exit_code"], 3)
            self.assertTrue(health["halt_reason"])
            self.assertTrue(health["last_error"])
            self.assertEqual(health["open_negrisk"], 2)
            with open(exceptions_path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertGreaterEqual(len(rows), 2)
            self.assertTrue(all(row["kind"] == "process_exit" for row in rows))

    def test_stop_file_before_start_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            stop_file = os.path.join(directory, "paper-supervisor.stop")
            health_path = os.path.join(directory, "live-health.json")
            with open(stop_file, "w", encoding="utf-8") as handle:
                handle.write("stop\n")
            started = []

            def popen(command, **_kwargs):
                started.append(command)
                raise AssertionError("should not spawn when stop file exists")

            supervisor = PaperSupervisor(
                ["uv", "run", "polywang"],
                health_path=health_path,
                stop_file=stop_file,
                exceptions_path=os.path.join(directory, "monitor-exceptions.jsonl"),
                sleep=lambda _s: None,
                popen=popen,
                environ={},
            )
            self.assertEqual(supervisor.run(), 0)
            self.assertEqual(started, [])
            with open(health_path, encoding="utf-8") as handle:
                health = json.load(handle)
            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["halt_reason"], "supervisor stop flag")

    def test_stop_file_after_kill_does_not_respawn(self):
        with tempfile.TemporaryDirectory() as directory:
            health_path = os.path.join(directory, "live-health.json")
            exceptions_path = os.path.join(directory, "monitor-exceptions.jsonl")
            stop_file = os.path.join(directory, "paper-supervisor.stop")
            started = []

            def popen(command, **_kwargs):
                started.append(command)
                with open(stop_file, "w", encoding="utf-8") as handle:
                    handle.write("stop\n")
                return FakeChild(returncode=-9, pid=99)

            supervisor = PaperSupervisor(
                ["uv", "run", "polywang"],
                health_path=health_path,
                stop_file=stop_file,
                exceptions_path=exceptions_path,
                backoff_start=0.01,
                sleep=lambda _s: None,
                popen=popen,
                environ={},
                max_restarts=4,
            )
            code = supervisor.run()
            self.assertEqual(code, -9)
            self.assertEqual(len(started), 1)
            with open(health_path, encoding="utf-8") as handle:
                health = json.load(handle)
            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["exit_code"], -9)
            self.assertIn("signal 9", health["halt_reason"])

    def test_real_child_kill_flushes_stopped_then_respawns(self):
        with tempfile.TemporaryDirectory() as directory:
            health_path = os.path.join(directory, "live-health.json")
            exceptions_path = os.path.join(directory, "monitor-exceptions.jsonl")
            stop_file = os.path.join(directory, "paper-supervisor.stop")
            counter_path = os.path.join(directory, "starts.txt")
            child = (
                "import os, sys, time\n"
                f"path = {counter_path!r}\n"
                "n = int(open(path, encoding='utf-8').read()) if os.path.isfile(path) else 0\n"
                "open(path, 'w', encoding='utf-8').write(str(n + 1))\n"
                "time.sleep(30)\n"
            )

            def popen(command, **kwargs):
                return subprocess.Popen(command, **kwargs)

            sleeps = {"n": 0}

            def sleep(seconds):
                sleeps["n"] += 1
                if os.path.isfile(counter_path):
                    with open(counter_path, encoding="utf-8") as handle:
                        started = int(handle.read() or "0")
                    if started >= 2:
                        with open(stop_file, "w", encoding="utf-8") as handle:
                            handle.write("stop\n")
                time.sleep(min(0.05, float(seconds)))

            supervisor = PaperSupervisor(
                [sys.executable, "-c", child],
                health_path=health_path,
                stop_file=stop_file,
                exceptions_path=exceptions_path,
                backoff_start=0.05,
                backoff_max=0.05,
                poll_seconds=0.05,
                kill_grace_seconds=0.2,
                sleep=sleep,
                popen=popen,
                environ={},
                max_restarts=3,
            )

            def kill_first_child():
                deadline = time.time() + 5
                while supervisor.child is None and time.time() < deadline:
                    time.sleep(0.01)
                child_proc = supervisor.child
                self.assertIsNotNone(child_proc)
                child_proc.kill()

            import threading
            killer = threading.Thread(target=kill_first_child)
            killer.start()
            code = supervisor.run()
            killer.join(timeout=5)
            with open(counter_path, encoding="utf-8") as handle:
                self.assertGreaterEqual(int(handle.read()), 2)
            with open(health_path, encoding="utf-8") as handle:
                health = json.load(handle)
            self.assertEqual(health["status"], "stopped")
            self.assertIsNotNone(health.get("exit_code"))
            self.assertTrue(health.get("halt_reason"))
            self.assertTrue(health.get("last_error"))
            self.assertTrue(os.path.isfile(exceptions_path))
            self.assertIn(code, (-9, 137, 1, -15, 143, 0))

    def test_health_loop_flush_writes_stopped_on_shutdown(self):
        import asyncio

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "live-health.json")
            publisher = HealthPublisher(path, pid=os.getpid())
            publisher.flush(running=True)

            async def run():
                stop = asyncio.Event()
                stop.set()
                await run_health_loop(path, None, None, None, stop, publisher=publisher)

            asyncio.run(run())
            with open(path, encoding="utf-8") as handle:
                health = json.load(handle)
            self.assertEqual(health["status"], "stopped")
            self.assertEqual(health["halt_reason"], "shutdown")
            self.assertEqual(health["last_error"], "health loop stopped")

    def test_policy_floors_untouched(self):
        scanner = BinaryArbitrageScanner()
        self.assertEqual(scanner.min_net_profit_usd, 0.05)
        self.assertEqual(scanner.min_return, 0.002)
        self.assertEqual(scanner.safety_buffer_usd, 0.02)
        self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))
        self.assertNotEqual(os.environ.get("POLYMARKET_LIVE_CONFIRM"), "I_UNDERSTAND_THE_RISK")


if __name__ == "__main__":
    unittest.main()
