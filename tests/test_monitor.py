#!/usr/bin/env python3
"""QUANT-20260908-01: health-from-ledger, dead pid, rotation, exception tape."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from polywang.arbitrage_bot import PaperMarketRunner, write_health
from polywang.arbitrage_core import BinaryArbitrageScanner, BinaryMarket, JsonLedger, OrderBook
from polywang.market_replay import JsonlEventRecorder
from polywang.monitor import (
    HealthPublisher,
    bind_health_publisher,
    book_health_report,
    classify_feed_fault,
    compute_health_payload,
    count_exception_kinds,
    mark_health_stopped,
    next_backoff_seconds,
    process_is_alive,
    record_monitor_exception,
    record_process_exit,
    resolve_runtime_status,
    should_respawn,
    stop_flag_is_set,
    tail_jsonl_bytes,
)
from polywang.negrisk import LiveNegRiskJournal, NegRiskBookScanner, NegRiskMarket, PaperNegRiskExecutor


def _nway_payload(**overrides):
    row = {
        "id": "nr1",
        "conditionId": "cnr",
        "question": "Who wins",
        "clobTokenIds": '["tok-a", "tok-b", "tok-c"]',
        "outcomes": '["A", "B", "C"]',
        "outcomePrices": '["0.20", "0.20", "0.20"]',
        "category": "geopolitics",
        "active": True,
        "closed": False,
    }
    row.update(overrides)
    return row


def _synced_book(asks, timestamp_ms=1):
    book = OrderBook()
    book.asks = dict(asks)
    book.synced = True
    book.timestamp_ms = timestamp_ms
    book.hash = "h"
    return book


def _nr_market_and_opportunity(suffix="1"):
    tokens = (f"t{suffix}a", f"t{suffix}b", f"t{suffix}c")
    market = NegRiskMarket.from_gamma(_nway_payload(
        id=f"nr{suffix}",
        conditionId=f"c-nr{suffix}",
        clobTokenIds=json.dumps(list(tokens)),
        question=f"Who wins {suffix}",
    ))
    books = {token: _synced_book({0.20: 10}) for token in tokens}
    opportunity = NegRiskBookScanner(
        min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
    ).scan(market, books)
    return market, opportunity


class HealthFromLedgerTests(unittest.TestCase):
    def test_nr_settle_clears_open_negrisk_even_if_journal_is_ghost(self):
        _market, opportunity = _nr_market_and_opportunity("ghost")
        with tempfile.TemporaryDirectory() as directory:
            ledger = JsonLedger(os.path.join(directory, "paper-ledger.json"), initial_cash=1000.0)
            journal = LiveNegRiskJournal(os.path.join(directory, "paper-negrisk.json"))
            executor = PaperNegRiskExecutor(journal, ledger)
            executor.execute(opportunity)
            self.assertEqual(len(journal.incomplete_baskets()), 1)
            self.assertEqual(compute_health_payload(ledger=ledger, negrisk=journal)["open_negrisk"], 1)

            for position in ledger.state["positions"].values():
                if position.get("kind") == "negrisk":
                    ledger.settle(position["position_id"], "A")
            # Journal left at ASSEMBLED on purpose — the ghost QUANT-20260908-01 saw.
            self.assertEqual(len(journal.incomplete_baskets()), 1)
            snapshot = compute_health_payload(ledger=ledger, negrisk=journal)
            self.assertEqual(snapshot["open_negrisk"], 0)
            self.assertEqual(snapshot["negrisk_exposure"], 0.0)
            self.assertTrue(all(
                pos.get("settled") for pos in ledger.state["positions"].values()
                if pos.get("kind") == "negrisk"
            ))

    def test_publisher_flush_after_nr_settle_writes_open_negrisk_zero(self):
        market, opportunity = _nr_market_and_opportunity("flush")
        binary = BinaryMarket("m1", "c1", "Binary", "yes-token", "no-token", category="geopolitics")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WHALE_STATE_PATH": ""}, clear=False,
        ):
            os.environ.pop("ENABLE_NEGRISK_LIVE", None)
            journal = LiveNegRiskJournal(os.path.join(directory, "paper-negrisk.json"))
            health_path = os.path.join(directory, "live-health.json")
            runner = PaperMarketRunner(
                [binary], os.path.join(directory, "ledger.json"), 1000.0,
                BinaryArbitrageScanner(min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0),
                negrisk_markets=[market],
                negrisk_scanner=NegRiskBookScanner(
                    min_net_profit_usd=0.01, min_return=0.0, safety_buffer_usd=0.0,
                ),
                negrisk_executor=PaperNegRiskExecutor(journal),
            )
            runner.negrisk_executor.ledger = runner.ledger
            publisher = HealthPublisher(health_path, ledger=runner.ledger, negrisk=journal)
            bind_health_publisher(publisher, runner.ledger, journal)
            runner.health_publisher = publisher
            runner.negrisk_executor.execute(opportunity)
            publisher.flush(running=True)
            with open(health_path, encoding="utf-8") as handle:
                before = json.load(handle)
            self.assertEqual(before["open_negrisk"], 1)
            self.assertEqual(before["status"], "running")
            self.assertEqual(before["pid"], os.getpid())

            asyncio.run(runner.process({
                "event_type": "market_resolved", "market": "nrflush", "winning_outcome": "A",
            }))
            with open(health_path, encoding="utf-8") as handle:
                after = json.load(handle)
            self.assertEqual(after["open_negrisk"], 0)
            self.assertEqual(after["negrisk_exposure"], 0.0)
            self.assertEqual(runner.scanner.min_net_profit_usd, 0.01)
            self.assertEqual(runner.negrisk_scanner.min_net_profit_usd, 0.01)
            self.assertIsNone(os.environ.get("ENABLE_NEGRISK_LIVE"))

    def test_dead_process_reports_status_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "live-health.json")
            write_health(path, {
                "status": "running",
                "open_negrisk": 49,
                "negrisk_exposure": 1197.0,
                "pid": 999999999,
                "heartbeat_at": time.time() - 1.0,
            })
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertFalse(process_is_alive(999999999))
            self.assertEqual(resolve_runtime_status(stored), "stopped")
            snapshot = compute_health_payload(stored=stored, running=None, pid=999999999)
            self.assertEqual(snapshot["status"], "stopped")
            self.assertNotEqual(snapshot["status"], "running")

    def test_live_pid_stays_running(self):
        stored = {"status": "running", "pid": os.getpid(), "heartbeat_at": time.time()}
        self.assertTrue(process_is_alive(os.getpid()))
        self.assertEqual(resolve_runtime_status(stored), "running")
        snapshot = compute_health_payload(stored=stored, running=True, pid=os.getpid())
        self.assertEqual(snapshot["status"], "running")

    def test_mark_health_stopped_keeps_exposure_and_writes_exit_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "live-health.json")
            write_health(path, {
                "status": "running",
                "open_negrisk": 3,
                "negrisk_exposure": 12.5,
                "pid": 4242,
                "heartbeat_at": 1_700_000_000.0,
            })
            payload = mark_health_stopped(
                path,
                exit_code=137,
                last_error="signal 9",
                halt_reason="child killed by signal 9",
                now=1_700_000_030.0,
            )
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["exit_code"], 137)
            self.assertEqual(payload["last_error"], "signal 9")
            self.assertEqual(payload["halt_reason"], "child killed by signal 9")
            self.assertEqual(payload["open_negrisk"], 3)
            self.assertEqual(payload["negrisk_exposure"], 12.5)
            self.assertEqual(payload["pid"], 4242)
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertEqual(stored["status"], "stopped")
            self.assertEqual(stored["exit_code"], 137)

    def test_publisher_flush_stopped_includes_exit_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "live-health.json")
            publisher = HealthPublisher(path, pid=os.getpid())
            publisher.flush(
                running=False,
                exit_code=1,
                last_error="RuntimeError: boom",
                halt_reason="uncaught exception",
            )
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertEqual(stored["status"], "stopped")
            self.assertEqual(stored["exit_code"], 1)
            self.assertEqual(stored["last_error"], "RuntimeError: boom")
            self.assertEqual(stored["halt_reason"], "uncaught exception")
            self.assertNotEqual(stored["halt_reason"], "")

    def test_stop_flag_blocks_respawn(self):
        with tempfile.TemporaryDirectory() as directory:
            stop_file = os.path.join(directory, "paper-supervisor.stop")
            self.assertTrue(should_respawn(stop_file=stop_file, environ={}))
            self.assertFalse(stop_flag_is_set(stop_file, {}))
            with open(stop_file, "w", encoding="utf-8") as handle:
                handle.write("stop\n")
            self.assertFalse(should_respawn(stop_file=stop_file, environ={}))
            self.assertTrue(stop_flag_is_set(stop_file, {}))
            self.assertFalse(should_respawn(environ={"PAPER_SUPERVISOR_STOP": "1"}))
            self.assertTrue(next_backoff_seconds(0.0, start=1.0, maximum=30.0) <= 60.0)
            self.assertEqual(next_backoff_seconds(0.0, start=1.0, maximum=30.0), 1.0)
            self.assertEqual(next_backoff_seconds(1.0, start=1.0, maximum=30.0), 2.0)
            self.assertEqual(next_backoff_seconds(16.0, start=1.0, maximum=30.0), 30.0)


class EventRotationAndExceptionTests(unittest.TestCase):
    def test_rotation_bounds_active_file_by_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "market-events.jsonl")
            recorder = JsonlEventRecorder(path, max_bytes=180, max_age_seconds=86_400)
            for index in range(8):
                recorder.record({
                    "event_type": "book",
                    "asset_id": f"tok-{index}",
                    "asks": [{"price": "0.40", "size": "10"}],
                    "bids": [],
                })
            self.assertTrue(recorder.rotated_paths)
            if os.path.isfile(path):
                self.assertLessEqual(os.path.getsize(path), 180)
                with open(path, encoding="utf-8") as handle:
                    live_rows = [json.loads(line) for line in handle if line.strip()]
                self.assertTrue(all(row.get("event_type") == "book" for row in live_rows))
            rotated_size = os.path.getsize(recorder.rotated_paths[0])
            self.assertGreater(rotated_size, 0)

    def test_rotation_bounds_active_file_by_age(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "market-events.jsonl")
            clock = {"now": 1_700_000_000.0}

            def now():
                return clock["now"]

            recorder = JsonlEventRecorder(
                path, max_bytes=50_000_000, max_age_seconds=24 * 3600, clock=now,
            )
            recorder.record({"event_type": "book", "asset_id": "tok-0", "asks": [], "bids": []})
            self.assertFalse(recorder.rotated_paths)
            clock["now"] += 24 * 3600 + 1
            recorder.record({"event_type": "book", "asset_id": "tok-1", "asks": [], "bids": []})
            self.assertEqual(len(recorder.rotated_paths), 1)
            with open(path, encoding="utf-8") as handle:
                live_rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(live_rows), 1)
            self.assertEqual(live_rows[0]["asset_id"], "tok-1")

    def test_exception_log_is_independent_of_book_tape(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path = os.path.join(directory, "market-events.jsonl")
            exceptions_path = os.path.join(directory, "monitor-exceptions.jsonl")
            line = json.dumps({
                "event_type": "book", "asset_id": "tok", "asks": [{"price": "0.41", "size": "99"}],
                "source": "market", "received_at_ms": 1_700_000_000_000,
            }, separators=(",", ":")) + "\n"
            # ~2MB of book depth — morning health must not need the whole tape.
            with open(events_path, "w", encoding="utf-8") as handle:
                for _ in range(8000):
                    handle.write(line)
            file_bytes = os.path.getsize(events_path)
            self.assertGreater(file_bytes, 64 * 1024)
            record_monitor_exception(
                "market stream shard 0 disconnected", path=exceptions_path, source="market",
            )
            record_monitor_exception(
                "CLOB REST rate limited; retry after 1.0s", path=exceptions_path,
                source="rest-book", kind="http_429",
            )
            record_monitor_exception(
                "HTTP 500", path=exceptions_path, source="rest-book", kind="http_500",
            )
            record_monitor_exception(
                "EIP-712 signature failed", path=exceptions_path, source="official-client",
            )
            record_monitor_exception(
                "market stream text PONG timeout", path=exceptions_path, source="market",
            )
            counts = count_exception_kinds(exceptions_path)
            self.assertEqual(counts["disconnect"], 1)
            self.assertEqual(counts["http_429"], 1)
            self.assertEqual(counts["http_500"], 1)
            self.assertEqual(counts["eip712"], 1)
            self.assertEqual(counts["pong_timeout"], 1)
            self.assertEqual(counts["process_exit"], 0)
            self.assertEqual(counts["total"], 5)
            record_process_exit(
                "child killed by signal 9",
                path=exceptions_path,
                extra={"exit_code": -9, "pid": 99},
            )
            after = count_exception_kinds(exceptions_path)
            self.assertEqual(after["process_exit"], 1)
            self.assertEqual(after["total"], 6)
            report = book_health_report(
                events_path=events_path,
                exceptions_path=exceptions_path,
                max_event_bytes=4096,
            )
            self.assertLessEqual(report["events_bytes_read"], 4096)
            self.assertLess(report["events_bytes_read"], file_bytes)
            self.assertTrue(report["events_truncated"])
            self.assertEqual(report["exceptions"]["http_429"], 1)
            self.assertEqual(report["exceptions"]["eip712"], 1)
            self.assertEqual(report["exceptions"]["pong_timeout"], 1)
            self.assertEqual(report["exceptions"]["process_exit"], 1)
            self.assertEqual(report["exceptions"]["total"], 6)
            tail = tail_jsonl_bytes(events_path, 4096)
            self.assertLessEqual(tail["bytes_read"], 4096)
            self.assertTrue(all(row.get("event_type") == "book" for row in tail["rows"]))

    def test_book_health_records_process_exit_when_pid_is_dead(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path = os.path.join(directory, "market-events.jsonl")
            exceptions_path = os.path.join(directory, "monitor-exceptions.jsonl")
            health_path = os.path.join(directory, "live-health.json")
            open(events_path, "w", encoding="utf-8").close()
            write_health(health_path, {
                "status": "running",
                "pid": 999999999,
                "open_negrisk": 4,
                "heartbeat_at": time.time() - 1.0,
            })
            report = book_health_report(
                events_path=events_path,
                exceptions_path=exceptions_path,
                health_path=health_path,
            )
            self.assertEqual(report["health"]["status"], "stopped")
            with open(health_path, encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertEqual(stored["status"], "stopped")
            self.assertTrue(stored.get("halt_reason"))
            self.assertTrue(stored.get("last_error"))
            counts = count_exception_kinds(exceptions_path)
            self.assertEqual(counts["process_exit"], 1)

    def test_classify_feed_faults(self):
        self.assertEqual(classify_feed_fault(TimeoutError("market stream text PONG timeout")), "pong_timeout")
        self.assertEqual(classify_feed_fault(RuntimeError("EIP-712 typed data rejected")), "eip712")
        self.assertEqual(classify_feed_fault(RuntimeError("HTTP 500")), "http_500")
        self.assertEqual(classify_feed_fault(RuntimeError("status 429")), "http_429")
        self.assertEqual(classify_feed_fault(RuntimeError("socket disconnected")), "disconnect")

    def test_policy_floors_untouched_in_this_module(self):
        scanner = BinaryArbitrageScanner()
        self.assertEqual(scanner.min_net_profit_usd, 0.05)
        self.assertEqual(scanner.min_return, 0.002)
        self.assertEqual(scanner.safety_buffer_usd, 0.02)


if __name__ == "__main__":
    unittest.main()
