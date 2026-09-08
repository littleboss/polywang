"""Operator health, event-tape bounds, and feed-exception logging.

Health counts come from the paper ledger first. Journals only add baskets
that are still open and not already represented on the ledger. A dead
trading pid makes status ``stopped`` so monitors do not keep a stale
``running`` snapshot.

The market-event tape is for book replay. Feed faults go to
``monitor-exceptions.jsonl`` so morning / book-health can count them
without scanning multi-GB depth events.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

BOOK_HEALTH_MAX_EVENT_BYTES = 64 * 1024 * 1024
DEFAULT_HEALTH_STALE_SECONDS = 30.0
DEFAULT_EXCEPTIONS_LOG = "monitor-exceptions.jsonl"

DISCONNECT_KINDS = frozenset({
    "disconnect",
    "http_429",
    "http_500",
    "eip712",
    "pong_timeout",
    "feed_fault",
})

_exception_log_path = ""


def configure_exception_log(path: str) -> None:
    """Set the process-wide exception tape. Empty disables writes."""
    global _exception_log_path
    _exception_log_path = str(path or "")


def exception_log_path() -> str:
    if _exception_log_path:
        return _exception_log_path
    return os.getenv("MONITOR_EXCEPTIONS_LOG", "")


def process_is_alive(pid) -> bool:
    """True only when ``pid`` still exists (signal 0 / ESRCH)."""
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def write_health(path: str, payload: dict, *, now: Optional[float] = None) -> dict:
    """Atomically write a health snapshot. Adds ``updated_at``."""
    if not path:
        return dict(payload)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    body = dict(payload)
    body["updated_at"] = float(now if now is not None else time.time())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return body


def load_health_file(path: str) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def resolve_runtime_status(
    stored: Optional[dict],
    *,
    risk=None,
    running: Optional[bool] = None,
    now: Optional[float] = None,
    stale_after: Optional[float] = None,
) -> str:
    """Map halt / pid / heartbeat onto running | halted | stopped."""
    if risk is not None and getattr(risk, "state", None) and risk.state.get("halted"):
        if running is False:
            return "stopped"
        if running is True:
            return "halted"
        payload = stored or {}
        if process_is_alive(payload.get("pid")):
            return "halted"
        return "stopped"
    if running is False:
        return "stopped"
    if running is True:
        return "running"
    payload = stored or {}
    pid = payload.get("pid")
    if pid not in (None, ""):
        return "running" if process_is_alive(pid) else "stopped"
    beat = payload.get("heartbeat_at") or payload.get("updated_at")
    try:
        beat_at = float(beat)
    except (TypeError, ValueError):
        return "stopped"
    limit = DEFAULT_HEALTH_STALE_SECONDS if stale_after is None else float(stale_after)
    clock = float(now if now is not None else time.time())
    if clock - beat_at <= max(0.0, limit):
        return str(payload.get("status") or "running")
    return "stopped"


def _positions(ledger) -> List[dict]:
    if ledger is None:
        return []
    state = getattr(ledger, "state", None) or {}
    rows = state.get("positions") or {}
    if not isinstance(rows, dict):
        return []
    return [row for row in rows.values() if isinstance(row, dict)]


def _is_negrisk_position(position: dict) -> bool:
    return str(position.get("kind") or "") == "negrisk"


def _position_exposure(position: dict) -> float:
    try:
        cost = float(position.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    try:
        fees = float(position.get("fees") or 0.0)
    except (TypeError, ValueError):
        fees = 0.0
    return cost + fees


def _journal_reserved(record: dict) -> float:
    try:
        reserved = float(record.get("capital_reserved") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return reserved if reserved > 0.0 else 0.0


def journal_only_open_baskets(ledger, negrisk) -> List[dict]:
    """Incomplete journal baskets that are not already on the ledger.

    Ledger rows (open or settled) claim matching journal baskets by
    ``basket_id``, then leftover paper rows without ``basket_id`` claim one
    still-open basket on the same market. Anything left is still-open
    journal inventory (pending assemble, live-only, etc.).
    """
    if negrisk is None:
        return []
    incomplete = list(negrisk.incomplete_baskets())
    claimed_ids = set()
    leftover_by_market: Dict[str, int] = {}
    for position in _positions(ledger):
        if not _is_negrisk_position(position):
            continue
        basket_id = str(position.get("basket_id") or "")
        if basket_id:
            claimed_ids.add(basket_id)
            continue
        market_id = str(position.get("market_id") or "")
        if market_id:
            leftover_by_market[market_id] = leftover_by_market.get(market_id, 0) + 1
    leftover = []
    for record in incomplete:
        basket_id = str(record.get("basket_id") or "")
        if basket_id and basket_id in claimed_ids:
            continue
        market_id = str(record.get("market_id") or "")
        if leftover_by_market.get(market_id, 0) > 0:
            leftover_by_market[market_id] -= 1
            continue
        leftover.append(record)
    return leftover


def compute_health_payload(
    *,
    ledger=None,
    pair_journal=None,
    directional=None,
    negrisk=None,
    risk=None,
    stored: Optional[dict] = None,
    running: Optional[bool] = None,
    pid: Optional[int] = None,
    now: Optional[float] = None,
    stale_after: Optional[float] = None,
) -> dict:
    """Recompute health from the ledger; journals only add still-open rows."""
    clock = float(now if now is not None else time.time())
    binary_open = []
    nr_open_ledger = []
    for position in _positions(ledger):
        if position.get("settled"):
            continue
        if _is_negrisk_position(position):
            nr_open_ledger.append(position)
        else:
            binary_open.append(position)

    journal_only = journal_only_open_baskets(ledger, negrisk)
    if ledger is not None:
        open_pairs = len(binary_open)
        pair_exposure = sum(_position_exposure(position) for position in binary_open)
        open_negrisk = len(nr_open_ledger) + len(journal_only)
        negrisk_exposure = (
            sum(_position_exposure(position) for position in nr_open_ledger)
            + sum(_journal_reserved(record) for record in journal_only)
        )
    else:
        open_pairs = len(pair_journal.incomplete_pairs()) if pair_journal else 0
        pair_exposure = pair_journal.open_exposure() if pair_journal else 0.0
        open_negrisk = len(negrisk.incomplete_baskets()) if negrisk else 0
        negrisk_exposure = negrisk.open_exposure() if negrisk else 0.0

    status = resolve_runtime_status(
        stored, risk=risk, running=running, now=clock, stale_after=stale_after,
    )
    halt_reason = ""
    if risk is not None and getattr(risk, "state", None):
        halt_reason = str(risk.state.get("halt_reason") or "")
    if status == "stopped":
        halt_reason = halt_reason or "process not alive"

    payload = {
        "status": status,
        "halt_reason": halt_reason,
        "pair_exposure": float(pair_exposure),
        "directional_exposure": directional.open_exposure() if directional else 0.0,
        "negrisk_exposure": float(negrisk_exposure),
        "open_pairs": int(open_pairs),
        "open_directional": len(directional.incomplete_trades()) if directional else 0,
        "open_negrisk": int(open_negrisk),
        "heartbeat_at": clock,
        "last_flush_at": clock,
    }
    if pid is not None:
        payload["pid"] = int(pid)
    elif stored and stored.get("pid") not in (None, ""):
        payload["pid"] = int(stored["pid"])
    return payload


class HealthPublisher:
    """Write live-health from current ledger / journal objects."""

    def __init__(
        self,
        path: str,
        *,
        ledger=None,
        pair_journal=None,
        directional=None,
        negrisk=None,
        risk=None,
        pid: Optional[int] = None,
        stale_after: Optional[float] = None,
    ):
        self.path = path
        self.ledger = ledger
        self.pair_journal = pair_journal
        self.directional = directional
        self.negrisk = negrisk
        self.risk = risk
        self.pid = int(pid if pid is not None else os.getpid())
        self.stale_after = stale_after
        self.last_payload: Optional[dict] = None

    def snapshot(self, *, running: Optional[bool] = True, now: Optional[float] = None) -> dict:
        stored = load_health_file(self.path)
        return compute_health_payload(
            ledger=self.ledger,
            pair_journal=self.pair_journal,
            directional=self.directional,
            negrisk=self.negrisk,
            risk=self.risk,
            stored=stored,
            running=running,
            pid=self.pid,
            now=now,
            stale_after=self.stale_after,
        )

    def flush(self, running: Optional[bool] = True, now: Optional[float] = None) -> dict:
        payload = self.snapshot(running=running, now=now)
        self.last_payload = write_health(self.path, payload, now=now)
        return self.last_payload


def bind_health_publisher(publisher: HealthPublisher, *flushables) -> HealthPublisher:
    """Point ledger / journal ``on_flush`` hooks at the publisher."""
    for item in flushables:
        if item is None:
            continue
        item.on_flush = publisher.flush
    return publisher


def classify_feed_fault(error, *, default: str = "disconnect") -> str:
    """Map stream / HTTP / signing faults onto the exception channel kinds."""
    if error is None:
        return default
    status = getattr(error, "status_code", None) or getattr(error, "retry_after", None)
    name = type(error).__name__
    text = " ".join(str(error).strip().split())
    blob = f"{name} {text}".lower()
    if "eip-712" in blob or "eip712" in blob:
        return "eip712"
    if "pong timeout" in blob or "text pong" in blob:
        return "pong_timeout"
    if "429" in blob or "rate limited" in blob or name == "RestRateLimitError":
        return "http_429"
    if " 500" in f" {blob}" or blob.endswith("500") or "internal server error" in blob:
        return "http_500"
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = None
    if code == 429:
        return "http_429"
    if code == 500:
        return "http_500"
    if "disconnect" in blob or "connection" in blob or default == "disconnect":
        if default in DISCONNECT_KINDS:
            return default
        return "disconnect"
    return default if default in DISCONNECT_KINDS else "feed_fault"


class JsonlExceptionLogger:
    """Append-only feed-fault log, separate from the book-depth tape."""

    def __init__(self, path: str = ""):
        self.path = path

    def record(self, kind: str, message: str, *, source: str = "", extra: Optional[dict] = None,
               now: Optional[float] = None) -> Optional[dict]:
        if not self.path:
            return None
        label = str(kind or "feed_fault")
        if label not in DISCONNECT_KINDS:
            label = "feed_fault"
        payload = {
            "kind": label,
            "message": str(message or ""),
            "source": str(source or ""),
            "ts": float(now if now is not None else time.time()),
        }
        if extra:
            for key, value in extra.items():
                if key in payload:
                    continue
                payload[key] = value
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
        return payload


def record_monitor_exception(
    error,
    *,
    source: str = "",
    kind: str = "",
    extra: Optional[dict] = None,
    path: str = "",
    now: Optional[float] = None,
) -> Optional[dict]:
    target = path or exception_log_path()
    if not target:
        return None
    label = kind or classify_feed_fault(error)
    return JsonlExceptionLogger(target).record(
        label, str(error), source=source, extra=extra, now=now,
    )


def count_exception_kinds(path: str) -> Dict[str, int]:
    """Read the exception tape only. Never opens the book-depth jsonl."""
    counts = {kind: 0 for kind in sorted(DISCONNECT_KINDS)}
    counts["total"] = 0
    if not path or not os.path.isfile(path):
        return counts
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "feed_fault")
            if kind not in DISCONNECT_KINDS:
                kind = "feed_fault"
            counts[kind] = counts.get(kind, 0) + 1
            counts["total"] += 1
    return counts


def tail_jsonl_bytes(path: str, max_bytes: int, *, read_fn: Optional[Callable] = None) -> dict:
    """Read at most ``max_bytes`` from the end of a jsonl file.

    Skips a leading partial line after a mid-file seek so callers never
    full-scan a multi-GB tape.
    """
    limit = max(0, int(max_bytes))
    result = {"bytes_read": 0, "rows": [], "file_bytes": 0, "truncated": False}
    if not path or not os.path.isfile(path) or limit <= 0:
        if path and os.path.isfile(path):
            result["file_bytes"] = os.path.getsize(path)
        return result
    size = os.path.getsize(path)
    result["file_bytes"] = size
    start = 0
    if size > limit:
        start = size - limit
        result["truncated"] = True
    opener = read_fn or (lambda p, mode="rb": open(p, mode))
    with opener(path, "rb") as handle:
        handle.seek(start)
        blob = handle.read(limit)
    result["bytes_read"] = len(blob)
    if start > 0:
        newline = blob.find(b"\n")
        if newline >= 0:
            blob = blob[newline + 1:]
    rows = []
    for raw in blob.splitlines():
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    result["rows"] = rows
    return result


def book_health_report(
    *,
    events_path: str,
    exceptions_path: str,
    health_path: str = "",
    ledger=None,
    pair_journal=None,
    directional=None,
    negrisk=None,
    risk=None,
    max_event_bytes: int = BOOK_HEALTH_MAX_EVENT_BYTES,
    now: Optional[float] = None,
) -> dict:
    """Morning / book-health: bounded event tail + exception file counts."""
    stored = load_health_file(health_path) if health_path else None
    health = compute_health_payload(
        ledger=ledger,
        pair_journal=pair_journal,
        directional=directional,
        negrisk=negrisk,
        risk=risk,
        stored=stored,
        running=None,
        pid=None if not stored else stored.get("pid"),
        now=now,
    )
    if stored and health.get("status") == "stopped" and health_path:
        persist = dict(health)
        persist["heartbeat_at"] = stored.get("heartbeat_at", health.get("heartbeat_at"))
        write_health(health_path, persist, now=now)
    tail = tail_jsonl_bytes(events_path, max_event_bytes)
    last_event_ms = None
    for row in reversed(tail["rows"]):
        stamp = row.get("received_at_ms") or row.get("timestamp")
        try:
            last_event_ms = int(stamp)
            break
        except (TypeError, ValueError):
            continue
    exceptions = count_exception_kinds(exceptions_path)
    return {
        "health": health,
        "events_path": events_path,
        "events_file_bytes": tail["file_bytes"],
        "events_bytes_read": tail["bytes_read"],
        "events_tailed": len(tail["rows"]),
        "events_truncated": tail["truncated"],
        "last_event_received_at_ms": last_event_ms,
        "exceptions_path": exceptions_path,
        "exceptions": exceptions,
        "max_event_bytes": int(max_event_bytes),
    }


def maybe_load_json_state(loader, path: str):
    """Load a ledger/journal only when the file already exists."""
    if not path or not os.path.isfile(path):
        return None
    return loader(path)
