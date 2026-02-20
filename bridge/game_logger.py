"""Structured JSON-lines game logging with atexit cleanup.

All file I/O wrapped in try/except to prevent logging failures from
crashing the bridge server.
"""
import atexit
import json
import logging
import os
import time
from typing import Any, Optional

from bridge import config

logger = logging.getLogger(__name__)


class GameLogger:
    """Structured JSON-lines game logger with safe resource management.

    Resource management:
    - File handle opened via start_game() with try/except
    - atexit.register(self.end_game) ensures cleanup on unexpected exit
    - All write operations wrapped in try/except to prevent logging failures
      from crashing the bridge server
    - end_game() is idempotent (safe to call multiple times)
    """

    def __init__(self):
        self._file = None
        self._game_id = None
        atexit.register(self.end_game)

    def start_game(self) -> None:
        """Open a new game log file.

        Reads config.LOG_DIR at call time so tests can override the module-level
        config.LOG_DIR attribute without needing to reload the module.
        """
        self._game_id = f"game_{int(time.time())}"
        log_dir = config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{self._game_id}.jsonl")
        try:
            self._file = open(filepath, "w")
            logger.info(f"Game log started: {filepath}")
        except OSError as e:
            logger.error(f"Failed to open log file {filepath}: {e}")
            self._file = None

    def log_incoming(self, msg_type: str, raw: str) -> None:
        """Log an incoming WebSocket message."""
        self._write_entry({
            "direction": "in",
            "type": msg_type,
            "ts": time.time(),
            "raw": raw[:50000],
        })

    def log_outgoing(self, msg_type: str, data: Any) -> None:
        """Log an outgoing WebSocket message."""
        self._write_entry({
            "direction": "out",
            "type": msg_type,
            "ts": time.time(),
            "data": self._safe_serialize(data),
        })

    def log_security_event(self, event_type: str, details: str) -> None:
        """Log a security-relevant event (unexpected messages, validation failures)."""
        self._write_entry({
            "direction": "security",
            "type": event_type,
            "ts": time.time(),
            "details": str(details)[:1000],
        })

    def log_decision(self, state_summary: dict, action: Any) -> None:
        """Log a bot decision point with state summary."""
        self._write_entry({
            "direction": "decision",
            "ts": time.time(),
            "state": self._safe_serialize(state_summary),
            "action": str(action),
        })

    def _write_entry(self, entry: dict) -> None:
        """Write a single log entry. Wrapped in try/except to never crash the bridge."""
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
        except (OSError, ValueError) as e:
            logger.error(f"Failed to write log entry: {e}")

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize an object to JSON-compatible form."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)[:500]

    def end_game(self) -> None:
        """Close the log file. Idempotent - safe to call multiple times."""
        if self._file is not None:
            try:
                self._file.close()
                logger.info(f"Game log closed: {self._game_id}")
            except OSError as e:
                logger.error(f"Failed to close log file: {e}")
            finally:
                self._file = None
