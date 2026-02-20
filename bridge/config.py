"""Configuration constants for the AICatan bridge server.

All network and timeout settings can be overridden via environment variables.
Uses _safe_int() to validate integer env vars with range checking.
"""
import os
import logging

_logger = logging.getLogger(__name__)


def _safe_int(env_var: str, default: int, min_val: int = 1, max_val: int = 65535) -> int:
    """Parse integer from environment variable with validation and fallback.

    Args:
        env_var: Name of the environment variable.
        default: Default value if env var is unset or invalid.
        min_val: Minimum acceptable value (inclusive).
        max_val: Maximum acceptable value (inclusive).

    Returns:
        Parsed integer, or default if parsing/validation fails.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        val = int(raw)
        if val < min_val or val > max_val:
            _logger.warning(
                f"{env_var}={val} out of range [{min_val}, {max_val}], "
                f"using default {default}"
            )
            return default
        return val
    except ValueError:
        _logger.warning(
            f"{env_var}={raw!r} is not a valid integer, using default {default}"
        )
        return default


# Network configuration with environment overrides
WS_HOST = os.environ.get("AICATAN_HOST", "localhost")
WS_PORT = _safe_int("AICATAN_PORT", 8765, 1024, 65535)
LOG_DIR = os.environ.get("AICATAN_LOG_DIR", "logs")

# Timeout constants with environment overrides
DECISION_TIMEOUT_SECONDS = _safe_int("AICATAN_DECISION_TIMEOUT", 5, 1, 300)
WATCHDOG_TIMEOUT_SECONDS = _safe_int("AICATAN_WATCHDOG_TIMEOUT", 30, 5, 600)
WS_SEND_TIMEOUT_SECONDS = _safe_int("AICATAN_WS_SEND_TIMEOUT", 10, 1, 120)
WS_RECV_TIMEOUT_SECONDS = _safe_int("AICATAN_WS_RECV_TIMEOUT", 30, 1, 300)

# colonist.io resource values -> Catanatron resource strings
COLONIST_RESOURCE_TO_CATAN = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}
CATAN_RESOURCE_TO_COLONIST = {
    "WOOD": 1,
    "BRICK": 2,
    "SHEEP": 3,
    "WHEAT": 4,
    "ORE": 5,
}

# colonist.io tile types -> Catanatron resource strings (0=desert/None)
COLONIST_TILE_TO_RESOURCE = {
    0: None,
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

# colonist.io dev card values (from DevCards enum in robottler)
COLONIST_DEVCARD_VALUES = {
    "KNIGHT": 11,
    "VICTORY_POINT": 12,
    "MONOPOLY": 13,
    "ROAD_BUILDING": 14,
    "YEAR_OF_PLENTY": 15,
}
COLONIST_VALUE_TO_DEVCARD = {
    11: "KNIGHT",
    12: "VICTORY_POINT",
    13: "MONOPOLY",
    14: "ROAD_BUILDING",
    15: "YEAR_OF_PLENTY",
}

# colonist.io harbor types -> Catanatron resource
# 0 = no harbor, 1 = 3:1 (None resource), 2-6 = 2:1 specific resource
COLONIST_HARBOR_TO_RESOURCE = {
    0: "NONE",   # no harbor
    1: None,     # 3:1 generic
    2: "WOOD",
    3: "BRICK",
    4: "SHEEP",
    5: "WHEAT",
    6: "ORE",
}

# Bot action codes sent to userscript
ACTION_BUILD_ROAD = 0
ACTION_BUILD_SETTLEMENT = 1
ACTION_BUILD_CITY = 2
ACTION_BUY_DEV_CARD = 3
ACTION_THROW_DICE = 4
ACTION_PASS_TURN = 5
ACTION_ACCEPT_TRADE = 6
ACTION_REJECT_TRADE = 7
ACTION_MOVE_ROBBER = 8
ACTION_ROB_PLAYER = 9
ACTION_SELECT_CARDS = 10
ACTION_CREATE_TRADE = 11
ACTION_PLAY_DEV_CARD = 12
ACTION_BUILD_ROAD_DEV = 13  # Road placement during Road Building dev card

# colonist.io incoming message type numbers
MSG_TYPE_GAME_INIT = 4
MSG_TYPE_GAME_STATE_DIFF = 91
MSG_TYPE_RESOURCE_DISTRIBUTION = 28
MSG_TYPE_TRADE_EXECUTION = 43
MSG_TYPE_DISCARD_PROMPT = 13
MSG_TYPE_AVAILABLE_SETTLEMENTS = 30
MSG_TYPE_AVAILABLE_ROADS = 31
MSG_TYPE_AVAILABLE_CITIES = 32
MSG_TYPE_AVAILABLE_DEV_CARD = 33
MSG_TYPE_AVAILABLE_DEV_CARDS_TO_PLAY = 59
MSG_TYPE_END_TURN = 80
MSG_TYPE_GAME_OVER = 45

# colonist.io action states (from currentState.actionState in diffs)
COLONIST_ACTION_STATE_MAIN = 0
COLONIST_ACTION_STATE_SETUP_SETTLEMENT = 1
COLONIST_ACTION_STATE_SETUP_ROAD = 3
COLONIST_ACTION_STATE_BUILDING = 4
COLONIST_ACTION_STATE_MOVE_ROBBER = 24
COLONIST_ACTION_STATE_ROB_VICTIM = 27
COLONIST_ACTION_STATE_DISCARD = 28

# colonist.io turn states (from currentState.turnState in diffs)
COLONIST_TURN_STATE_ROLL = 1
COLONIST_TURN_STATE_PLAY = 2
COLONIST_TURN_STATE_GAME_OVER = 3

# colonist.io trade response values
TRADE_RESPONSE_PENDING = 0
TRADE_RESPONSE_ACCEPT = 1
TRADE_RESPONSE_REJECT = 2

# Anti-cheat delay settings (env vars use milliseconds, converted to seconds at use time)
ANTICHEAT_ENABLED = _safe_int("AICATAN_ANTICHEAT", 1, 0, 1) == 1
ANTICHEAT_MIN_THINK_MS = _safe_int("AICATAN_ANTICHEAT_MIN_THINK_MS", 1500, 0, 30000)
ANTICHEAT_MAX_THINK_MS = _safe_int("AICATAN_ANTICHEAT_MAX_THINK_MS", 4000, 0, 30000)
ANTICHEAT_ACTION_INTERVAL_MS = _safe_int("AICATAN_ANTICHEAT_ACTION_INTERVAL_MS", 200, 0, 5000)

