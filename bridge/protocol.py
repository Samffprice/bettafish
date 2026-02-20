"""Colonist.io WebSocket message parser.

Classifies and parses incoming JSON messages into typed dataclass objects.
The colonist.io protocol uses numbered message types with payloads:
  {"type": <int>, "payload": {...}, "sequence": N}

Key message types:
  4  - Game initialization (board, player color, play order)
  91 - State diff (incremental updates to game state)
  28 - Resource distribution
  43 - Trade execution
  13 - Discard prompt
  30/31/32/33 - Available build positions
  59 - Available dev cards to play
  80 - End turn
  45 - Game over

Also handles heartbeat messages: {"timestamp": N}
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed message dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GameInitMsg:
    """Type 4: Full game initialization with board, player color, and play order."""
    player_color: int        # Our player color index (1-based)
    play_order: List[int]    # Player colors in seating/turn order
    game_state: Dict         # Raw gameState with mapState, bankState, etc.


@dataclass
class GameStateDiffMsg:
    """Type 91: Incremental state diff applied to game state."""
    diff: Dict               # Dict with sub-sections: mapState, currentState, etc.
    time_left: float = 0.0   # Time remaining in current state


@dataclass
class ResourceDistributionMsg:
    """Type 28: Resources distributed to players."""
    distributions: List[Dict]  # [{owner, tileIndex, distributionType, card}, ...]


@dataclass
class TradeExecutionMsg:
    """Type 43: Trade executed between players (or bank trade)."""
    giving_player: int
    receiving_player: int
    giving_cards: List[int]
    receiving_cards: List[int]


@dataclass
class DiscardPromptMsg:
    """Type 13: Prompt for us to discard cards (rolled 7, >7 cards)."""
    amount: int              # Number of cards to discard
    valid_cards: List[int]   # Resource values of cards we can discard


@dataclass
class AvailableActionsMsg:
    """Types 30/31/32/33/59: Available building positions or playable dev cards."""
    action_type: int         # 30=settlement, 31=road, 32=city, 33=devcard, 59=play devcard
    indices: List[int]       # List of valid position indices


@dataclass
class RobVictimPromptMsg:
    """Type 29: Prompt to select which player to rob."""
    players_to_select: List[int]  # colonist.io color indices of players to rob


@dataclass
class EndTurnMsg:
    """Type 80: End turn signal."""
    pass


@dataclass
class GameOverMsg:
    """Type 45: Game over with end-game statistics."""
    end_game_state: Dict


@dataclass
class HeartbeatMsg:
    """Timestamp heartbeat (keepalive)."""
    timestamp: int


@dataclass
class UnknownMsg:
    """Fallback for unrecognized or non-game messages."""
    raw_data: Any = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_message(raw: str) -> Any:
    """Parse a raw WebSocket message string into a typed message object.

    Args:
        raw: Raw JSON string received from the WebSocket.

    Returns:
        A typed message dataclass, or UnknownMsg if unrecognized.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"SECURITY: Malformed JSON received: {e}")
        return UnknownMsg(raw_data={"error": "malformed_json", "preview": str(raw)[:200]})

    if not isinstance(data, dict):
        logger.debug(f"Non-dict JSON: type={type(data).__name__}")
        return UnknownMsg(raw_data=data)

    # Heartbeat: {"timestamp": N}
    if "timestamp" in data and "type" not in data:
        return HeartbeatMsg(timestamp=int(data["timestamp"]))

    msg_type = data.get("type")

    # String-typed connection messages (e.g. "Connected", "SessionEstablished")
    if isinstance(msg_type, str):
        return UnknownMsg(raw_data=data)

    if not isinstance(msg_type, int):
        return UnknownMsg(raw_data=data)

    payload = data.get("payload")

    try:
        if msg_type == 4:
            return _parse_game_init(payload)
        if msg_type == 91:
            return _parse_game_state_diff(payload)
        if msg_type == 28:
            return _parse_resource_distribution(payload)
        if msg_type == 43:
            return _parse_trade_execution(payload)
        if msg_type == 13:
            return _parse_discard_prompt(payload)
        if msg_type == 29:
            return _parse_rob_victim_prompt(payload)
        if msg_type in (30, 31, 32, 33, 59):
            return _parse_available_actions(msg_type, payload)
        if msg_type == 80:
            return EndTurnMsg()
        if msg_type == 45:
            return _parse_game_over(payload)
    except (KeyError, TypeError, ValueError, IndexError) as e:
        logger.warning(f"SECURITY: Error parsing type {msg_type} message: {e}")
        return UnknownMsg(raw_data=data)

    # Unhandled type numbers (1, 2, 3, 6, 27, 62, 78, etc.)
    return UnknownMsg(raw_data=data)


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------

def _parse_game_init(payload: Any) -> Any:
    """Parse type 4: game initialization."""
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    player_color = payload.get("playerColor")
    play_order = payload.get("playOrder")
    game_state = payload.get("gameState")

    if player_color is None or play_order is None or game_state is None:
        # Type 4 is also used for non-game-init messages (lobby info, etc.)
        return UnknownMsg(raw_data=payload)

    return GameInitMsg(
        player_color=int(player_color),
        play_order=[int(c) for c in play_order],
        game_state=game_state,
    )


def _parse_game_state_diff(payload: Any) -> Any:
    """Parse type 91: incremental state diff."""
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    diff = payload.get("diff")
    if not isinstance(diff, dict):
        return UnknownMsg(raw_data=payload)

    time_left = payload.get("timeLeftInState", 0.0)
    return GameStateDiffMsg(diff=diff, time_left=float(time_left))


def _parse_resource_distribution(payload: Any) -> Any:
    """Parse type 28: resource distribution.

    Payload is a list of dicts: [{owner, tileIndex, distributionType, card}, ...]
    Can be an empty list when no resources are distributed (e.g. robber tile).
    """
    if not isinstance(payload, list):
        return UnknownMsg(raw_data=payload)

    return ResourceDistributionMsg(distributions=payload)


def _parse_trade_execution(payload: Any) -> Any:
    """Parse type 43: trade execution.

    Payload: {givingPlayer, givingCards, receivingPlayer, receivingCards}
    receivingPlayer=0 means bank trade (discarded to bank).
    """
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    return TradeExecutionMsg(
        giving_player=int(payload.get("givingPlayer", -1)),
        receiving_player=int(payload.get("receivingPlayer", -1)),
        giving_cards=list(payload.get("givingCards", [])),
        receiving_cards=list(payload.get("receivingCards", [])),
    )


def _parse_discard_prompt(payload: Any) -> Any:
    """Parse type 13: discard prompt.

    Payload: {selectCardFormat: {amountOfCardsToSelect, validCardsToSelect, ...}, ...}
    """
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    scf = payload.get("selectCardFormat")
    if not isinstance(scf, dict):
        return UnknownMsg(raw_data=payload)

    amount = scf.get("amountOfCardsToSelect", 0)
    valid_cards = scf.get("validCardsToSelect", [])

    return DiscardPromptMsg(
        amount=int(amount),
        valid_cards=list(valid_cards),
    )


def _parse_rob_victim_prompt(payload: Any) -> Any:
    """Parse type 29: rob victim selection prompt.

    Payload: {playersToSelect: [4, 2], allowableActionState: 27, ...}
    """
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    players = payload.get("playersToSelect", [])
    if not isinstance(players, list):
        return UnknownMsg(raw_data=payload)

    return RobVictimPromptMsg(players_to_select=[int(p) for p in players])


def _parse_available_actions(msg_type: int, payload: Any) -> Any:
    """Parse types 30/31/32/33/59: available action positions."""
    if not isinstance(payload, list):
        # Some of these can have non-list payloads (e.g. type 33 with [])
        indices = []
    else:
        indices = payload

    return AvailableActionsMsg(action_type=msg_type, indices=indices)


def _parse_game_over(payload: Any) -> Any:
    """Parse type 45: game over."""
    if not isinstance(payload, dict):
        return UnknownMsg(raw_data=payload)

    end_game_state = payload.get("endGameState", {})
    return GameOverMsg(end_game_state=end_game_state)
