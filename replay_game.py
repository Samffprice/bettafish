"""Game replay visualizer — replays a colonist.io game log with ASCII board + event log.

Reads a JSONL log file (e.g. logs/full_game_2.jsonl), feeds each incoming message
through the protocol parser, and prints a turn-by-turn narration with an ASCII hex
board rendered at key moments (builds, robber moves, game init).

Usage:
    python3 replay_game.py logs/full_game_2.jsonl
    python3 replay_game.py logs/full_game_2.jsonl | less -R
    python3 replay_game.py logs/full_game_2.jsonl --no-color
"""
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bridge.protocol import (
    AvailableActionsMsg,
    DiscardPromptMsg,
    EndTurnMsg,
    GameInitMsg,
    GameOverMsg,
    GameStateDiffMsg,
    HeartbeatMsg,
    ResourceDistributionMsg,
    RobVictimPromptMsg,
    TradeExecutionMsg,
    UnknownMsg,
    parse_message,
)
from bridge.config import COLONIST_TILE_TO_RESOURCE, COLONIST_RESOURCE_TO_CATAN


def try_parse_truncated_type4(raw: str) -> Optional[GameInitMsg]:
    """Try to parse a truncated type 4 message from the game log.

    The game logger truncates raw messages at 2000 chars, which cuts off
    the type 4 init message. We extract what we can: playerColor, playOrder,
    and tileHexStates (tiles are early in the JSON and usually complete).
    """
    if '"type":4' not in raw and '"type": 4' not in raw:
        return None

    # Extract playerColor
    m = re.search(r'"playerColor"\s*:\s*(\d+)', raw)
    if not m:
        return None
    player_color = int(m.group(1))

    # Extract playOrder
    m = re.search(r'"playOrder"\s*:\s*\[([^\]]+)\]', raw)
    if not m:
        return None
    play_order = [int(x.strip()) for x in m.group(1).split(",")]

    # Extract tileHexStates - find the complete JSON object
    # It starts after "tileHexStates":{ and ends at the matching }
    hex_start = raw.find('"tileHexStates":{')
    if hex_start < 0:
        hex_start = raw.find('"tileHexStates": {')
    if hex_start < 0:
        return None

    # Find the opening brace after tileHexStates
    brace_pos = raw.index("{", hex_start + len('"tileHexStates"'))
    depth = 0
    end_pos = brace_pos
    for i in range(brace_pos, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break

    tile_hex_json = raw[brace_pos:end_pos]
    try:
        tile_hex_states = json.loads(tile_hex_json)
    except json.JSONDecodeError:
        return None

    # Try to extract tileCornerStates (may be truncated)
    corner_states = {}
    cs_start = raw.find('"tileCornerStates":{')
    if cs_start < 0:
        cs_start = raw.find('"tileCornerStates": {')
    if cs_start >= 0:
        brace_pos = raw.index("{", cs_start + len('"tileCornerStates"'))
        # Try to parse as much as we can
        depth = 0
        end_pos = len(raw)
        for i in range(brace_pos, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break

        cs_json = raw[brace_pos:end_pos]
        try:
            corner_states = json.loads(cs_json)
        except json.JSONDecodeError:
            # Truncated - parse individual entries
            for m in re.finditer(
                r'"(\d+)"\s*:\s*\{"x"\s*:\s*(-?\d+)\s*,\s*"y"\s*:\s*(-?\d+)\s*,\s*"z"\s*:\s*(\d+)\s*\}',
                cs_json,
            ):
                corner_states[m.group(1)] = {
                    "x": int(m.group(2)),
                    "y": int(m.group(3)),
                    "z": int(m.group(4)),
                }

    game_state = {
        "mapState": {
            "tileHexStates": tile_hex_states,
            "tileCornerStates": corner_states,
            "tileEdgeStates": {},
        },
    }

    return GameInitMsg(
        player_color=player_color,
        play_order=play_order,
        game_state=game_state,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCE_ABBREV = {
    None: "De",
    "WOOD": "Wo",
    "BRICK": "Br",
    "SHEEP": "Sh",
    "WHEAT": "Wh",
    "ORE": "Or",
}

RESOURCE_SHORT = {
    None: "Desert",
    "WOOD": "Wood",
    "BRICK": "Brick",
    "SHEEP": "Sheep",
    "WHEAT": "Wheat",
    "ORE": "Ore",
}

# ANSI color codes for terminal
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
PLAYER_ANSI = {
    1: "\033[91m",   # bright red
    2: "\033[94m",   # bright blue
    3: "\033[93m",   # bright yellow/orange
    4: "\033[97m",   # bright white
}
ANSI_GREEN = "\033[92m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_GRAY = "\033[90m"  # dark gray for axes

use_color = True


def c(color_code: str, text: str) -> str:
    """Wrap text in ANSI color if colors are enabled."""
    if use_color:
        return f"{color_code}{text}{ANSI_RESET}"
    return text


def player_str(color_idx: int) -> str:
    """Format a player identifier with color."""
    ansi = PLAYER_ANSI.get(color_idx, "")
    return c(ansi, f"P{color_idx}")


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------

@dataclass
class TileInfo:
    index: int
    x: int
    y: int
    resource: Optional[str]  # "WOOD", "BRICK", etc. or None for desert
    dice_num: int


@dataclass
class VertexInfo:
    index: int
    x: int
    y: int
    z: int
    owner: int = -1
    building_type: int = 0  # 0=empty, 1=settlement, 2=city


@dataclass
class EdgeInfo:
    index: int
    x: int
    y: int
    z: int
    owner: int = -1


class ReplayState:
    def __init__(self):
        self.tiles: Dict[int, TileInfo] = {}
        self.vertices: Dict[int, VertexInfo] = {}
        self.edges: Dict[int, EdgeInfo] = {}

        self.player_order: List[int] = []
        self.my_color: int = -1
        self.robber_tile: int = -1

        self.current_player: int = -1
        self.turn_state: int = -1      # 1=roll, 2=play, 3=game_over
        self.action_state: int = -1
        self.is_setup: bool = True

        self.dice_total: int = 0
        self.dice1: int = 0
        self.dice2: int = 0

        # Per-player resource tracking (from distributions + trades)
        self.player_resources: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
        )

        # Counters
        self.settlements_built: int = 0
        self.cities_built: int = 0
        self.roads_built: int = 0
        self.total_messages: int = 0
        self.unknown_messages: int = 0
        self.completed_turns: int = 0

    def init_from_game_init(self, msg: GameInitMsg) -> List[str]:
        """Initialize state from type 4 game init message."""
        events = []
        self.my_color = msg.player_color
        self.player_order = list(msg.play_order)

        game_state = msg.game_state
        map_state = game_state.get("mapState", {})

        # Parse tiles
        tile_hex = map_state.get("tileHexStates", {})
        for idx_str, data in tile_hex.items():
            idx = int(idx_str)
            tile_type = data.get("type", 0)
            resource = COLONIST_TILE_TO_RESOURCE.get(tile_type)
            dice_num = data.get("diceNumber", 0) or 0
            self.tiles[idx] = TileInfo(
                index=idx,
                x=data.get("x", 0),
                y=data.get("y", 0),
                resource=resource,
                dice_num=dice_num,
            )
            # Desert tile has the robber initially
            if resource is None:
                self.robber_tile = idx

        # Parse vertices
        corners = map_state.get("tileCornerStates", {})
        for idx_str, data in corners.items():
            idx = int(idx_str)
            self.vertices[idx] = VertexInfo(
                index=idx,
                x=data.get("x", 0),
                y=data.get("y", 0),
                z=data.get("z", 0),
            )

        # Parse edges
        edge_states = map_state.get("tileEdgeStates", {})
        for idx_str, data in edge_states.items():
            idx = int(idx_str)
            if isinstance(data, dict) and "x" in data:
                self.edges[idx] = EdgeInfo(
                    index=idx,
                    x=data.get("x", 0),
                    y=data.get("y", 0),
                    z=data.get("z", 0),
                )

        events.append(
            f"  You: {player_str(self.my_color)}"
        )
        events.append(
            f"  Play order: {[f'P{p}' for p in self.player_order]}"
        )
        events.append(
            f"  Board: {len(self.tiles)} tiles, {len(self.vertices)} vertices, "
            f"{len(self.edges)} edges"
        )

        return events

    def apply_diff(self, msg: GameStateDiffMsg) -> Tuple[List[str], bool]:
        """Apply a type 91 diff. Returns (events, board_changed)."""
        events = []
        board_changed = False
        diff = msg.diff

        # Dice rolls — diceState is a diff, so missing keys mean "unchanged".
        # Only update the values that are present.
        dice_state = diff.get("diceState")
        if dice_state and dice_state.get("diceThrown"):
            if "dice1" in dice_state:
                self.dice1 = dice_state["dice1"]
            if "dice2" in dice_state:
                self.dice2 = dice_state["dice2"]
            self.dice_total = self.dice1 + self.dice2
            events.append(
                f"  {c(ANSI_CYAN, '[DICE]')}  Rolled "
                f"{c(ANSI_BOLD, str(self.dice_total))} ({self.dice1}+{self.dice2})"
            )
        # Note: diceThrown=false does NOT reset dice values.
        # Partial diffs carry forward die values from the previous roll.

        # Building updates (settlements/cities)
        map_state = diff.get("mapState")
        if map_state:
            corners = map_state.get("tileCornerStates")
            if corners:
                for idx_str, data in corners.items():
                    if not isinstance(data, dict):
                        continue
                    idx = int(idx_str)
                    owner = data.get("owner", -1)
                    btype = data.get("buildingType", 0)

                    # City upgrades only send {"buildingType": 2} without owner.
                    # Look up the existing vertex owner when owner is missing.
                    if owner < 0 and btype > 0 and idx in self.vertices:
                        owner = self.vertices[idx].owner

                    if owner >= 0 and btype > 0:
                        # Update or create vertex
                        if idx in self.vertices:
                            self.vertices[idx].owner = owner
                            self.vertices[idx].building_type = btype
                        else:
                            # Vertex wasn't in init data (truncated)
                            x = data.get("x", 0)
                            y = data.get("y", 0)
                            z = data.get("z", 0)
                            self.vertices[idx] = VertexInfo(
                                index=idx, x=x, y=y, z=z,
                                owner=owner, building_type=btype,
                            )

                        bname = "city" if btype == 2 else "settlement"
                        tile_desc = self._nearby_tiles_desc(idx)
                        events.append(
                            f"  {c(ANSI_GREEN, '[BUILD]')} {player_str(owner)} "
                            f"{bname} at vertex {idx}{tile_desc}"
                        )
                        board_changed = True
                        if btype == 2:
                            self.cities_built += 1
                        else:
                            self.settlements_built += 1

            # Road updates
            edges = map_state.get("tileEdgeStates")
            if edges:
                for idx_str, data in edges.items():
                    if not isinstance(data, dict):
                        continue
                    idx = int(idx_str)
                    owner = data.get("owner", -1)
                    if owner >= 0:
                        if idx in self.edges:
                            self.edges[idx].owner = owner
                        else:
                            x = data.get("x", 0)
                            y = data.get("y", 0)
                            z = data.get("z", 0)
                            self.edges[idx] = EdgeInfo(
                                index=idx, x=x, y=y, z=z, owner=owner,
                            )

                        events.append(
                            f"  {c(ANSI_GREEN, '[BUILD]')} {player_str(owner)} "
                            f"road at edge {idx}"
                        )
                        board_changed = True
                        self.roads_built += 1

        # Robber movement
        robber_state = diff.get("mechanicRobberState")
        if robber_state:
            tile_idx = robber_state.get("locationTileIndex")
            if tile_idx is not None and tile_idx != self.robber_tile:
                old = self.robber_tile
                self.robber_tile = tile_idx
                tile = self.tiles.get(tile_idx)
                tile_desc = ""
                if tile:
                    res = RESOURCE_SHORT.get(tile.resource, "?")
                    tile_desc = f" ({res}:{tile.dice_num})"
                events.append(
                    f"  {c(ANSI_MAGENTA, '[ROBBER]')} Moved to tile {tile_idx}{tile_desc}"
                )
                board_changed = True

        # Turn/action state changes
        current_state = diff.get("currentState")
        if current_state:
            new_player = current_state.get("currentTurnPlayerColor")
            new_turn = current_state.get("turnState")
            new_action = current_state.get("actionState")
            completed = current_state.get("completedTurns")

            if new_turn is not None:
                self.turn_state = new_turn
                self.is_setup = False
            if new_action is not None:
                self.action_state = new_action
            if new_player is not None and new_player != self.current_player:
                self.current_player = new_player
            if completed is not None:
                self.completed_turns = completed

        # VP updates
        player_states = diff.get("playerStates")
        if player_states:
            for col_str, pdata in player_states.items():
                if not isinstance(pdata, dict):
                    continue
                vp_state = pdata.get("victoryPointsState")
                if vp_state and isinstance(vp_state, dict):
                    # VP state has index→count format, we just note it
                    pass

        return events, board_changed

    def apply_resource_dist(self, msg: ResourceDistributionMsg) -> List[str]:
        """Apply type 28 resource distribution."""
        events = []
        # Group by owner for compact display
        by_owner: Dict[int, List[str]] = defaultdict(list)
        for dist in msg.distributions:
            owner = dist.get("owner", -1)
            card = dist.get("card", 0)
            tile_idx = dist.get("tileIndex", -1)
            resource = COLONIST_RESOURCE_TO_CATAN.get(card)
            if not resource:
                continue  # Skip non-resource cards
            self.player_resources[owner][resource] = (
                self.player_resources[owner].get(resource, 0) + 1
            )
            tile_info = ""
            tile = self.tiles.get(tile_idx)
            if tile:
                tile_info = f" (tile {tile_idx})"
            by_owner[owner].append(f"+{resource}{tile_info}")

        for owner, gains in by_owner.items():
            events.append(
                f"  {c(ANSI_CYAN, '[RSRC]')}  {player_str(owner)} {', '.join(gains)}"
            )
        return events

    def apply_trade(self, msg: TradeExecutionMsg) -> List[str]:
        """Apply type 43 trade execution."""
        giving = msg.giving_player
        receiving = msg.receiving_player

        # Filter out non-resource cards (0=hidden/steal, 7+=dev cards)
        def card_name(card_val):
            return COLONIST_RESOURCE_TO_CATAN.get(card_val)

        giving_resources = [card_name(c_) for c_ in msg.giving_cards]
        receiving_resources = [card_name(c_) for c_ in msg.receiving_cards]

        # Check for steal/rob events (card value 0 = hidden card)
        has_hidden = any(c_ == 0 for c_ in msg.giving_cards) or any(
            c_ == 0 for c_ in msg.receiving_cards
        )
        if has_hidden and len(msg.giving_cards) <= 1 and len(msg.receiving_cards) <= 1:
            return [
                f"  {c(ANSI_MAGENTA, '[STEAL]')} {player_str(giving)} "
                f"steals from {player_str(receiving)}"
            ]

        # Count actual resources (skip None = dev cards / hidden)
        give_counts = defaultdict(int)
        for r in giving_resources:
            if r:
                give_counts[r] += 1
        get_counts = defaultdict(int)
        for r in receiving_resources:
            if r:
                get_counts[r] += 1

        # Detect dev card transactions (all cards are non-resource)
        has_dev_give = any(c_ >= 7 for c_ in msg.giving_cards)
        has_dev_recv = any(c_ >= 7 for c_ in msg.receiving_cards)

        # Pure dev card events — no resource cards on either side
        if not give_counts and not get_counts:
            if has_dev_give or has_dev_recv:
                # Dev card purchase/play — show but don't track resources
                if giving == 0 or receiving == 0:
                    player = giving if giving > 0 else receiving
                    return [
                        f"  {c(ANSI_DIM, '[DEV]')}   {player_str(player)} "
                        f"dev card transaction"
                    ]
                return []  # Internal dev card event, skip
            return []  # All hidden cards, nothing to show

        # Update resource tracking (only for known resources)
        for res, count in give_counts.items():
            self.player_resources[giving][res] = max(
                0, self.player_resources[giving].get(res, 0) - count
            )
            if receiving > 0:
                self.player_resources[receiving][res] = (
                    self.player_resources[receiving].get(res, 0) + count
                )
        for res, count in get_counts.items():
            if receiving > 0:
                self.player_resources[receiving][res] = max(
                    0, self.player_resources[receiving].get(res, 0) - count
                )
            self.player_resources[giving][res] = (
                self.player_resources[giving].get(res, 0) + count
            )

        give_str = ", ".join(
            f"{r} x{n}" if n > 1 else r for r, n in give_counts.items()
        )
        get_str = ", ".join(
            f"{r} x{n}" if n > 1 else r for r, n in get_counts.items()
        )

        # Bank interactions
        if receiving == 0 or giving == 0:
            player = giving if giving > 0 else receiving
            if giving == 0:
                # Bank gives resources to player (Year of Plenty, etc.)
                # give_counts has what the bank gives = what the player gets
                if give_counts:
                    return [
                        f"  {c(ANSI_MAGENTA, '[GAIN]')} {player_str(player)} "
                        f"receives [{give_str}] from bank"
                    ]
                return [
                    f"  {c(ANSI_MAGENTA, '[GAIN]')} {player_str(player)} "
                    f"receives from bank"
                ]
            if not get_counts:
                # Giving to bank with no return = discard
                return [
                    f"  {c(ANSI_MAGENTA, '[DISCARD]')} {player_str(giving)} "
                    f"discards [{give_str}]"
                ]
            # Bank trade (4:1, 3:1, 2:1 port)
            return [
                f"  {c(ANSI_MAGENTA, '[TRADE]')} {player_str(giving)} gives [{give_str}] "
                f"to BANK for [{get_str}]"
            ]

        # Player-to-player
        if not get_counts:
            # One-way transfer (monopoly card effect)
            return [
                f"  {c(ANSI_MAGENTA, '[TRADE]')} {player_str(giving)} gives [{give_str}] "
                f"to {player_str(receiving)}"
            ]
        return [
            f"  {c(ANSI_MAGENTA, '[TRADE]')} {player_str(giving)} gives [{give_str}] "
            f"to {player_str(receiving)} for [{get_str}]"
        ]

    def _nearby_tiles_desc(self, vertex_idx: int) -> str:
        """Get a description of tiles near a vertex for context."""
        v = self.vertices.get(vertex_idx)
        if not v:
            return ""
        # Find tiles adjacent to this vertex by checking proximity
        # In the colonist coord system, a vertex at (vx, vy, vz) is adjacent to
        # specific tiles based on z value
        nearby = []
        vx, vy, vz = v.x, v.y, v.z
        if vz == 0:
            # z=0 vertex: adjacent tiles at (vx, vy), (vx-1, vy+1), (vx, vy+1) roughly
            candidates = [(vx, vy), (vx - 1, vy), (vx - 1, vy + 1)]
        else:
            # z=1 vertex: adjacent tiles at (vx, vy), (vx, vy+1), (vx+1, vy) roughly
            candidates = [(vx, vy), (vx, vy + 1), (vx - 1, vy + 1)]

        for tile in self.tiles.values():
            if (tile.x, tile.y) in candidates:
                abbrev = RESOURCE_ABBREV.get(tile.resource, "??")
                num = str(tile.dice_num) if tile.dice_num > 0 else "--"
                nearby.append(f"{abbrev}:{num}")

        if nearby:
            return f" (near {', '.join(nearby)})"
        return ""


# ---------------------------------------------------------------------------
# ASCII hex board renderer
# ---------------------------------------------------------------------------

def render_board(state: ReplayState) -> str:
    """Render the hex board as ASCII art with subtle coordinate axes."""
    if not state.tiles:
        return "  (no board data)"

    # Group tiles by y coordinate (rows)
    rows: Dict[int, List[TileInfo]] = defaultdict(list)
    for tile in state.tiles.values():
        rows[tile.y].append(tile)

    # Sort rows by y, tiles within row by x
    sorted_y = sorted(rows.keys())
    for y in sorted_y:
        rows[y].sort(key=lambda t: t.x)

    # Find the max row width for centering
    max_tiles = max(len(rows[y]) for y in sorted_y)

    lines = []
    cell_width = 8  # width of each tile cell like "[Wo:6 ]"
    y_label_width = 4  # space for "y=N " label on the left

    # X-axis header: show x values for the widest row
    widest_y = max(sorted_y, key=lambda y: len(rows[y]))
    widest_row = rows[widest_y]
    x_labels = []
    for tile in widest_row:
        x_labels.append(f"x={tile.x}")
    # Pad x-labels to cell_width and align with the widest row
    x_header_indent = " " * y_label_width  # no row gap for widest row
    x_header = x_header_indent + "  ".join(f"{xl:^{cell_width - 1}s}" for xl in x_labels)
    lines.append(c(ANSI_GRAY, x_header))

    for y in sorted_y:
        row_tiles = rows[y]
        # Indent to center shorter rows
        gap = max_tiles - len(row_tiles)
        indent = " " * (gap * (cell_width // 2))

        # Y-axis label
        y_label = c(ANSI_GRAY, f"y={y:+d}") if y != 0 else c(ANSI_GRAY, "y=0 ")
        # Pad label to fixed width (use raw width since ANSI is invisible)
        y_prefix = f"{y_label} " if len(str(y)) < 2 else f"{y_label}"

        cells = []
        for tile in row_tiles:
            abbrev = RESOURCE_ABBREV.get(tile.resource, "??")
            if tile.dice_num > 0:
                num_str = str(tile.dice_num)
            else:
                num_str = "--"

            # Highlight 6 and 8 (most common rolls)
            if tile.dice_num in (6, 8) and use_color:
                num_str = c(ANSI_BOLD + "\033[91m", num_str)

            robber = ""
            if tile.index == state.robber_tile:
                robber = c(ANSI_MAGENTA, "*") if use_color else "*"

            cell = f"[{abbrev}:{num_str:>2s}{robber}]"
            cells.append(cell)

        lines.append(y_prefix + indent + " ".join(cells))

    # Add buildings list
    buildings = []
    for v in sorted(state.vertices.values(), key=lambda v: v.index):
        if v.owner >= 0 and v.building_type > 0:
            bname = "C" if v.building_type == 2 else "S"
            ansi = PLAYER_ANSI.get(v.owner, "")
            buildings.append(
                c(ansi, f"P{v.owner}:{bname}@v{v.index}")
            )

    roads = []
    for e in sorted(state.edges.values(), key=lambda e: e.index):
        if e.owner >= 0:
            ansi = PLAYER_ANSI.get(e.owner, "")
            roads.append(c(ansi, f"P{e.owner}:R@e{e.index}"))

    result = "\n".join(lines)
    if buildings:
        result += "\n  Buildings: " + "  ".join(buildings)
    if roads:
        result += "\n  Roads: " + "  ".join(roads)

    return result


# ---------------------------------------------------------------------------
# Turn header formatting
# ---------------------------------------------------------------------------

def turn_header(state: ReplayState) -> str:
    """Format a turn header line."""
    player = player_str(state.current_player)

    if state.is_setup:
        phase = "Setup"
    elif state.turn_state == 1:
        phase = "Roll"
    elif state.turn_state == 2:
        phase = "Play"
    elif state.turn_state == 3:
        phase = "Game Over"
    else:
        phase = f"state={state.turn_state}"

    # Action state detail
    action_desc = ""
    if state.action_state == 1:
        action_desc = " - Place Settlement"
    elif state.action_state == 3:
        action_desc = " - Place Road"
    elif state.action_state == 24:
        action_desc = " - Move Robber"
    elif state.action_state == 27:
        action_desc = " - Rob Player"
    elif state.action_state == 28:
        action_desc = " - Discard"

    me = " (YOU)" if state.current_player == state.my_color else ""
    line = f"── Turn: {player} ({phase}{action_desc}){me} "
    return c(ANSI_DIM, line + "─" * max(0, 50 - len(line)))


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay_log(filepath: str):
    """Replay a JSONL game log with ASCII board and event narration."""
    state = ReplayState()
    last_player = -1
    last_turn_state = -1
    initialized = False

    print(c(ANSI_BOLD, "═" * 55))
    print(c(ANSI_BOLD, "  GAME REPLAY"))
    print(c(ANSI_BOLD, "═" * 55))
    print(f"  Log file: {filepath}")
    print()

    # Pre-parse all incoming messages so we can do lookahead for dice ordering.
    # colonist.io sends ResourceDistribution (type 28) BEFORE the diceState diff,
    # so we peek ahead to get dice values and display them first.
    messages = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("direction") != "in":
                continue
            raw = entry.get("raw", "")
            if not raw:
                continue
            msg = parse_message(raw)
            if isinstance(msg, UnknownMsg):
                fallback = try_parse_truncated_type4(raw)
                if fallback:
                    msg = fallback
            messages.append(msg)

    def _find_dice_ahead(start_idx: int) -> Optional[Tuple[int, int, int]]:
        """Look ahead from start_idx for the next diceState with diceThrown=True.
        Returns (total, dice1, dice2) or None. Only looks within 5 messages.
        Handles partial diffs where only dice1 or dice2 is present."""
        for j in range(start_idx, min(start_idx + 5, len(messages))):
            m = messages[j]
            if isinstance(m, GameStateDiffMsg):
                ds = m.diff.get("diceState", {})
                if ds.get("diceThrown"):
                    # Partial diff: use current state values for missing keys
                    d1 = ds.get("dice1", state.dice1)
                    d2 = ds.get("dice2", state.dice2)
                    return (d1 + d2, d1, d2)
        return None

    # Track which dice diffs we already printed via lookahead
    dice_already_printed: set = set()

    for i, msg in enumerate(messages):
        state.total_messages += 1

        if isinstance(msg, UnknownMsg):
            state.unknown_messages += 1
            continue

        if isinstance(msg, HeartbeatMsg):
            continue

        # --- Game Init ---
        if isinstance(msg, GameInitMsg):
            print(c(ANSI_BOLD, "═" * 55))
            print(c(ANSI_BOLD, "  GAME INIT"))
            print(c(ANSI_BOLD, "═" * 55))
            events = state.init_from_game_init(msg)
            for e in events:
                print(e)
            print()
            print(render_board(state))
            print()
            initialized = True
            continue

        if not initialized:
            continue

        # --- State Diff ---
        if isinstance(msg, GameStateDiffMsg):
            # Check for turn/player change to print header
            diff = msg.diff
            cs = diff.get("currentState", {})
            new_player = cs.get("currentTurnPlayerColor", state.current_player)
            new_turn = cs.get("turnState", state.turn_state)

            # Skip dice display if we already printed it via lookahead
            skip_dice = i in dice_already_printed

            if new_player != last_player or new_turn != last_turn_state:
                events, board_changed = state.apply_diff(msg)
                if skip_dice:
                    events = [e for e in events if "[DICE]" not in e]
                if state.current_player != last_player or state.turn_state != last_turn_state:
                    print()
                    print(turn_header(state))
                    last_player = state.current_player
                    last_turn_state = state.turn_state
                for e in events:
                    print(e)
                if board_changed:
                    print()
                    print(render_board(state))
                    print()
            else:
                events, board_changed = state.apply_diff(msg)
                if skip_dice:
                    events = [e for e in events if "[DICE]" not in e]
                for e in events:
                    print(e)
                if board_changed:
                    print()
                    print(render_board(state))
                    print()
            continue

        # --- Resource Distribution ---
        if isinstance(msg, ResourceDistributionMsg):
            # Lookahead for dice values — colonist sends resources before diceState
            if msg.distributions:
                dice = _find_dice_ahead(i + 1)
                if dice:
                    total, d1, d2 = dice
                    print(
                        f"  {c(ANSI_CYAN, '[DICE]')}  Rolled "
                        f"{c(ANSI_BOLD, str(total))} ({d1}+{d2})"
                    )
                    # Mark the dice diff so we don't print it again
                    for j in range(i + 1, min(i + 6, len(messages))):
                        if isinstance(messages[j], GameStateDiffMsg):
                            ds = messages[j].diff.get("diceState", {})
                            if ds.get("diceThrown"):
                                dice_already_printed.add(j)
                                break

            events = state.apply_resource_dist(msg)
            for e in events:
                print(e)
            continue

        # --- Trade Execution ---
        if isinstance(msg, TradeExecutionMsg):
            events = state.apply_trade(msg)
            for e in events:
                print(e)
            continue

        # --- Discard Prompt ---
        if isinstance(msg, DiscardPromptMsg):
            print(
                f"  {c(ANSI_MAGENTA, '[DISCARD]')} "
                f"Must discard {msg.amount} cards"
            )
            continue

        # --- Rob Victim ---
        if isinstance(msg, RobVictimPromptMsg):
            players = ", ".join(player_str(p) for p in msg.players_to_select)
            print(
                f"  {c(ANSI_MAGENTA, '[ROB]')}   Choose victim: {players}"
            )
            continue

        # --- End Turn ---
        if isinstance(msg, EndTurnMsg):
            print(f"  {c(ANSI_DIM, '[END]')}   Turn over")
            continue

        # --- Game Over ---
        if isinstance(msg, GameOverMsg):
            print()
            print(c(ANSI_BOLD, "═" * 55))
            print(c(ANSI_BOLD, "  GAME OVER"))
            print(c(ANSI_BOLD, "═" * 55))

            egs = msg.end_game_state
            if egs:
                print(f"  End state: {json.dumps(egs, indent=2)[:500]}")
            continue

        # --- Available Actions (quiet) ---
        if isinstance(msg, AvailableActionsMsg):
            continue

    # --- Summary ---
    print()
    print(c(ANSI_BOLD, "═" * 55))
    print(c(ANSI_BOLD, "  SUMMARY"))
    print(c(ANSI_BOLD, "═" * 55))
    print(f"  Messages processed: {state.total_messages} "
          f"({state.unknown_messages} unknown/skipped)")
    print(f"  Completed turns: {state.completed_turns}")
    print(f"  Buildings: {state.settlements_built} settlements, "
          f"{state.cities_built} cities, {state.roads_built} roads")

    print()
    print("  Final board:")
    print(render_board(state))
    print()

    if state.player_resources:
        print("  Resource tracking (from distributions + trades):")
        valid_resources = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        for p in sorted(state.player_resources.keys()):
            if p <= 0:
                continue  # Skip bank
            res = state.player_resources[p]
            res_str = " ".join(
                f"{r}:{res.get(r, 0)}" for r in valid_resources
            )
            me = " (YOU)" if p == state.my_color else ""
            print(f"    {player_str(p)}{me}: {res_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)  # Suppress protocol parser warnings

    if len(sys.argv) < 2:
        print("Usage: python3 replay_game.py <log_file.jsonl> [--no-color]")
        print("Example: python3 replay_game.py logs/full_game_2.jsonl")
        sys.exit(1)

    log_file = sys.argv[1]
    if "--no-color" in sys.argv:
        use_color = False

    try:
        replay_log(log_file)
    except FileNotFoundError:
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  (interrupted)")
        sys.exit(0)
