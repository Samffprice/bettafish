#!/usr/bin/env python3
"""Curses TUI game replay — navigate colonist.io game events with arrow keys.

Reads a JSONL log file, pre-processes all messages into navigable snapshots,
and presents a fullscreen TUI with:
  - Colored hex board with buildings and robber
  - Turn-by-turn event log with context
  - Per-player resource tracking
  - Arrow key navigation between events

Usage:
    python3 replay_tui.py logs/full_game_2.jsonl

Controls:
    ← →  or  h l     Previous / next event
    PgUp PgDn  k j   Skip 10 events
    [ ]               Jump to previous / next turn
    Home End   g G    First / last event
    q  ESC            Quit
"""
import copy
import curses
import json
import logging
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

import replay_game
from replay_game import (
    ReplayState,
    TileInfo,
    VertexInfo,
    EdgeInfo,
    try_parse_truncated_type4,
    RESOURCE_ABBREV,
    RESOURCE_SHORT,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Color pair IDs
# ═══════════════════════════════════════════════════════════════════════════════

CP_WOOD = 1
CP_BRICK = 2
CP_SHEEP = 3
CP_WHEAT = 4
CP_ORE = 5
CP_DESERT = 6
CP_P1 = 7
CP_P2 = 8
CP_P3 = 9
CP_P4 = 10
CP_STATUS = 11
CP_HEADER = 12
CP_ROBBER = 13
CP_DICE_HOT = 14
CP_DIM = 15
CP_TAG_BUILD = 16
CP_TAG_DICE = 17
CP_TAG_TRADE = 18
CP_DIVIDER = 19
CP_BOARD_BG = 20

RESOURCE_CP = {
    "WOOD": CP_WOOD,
    "BRICK": CP_BRICK,
    "SHEEP": CP_SHEEP,
    "WHEAT": CP_WHEAT,
    "ORE": CP_ORE,
    None: CP_DESERT,
}

PLAYER_CP = {1: CP_P1, 2: CP_P2, 3: CP_P3, 4: CP_P4}


def _init_colors():
    """Set up curses color pairs."""
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_WOOD, curses.COLOR_GREEN, -1)
    curses.init_pair(CP_BRICK, curses.COLOR_RED, -1)
    curses.init_pair(CP_SHEEP, curses.COLOR_WHITE, -1)
    curses.init_pair(CP_WHEAT, curses.COLOR_YELLOW, -1)
    curses.init_pair(CP_ORE, curses.COLOR_CYAN, -1)
    curses.init_pair(CP_DESERT, curses.COLOR_MAGENTA, -1)
    curses.init_pair(CP_P1, curses.COLOR_RED, -1)
    curses.init_pair(CP_P2, curses.COLOR_BLUE, -1)
    curses.init_pair(CP_P3, curses.COLOR_YELLOW, -1)
    curses.init_pair(CP_P4, curses.COLOR_WHITE, -1)
    curses.init_pair(CP_STATUS, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(CP_HEADER, curses.COLOR_WHITE, -1)
    curses.init_pair(CP_ROBBER, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
    curses.init_pair(CP_DICE_HOT, curses.COLOR_RED, -1)
    curses.init_pair(CP_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(CP_TAG_BUILD, curses.COLOR_GREEN, -1)
    curses.init_pair(CP_TAG_DICE, curses.COLOR_CYAN, -1)
    curses.init_pair(CP_TAG_TRADE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(CP_DIVIDER, curses.COLOR_WHITE, -1)
    curses.init_pair(CP_BOARD_BG, curses.COLOR_WHITE, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Snapshot — frozen game state at a moment in time
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Snapshot:
    """A navigable moment in the game replay."""
    tiles: Dict[int, TileInfo]
    vertices: Dict[int, VertexInfo]
    edges: Dict[int, EdgeInfo]
    robber_tile: int
    current_player: int
    turn_state: int
    action_state: int
    is_setup: bool
    my_color: int
    player_order: List[int]
    dice_total: int
    dice1: int
    dice2: int
    player_resources: Dict[int, Dict[str, int]]
    events: List[str]
    label: str
    settlements_built: int
    cities_built: int
    roads_built: int
    completed_turns: int


def _snap(state: ReplayState, events: List[str], label: str) -> Snapshot:
    """Create a snapshot from the current replay state."""
    return Snapshot(
        tiles=dict(state.tiles),
        vertices={k: copy.copy(v) for k, v in state.vertices.items()},
        edges={k: copy.copy(v) for k, v in state.edges.items()},
        robber_tile=state.robber_tile,
        current_player=state.current_player,
        turn_state=state.turn_state,
        action_state=state.action_state,
        is_setup=state.is_setup,
        my_color=state.my_color,
        player_order=list(state.player_order),
        dice_total=state.dice_total,
        dice1=state.dice1,
        dice2=state.dice2,
        player_resources={k: dict(v) for k, v in state.player_resources.items()},
        events=list(events),
        label=label,
        settlements_built=state.settlements_built,
        cities_built=state.cities_built,
        roads_built=state.roads_built,
        completed_turns=state.completed_turns,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Build snapshots from JSONL log
# ═══════════════════════════════════════════════════════════════════════════════

def build_snapshots(filepath: str) -> List[Snapshot]:
    """Process all messages into a navigable list of game state snapshots."""
    replay_game.use_color = False  # plain-text events

    state = ReplayState()
    snapshots: List[Snapshot] = []

    # Pre-parse all incoming messages
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

    # Dice lookahead — colonist sends resources BEFORE diceState diff
    def _find_dice_ahead(start_idx: int) -> Optional[Tuple[int, int, int]]:
        for j in range(start_idx, min(start_idx + 5, len(messages))):
            m = messages[j]
            if isinstance(m, GameStateDiffMsg):
                ds = m.diff.get("diceState", {})
                if ds.get("diceThrown"):
                    d1 = ds.get("dice1", state.dice1)
                    d2 = ds.get("dice2", state.dice2)
                    return (d1 + d2, d1, d2)
        return None

    dice_already_printed: set = set()
    initialized = False

    for i, msg in enumerate(messages):
        state.total_messages += 1

        if isinstance(msg, (UnknownMsg, HeartbeatMsg, AvailableActionsMsg)):
            if isinstance(msg, UnknownMsg):
                state.unknown_messages += 1
            continue

        # Game Init
        if isinstance(msg, GameInitMsg):
            events = state.init_from_game_init(msg)
            snapshots.append(_snap(state, ["GAME INIT"] + events, "Game Init"))
            initialized = True
            continue

        if not initialized:
            continue

        # State Diff
        if isinstance(msg, GameStateDiffMsg):
            skip_dice = i in dice_already_printed
            events, board_changed = state.apply_diff(msg)
            if skip_dice:
                events = [e for e in events if "[DICE]" not in e]
            if events:
                snapshots.append(_snap(state, events, events[0].strip()))
            continue

        # Resource Distribution
        if isinstance(msg, ResourceDistributionMsg):
            all_events: List[str] = []
            if msg.distributions:
                dice = _find_dice_ahead(i + 1)
                if dice:
                    total, d1, d2 = dice
                    all_events.append(f"  [DICE]  Rolled {total} ({d1}+{d2})")
                    for j in range(i + 1, min(i + 6, len(messages))):
                        if isinstance(messages[j], GameStateDiffMsg):
                            ds = messages[j].diff.get("diceState", {})
                            if ds.get("diceThrown"):
                                dice_already_printed.add(j)
                                break
            all_events.extend(state.apply_resource_dist(msg))
            if all_events:
                snapshots.append(_snap(state, all_events, all_events[0].strip()))
            continue

        # Trade
        if isinstance(msg, TradeExecutionMsg):
            events = state.apply_trade(msg)
            if events:
                snapshots.append(_snap(state, events, events[0].strip()))
            continue

        # Discard Prompt
        if isinstance(msg, DiscardPromptMsg):
            ev = f"  [DISCARD] Must discard {msg.amount} cards"
            snapshots.append(_snap(state, [ev], ev.strip()))
            continue

        # Rob Victim
        if isinstance(msg, RobVictimPromptMsg):
            players = ", ".join(f"P{p}" for p in msg.players_to_select)
            ev = f"  [ROB] Choose victim: {players}"
            snapshots.append(_snap(state, [ev], ev.strip()))
            continue

        # End Turn
        if isinstance(msg, EndTurnMsg):
            ev = "  [END] Turn over"
            snapshots.append(_snap(state, [ev], ev.strip()))
            continue

        # Game Over
        if isinstance(msg, GameOverMsg):
            extra = []
            if msg.end_game_state:
                egs = msg.end_game_state
                winner = egs.get("winner", "?")
                extra.append(f"  Winner: P{winner}")
                scores = egs.get("scores")
                if scores:
                    extra.append(f"  Scores: {scores}")
            snapshots.append(_snap(state, ["GAME OVER"] + extra, "Game Over"))
            continue

    return snapshots


# ═══════════════════════════════════════════════════════════════════════════════
#  Curses drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0):
    """addstr that silently ignores curses errors (e.g. writing past edge)."""
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _safe_addnstr(win, y: int, x: int, text: str, n: int, attr: int = 0):
    """addnstr that silently ignores curses errors."""
    try:
        win.addnstr(y, x, text, n, attr)
    except curses.error:
        pass


def _phase_str(snap: Snapshot) -> str:
    """Get the current phase description."""
    if snap.is_setup:
        return "Setup"
    if snap.turn_state == 1:
        return "Roll"
    if snap.turn_state == 2:
        return "Play"
    if snap.turn_state == 3:
        return "Game Over"
    return f"state={snap.turn_state}"


def _action_str(snap: Snapshot) -> str:
    """Get the current action state description."""
    m = {1: "Place Settlement", 3: "Place Road", 24: "Move Robber",
         27: "Rob Player", 28: "Discard"}
    return m.get(snap.action_state, "")


# ═══════════════════════════════════════════════════════════════════════════════
#  Board renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_board(win, snap: Snapshot, start_y: int, start_x: int,
                max_width: int) -> int:
    """Draw the hex board with colored tiles. Returns lines used."""
    if not snap.tiles:
        _safe_addstr(win, start_y, start_x, "(no board data)")
        return 1

    # Group tiles by y coordinate (rows)
    rows: Dict[int, List[TileInfo]] = defaultdict(list)
    for tile in snap.tiles.values():
        rows[tile.y].append(tile)

    sorted_y = sorted(rows.keys())
    for y in sorted_y:
        rows[y].sort(key=lambda t: t.x)

    max_tiles = max(len(rows[y]) for y in sorted_y)
    cell_width = 8  # "[Wo:5 ] " = 8 chars
    line = start_y

    for y_val in sorted_y:
        row_tiles = rows[y_val]
        gap = max_tiles - len(row_tiles)
        indent = start_x + gap * (cell_width // 2)

        col = indent
        for tile in row_tiles:
            abbrev = RESOURCE_ABBREV.get(tile.resource, "??")
            cp = RESOURCE_CP.get(tile.resource, CP_DESERT)
            num_str = str(tile.dice_num) if tile.dice_num > 0 else "--"
            is_robber = tile.index == snap.robber_tile
            is_hot = tile.dice_num in (6, 8)

            if col + cell_width > start_x + max_width:
                break  # Don't overflow

            if is_robber:
                # Robber tile: magenta reverse
                cell = f" {abbrev}:{num_str:>2s}R "
                _safe_addstr(win, line, col, cell,
                             curses.color_pair(CP_ROBBER) | curses.A_BOLD)
            else:
                _safe_addstr(win, line, col, "[",
                             curses.color_pair(CP_DIM) | curses.A_DIM)
                _safe_addstr(win, line, col + 1, abbrev,
                             curses.color_pair(cp) | curses.A_BOLD)
                _safe_addstr(win, line, col + 3, ":",
                             curses.color_pair(CP_DIM) | curses.A_DIM)
                if is_hot:
                    _safe_addstr(win, line, col + 4, f"{num_str:>2s}",
                                 curses.color_pair(CP_DICE_HOT) | curses.A_BOLD)
                else:
                    _safe_addstr(win, line, col + 4, f"{num_str:>2s}",
                                 curses.color_pair(CP_HEADER))
                _safe_addstr(win, line, col + 6, "] ",
                             curses.color_pair(CP_DIM) | curses.A_DIM)

            col += cell_width
        line += 1

    # --- Buildings ---
    buildings = []
    for v in sorted(snap.vertices.values(), key=lambda v: v.index):
        if v.owner >= 0 and v.building_type > 0:
            sym = "\u2605" if v.building_type == 2 else "\u25B2"  # ★ city, ▲ settlement
            buildings.append((v.owner, f"{sym}v{v.index}"))

    if buildings:
        line += 1
        col = start_x + 2
        _safe_addstr(win, line, start_x, "  ", curses.color_pair(CP_DIM))
        for owner, text in buildings:
            cp = PLAYER_CP.get(owner, CP_HEADER)
            label = f"P{owner}:"
            _safe_addstr(win, line, col, label,
                         curses.color_pair(cp) | curses.A_BOLD)
            col += len(label)
            _safe_addstr(win, line, col, text + " ",
                         curses.color_pair(cp))
            col += len(text) + 1
            if col > start_x + max_width - 15:
                line += 1
                col = start_x + 2

    # --- Roads ---
    roads = []
    for e in sorted(snap.edges.values(), key=lambda e: e.index):
        if e.owner >= 0:
            roads.append((e.owner, f"e{e.index}"))

    if roads:
        line += 1
        col = start_x + 2
        # Group roads by owner for compactness
        by_owner: Dict[int, List[str]] = defaultdict(list)
        for owner, text in roads:
            by_owner[owner].append(text)

        for owner in sorted(by_owner.keys()):
            cp = PLAYER_CP.get(owner, CP_HEADER)
            label = f"P{owner}:"
            road_list = ",".join(by_owner[owner][:12])  # cap display
            if len(by_owner[owner]) > 12:
                road_list += f"..+{len(by_owner[owner]) - 12}"
            entry = f"{label}{road_list} "
            _safe_addstr(win, line, col, f"P{owner}:",
                         curses.color_pair(cp) | curses.A_BOLD)
            col += len(label)
            _safe_addstr(win, line, col, road_list + " ",
                         curses.color_pair(cp) | curses.A_DIM)
            col += len(road_list) + 1
            if col > start_x + max_width - 20:
                line += 1
                col = start_x + 2

    return line - start_y + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Event renderer — colorized tags and player names
# ═══════════════════════════════════════════════════════════════════════════════

_TAG_RE = re.compile(
    r'\[(DICE|BUILD|RSRC|TRADE|STEAL|DISCARD|GAIN|DEV|ROBBER|ROB|END|TURN)\]'
)
_PLAYER_RE = re.compile(r'P([1-4])')
_SPLIT_RE = re.compile(
    r'(\[(?:DICE|BUILD|RSRC|TRADE|STEAL|DISCARD|GAIN|DEV|ROBBER|ROB|END|TURN)\]'
    r'|P[1-4])'
)

_TAG_CP = {
    "DICE": CP_TAG_DICE, "RSRC": CP_TAG_DICE,
    "BUILD": CP_TAG_BUILD,
    "TRADE": CP_TAG_TRADE, "STEAL": CP_TAG_TRADE, "DISCARD": CP_TAG_TRADE,
    "GAIN": CP_TAG_TRADE, "ROBBER": CP_TAG_TRADE, "ROB": CP_TAG_TRADE,
    "DEV": CP_DIM, "END": CP_DIM, "TURN": CP_HEADER,
}


def _draw_event(win, y: int, x: int, text: str, max_w: int,
                dim: bool = False):
    """Draw an event line with colorized tags and player names."""
    parts = _SPLIT_RE.split(text)
    pos = x
    for part in parts:
        if not part:
            continue
        remaining = max_w - (pos - x)
        if remaining <= 0:
            break

        # Is this a tag like [BUILD]?
        tag_m = _TAG_RE.fullmatch(part)
        if tag_m:
            tag_name = tag_m.group(1)
            cp = _TAG_CP.get(tag_name, CP_HEADER)
            attr = curses.color_pair(cp) | curses.A_BOLD
            if dim:
                attr = curses.color_pair(CP_DIM) | curses.A_DIM
            _safe_addnstr(win, y, pos, part, remaining, attr)
            pos += len(part)
            continue

        # Is this a player reference like P1?
        pl_m = _PLAYER_RE.fullmatch(part)
        if pl_m:
            p = int(pl_m.group(1))
            cp = PLAYER_CP.get(p, CP_HEADER)
            attr = curses.color_pair(cp) | curses.A_BOLD
            if dim:
                attr |= curses.A_DIM
            _safe_addnstr(win, y, pos, part, remaining, attr)
            pos += len(part)
            continue

        # Plain text
        attr = curses.A_NORMAL
        if dim:
            attr = curses.color_pair(CP_DIM) | curses.A_DIM
        _safe_addnstr(win, y, pos, part, remaining, attr)
        pos += len(part)


# ═══════════════════════════════════════════════════════════════════════════════
#  Resource display
# ═══════════════════════════════════════════════════════════════════════════════

RES_ORDER = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
RES_SHORT = {"WOOD": "W", "BRICK": "B", "SHEEP": "S", "WHEAT": "Wh", "ORE": "O"}


def _draw_resources(win, snap: Snapshot, y: int, x: int, width: int) -> int:
    """Draw compact resource display. Returns lines used."""
    col = x
    for p in snap.player_order:
        if p <= 0:
            continue
        res = snap.player_resources.get(p, {})
        cp = PLAYER_CP.get(p, CP_HEADER)
        me = "*" if p == snap.my_color else ""

        label = f"P{p}{me} "
        _safe_addstr(win, y, col, label,
                     curses.color_pair(cp) | curses.A_BOLD)
        col += len(label)

        for r in RES_ORDER:
            rcp = RESOURCE_CP.get(r, CP_HEADER)
            short = RES_SHORT[r]
            count = res.get(r, 0)
            _safe_addstr(win, y, col, short + ":",
                         curses.color_pair(rcp))
            col += len(short) + 1
            _safe_addstr(win, y, col, f"{count} ",
                         curses.color_pair(CP_HEADER))
            col += len(str(count)) + 1

        col += 2  # gap between players

        # Wrap to next line if needed
        if col > x + width - 30 and p != snap.player_order[-1]:
            y += 1
            col = x

    return 1  # typically fits on 1-2 lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Main TUI
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tui(stdscr, snapshots: List[Snapshot], filepath: str):
    """Main curses event loop."""
    _init_colors()
    curses.curs_set(0)
    stdscr.timeout(-1)
    stdscr.keypad(True)

    # Find first build event to start at
    start_idx = 0
    for i, s in enumerate(snapshots):
        if any("[BUILD]" in e for e in s.events):
            start_idx = i
            break

    idx = start_idx
    CONTEXT_COUNT = 10  # how many previous snapshots' events to show as context

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        if height < 10 or width < 40:
            _safe_addstr(stdscr, 0, 0, "Terminal too small. Resize to 80x24+.")
            stdscr.refresh()
            stdscr.getch()
            continue

        snap = snapshots[idx]

        # ┌─ Title bar ─────────────────────────────────────────────────────┐
        title = f" Catan Replay: {filepath.split('/')[-1]} "
        _safe_addstr(stdscr, 0, 0, " " * width, curses.color_pair(CP_STATUS))
        _safe_addstr(stdscr, 0, max(0, (width - len(title)) // 2), title,
                     curses.color_pair(CP_STATUS) | curses.A_BOLD)

        # ┌─ Turn header ──────────────────────────────────────────────────┐
        player_label = f"P{snap.current_player}"
        phase = _phase_str(snap)
        action = _action_str(snap)
        me = " (YOU)" if snap.current_player == snap.my_color else ""
        action_str = f" - {action}" if action else ""

        turn_text = f" Turn: {player_label} ({phase}{action_str}){me} "
        pad = max(0, width - len(turn_text) - 6)
        turn_line = f"──{turn_text}" + "─" * pad

        cp = PLAYER_CP.get(snap.current_player, CP_HEADER)
        _safe_addnstr(stdscr, 2, 1, turn_line, width - 2,
                      curses.color_pair(cp) | curses.A_BOLD)

        # ┌─ Board ────────────────────────────────────────────────────────┐
        board_start_y = 4
        board_lines = _draw_board(stdscr, snap, board_start_y, 3, width - 6)

        # ┌─ Divider ──────────────────────────────────────────────────────┐
        div_y = board_start_y + board_lines + 1
        _safe_addnstr(stdscr, div_y, 0, "─" * width, width,
                      curses.color_pair(CP_DIVIDER) | curses.A_DIM)

        # ┌─ Event log ────────────────────────────────────────────────────┐
        # Reserve space: 1 line divider + resource lines + 1 status bar
        res_lines = 2  # resources + divider above them
        event_start_y = div_y + 1
        event_end_y = height - res_lines - 1  # last usable event line
        event_capacity = event_end_y - event_start_y

        if event_capacity < 1:
            event_capacity = 1

        # Collect events: current snapshot's events + context from recent ones
        all_events: List[Tuple[str, bool]] = []  # (text, is_current)
        context_start = max(0, idx - CONTEXT_COUNT)
        for ci in range(context_start, idx + 1):
            is_current = ci == idx
            for ev in snapshots[ci].events:
                all_events.append((ev, is_current))

        # Show the tail that fits
        if len(all_events) > event_capacity:
            all_events = all_events[-event_capacity:]

        ey = event_start_y
        for ev_text, is_current in all_events:
            if ey >= event_end_y:
                break
            if is_current:
                # Current event: bright marker
                _safe_addstr(stdscr, ey, 0, "\u25B6",
                             curses.color_pair(CP_TAG_BUILD) | curses.A_BOLD)
                _draw_event(stdscr, ey, 1, ev_text, width - 2, dim=False)
            else:
                _safe_addstr(stdscr, ey, 0, " ")
                _draw_event(stdscr, ey, 1, ev_text, width - 2, dim=True)
            ey += 1

        # ┌─ Resources ────────────────────────────────────────────────────┐
        res_div_y = height - res_lines - 1
        _safe_addnstr(stdscr, res_div_y, 0, "─" * width, width,
                      curses.color_pair(CP_DIVIDER) | curses.A_DIM)
        _draw_resources(stdscr, snap, res_div_y + 1, 1, width - 2)

        # ┌─ Status bar ───────────────────────────────────────────────────┐
        status_y = height - 1
        builds = f"S:{snap.settlements_built} C:{snap.cities_built} R:{snap.roads_built}"
        status = (
            f" \u25C4 \u25BA Navigate | "
            f"{idx + 1}/{len(snapshots)} | "
            f"Turn {snap.completed_turns} | "
            f"{builds} | "
            f"[/] Jump turn | q Quit"
        )
        _safe_addstr(stdscr, status_y, 0, " " * width,
                     curses.color_pair(CP_STATUS))
        _safe_addnstr(stdscr, status_y, 0, status, width - 1,
                      curses.color_pair(CP_STATUS))

        stdscr.refresh()

        # ┌─ Input ────────────────────────────────────────────────────────┐
        key = stdscr.getch()

        if key in (ord('q'), ord('Q'), 27):  # q, Q, ESC
            break
        elif key == curses.KEY_RIGHT or key == ord('l'):
            idx = min(len(snapshots) - 1, idx + 1)
        elif key == curses.KEY_LEFT or key == ord('h'):
            idx = max(0, idx - 1)
        elif key == curses.KEY_NPAGE or key == ord('j'):
            idx = min(len(snapshots) - 1, idx + 10)
        elif key == curses.KEY_PPAGE or key == ord('k'):
            idx = max(0, idx - 10)
        elif key == curses.KEY_HOME or key == ord('g'):
            idx = 0
        elif key == curses.KEY_END or key == ord('G'):
            idx = len(snapshots) - 1
        elif key == ord(']'):
            # Jump to next turn (different current_player)
            cur_player = snap.current_player
            for ni in range(idx + 1, len(snapshots)):
                if snapshots[ni].current_player != cur_player:
                    idx = ni
                    break
            else:
                idx = len(snapshots) - 1
        elif key == ord('['):
            # Jump to previous turn (different current_player)
            cur_player = snap.current_player
            for ni in range(idx - 1, -1, -1):
                if snapshots[ni].current_player != cur_player:
                    idx = ni
                    break
            else:
                idx = 0
        elif key == curses.KEY_RESIZE:
            pass  # Just redraw


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.disable(logging.CRITICAL)

    if len(sys.argv) < 2:
        print("Usage: python3 replay_tui.py <log_file.jsonl>")
        print("Example: python3 replay_tui.py logs/full_game_2.jsonl")
        print()
        print("Controls:")
        print("  ← →  or  h l     Previous / next event")
        print("  PgUp PgDn  k j   Skip 10 events")
        print("  [ ]               Jump to previous / next turn")
        print("  Home End   g G    First / last event")
        print("  q  ESC            Quit")
        sys.exit(1)

    filepath = sys.argv[1]

    print("Building game snapshots...")
    try:
        snapshots = build_snapshots(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if not snapshots:
        print("Error: No game events found in log file.")
        sys.exit(1)

    print(f"Loaded {len(snapshots)} events. Launching TUI...")
    curses.wrapper(lambda stdscr: _run_tui(stdscr, snapshots, filepath))


if __name__ == "__main__":
    main()
