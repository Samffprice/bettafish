"""Core game replay and feature extraction for dataset parsing.

Replays colonist.io game events to reconstruct Catanatron Game objects,
then extracts feature vectors using create_sample() at each sampling point
(turnState transitions to PLAY=2, i.e. after each dice roll).

Performance: uses incremental board building (buildings applied once as they
occur) and single reconstruction per sampling point (not per player).
"""
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catanatron.game import Game
from catanatron.models.board import STATIC_GRAPH, Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    ActionPrompt,
    SETTLEMENT,
    CITY,
    ROAD,
)
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, Player
from catanatron.state import State, PLAYER_INITIAL_STATE
from catanatron.models.decks import (
    RESOURCE_FREQDECK_INDEXES,
    starting_resource_bank,
    starting_devcard_bank,
)
from catanatron.features import create_sample, get_feature_ordering

from bridge.coordinate_map import CoordinateMapper, build_coordinate_mapper
from bridge.config import (
    COLONIST_RESOURCE_TO_CATAN,
    COLONIST_VALUE_TO_DEVCARD,
)
from datasets.format_adapter import (
    adapt_tiles,
    adapt_vertices,
    adapt_edges,
    extract_harbor_pairs,
    build_index_maps,
)

logger = logging.getLogger(__name__)

CATANATRON_COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# Colonist resource ID -> Catanatron resource string
RESOURCE_ID_TO_STR = COLONIST_RESOURCE_TO_CATAN  # {1:WOOD, 2:BRICK, 3:SHEEP, 4:WHEAT, 5:ORE}

# Colonist dev card ID -> Catanatron dev card string
DEVCARD_ID_TO_STR = COLONIST_VALUE_TO_DEVCARD  # {11:KNIGHT, 12:VICTORY_POINT, ...}


class IncrementalGame:
    """Maintains a Catanatron Game incrementally for fast feature extraction.

    Buildings are applied to the board as events arrive (not replayed from
    scratch at each sampling point). At sampling points, only the lightweight
    player_state fields (resources, dev cards, VP) are updated before calling
    create_sample().
    """

    def __init__(self, mapper: CoordinateMapper, play_order: List[int],
                 colonist_to_catan: Dict[int, Color]):
        self.mapper = mapper
        self.play_order = play_order
        self.colonist_to_catan = colonist_to_catan

        catan_map = mapper.catan_map
        self.colors = tuple(colonist_to_catan[c] for c in play_order)
        players = [Player(color) for color in self.colors]

        self.game = Game(players=[], initialize=False)
        self.game.seed = 0
        self.game.id = "dataset-reconstructed"
        self.game.vps_to_win = 10

        st = State([], None, initialize=False)
        st.players = players
        st.colors = self.colors
        st.color_to_index = {color: i for i, color in enumerate(self.colors)}
        st.discard_limit = 7
        st.player_state = {}
        for index in range(len(self.colors)):
            for key, value in PLAYER_INITIAL_STATE.items():
                st.player_state[f"P{index}_{key}"] = value
        st.resource_freqdeck = starting_resource_bank()
        st.development_listdeck = starting_devcard_bank()
        st.buildings_by_color = {color: defaultdict(list) for color in self.colors}
        st.action_records = []
        st.num_turns = 0
        st.is_initial_build_phase = True
        st.is_discarding = False
        st.is_moving_knight = False
        st.is_road_building = False
        st.free_roads_available = 0
        st.current_prompt = ActionPrompt.PLAY_TURN
        st.is_resolving_trade = False
        st.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        st.acceptees = tuple(False for _ in self.colors)
        st.current_player_index = 0
        st.current_turn_index = 0

        board = Board(catan_map=catan_map, initialize=True)
        st.board = board
        self.game.state = st
        self.game.playable_actions = []

        # Precompute robber tile id -> cube coord lookup
        self._tile_id_to_cube = {}
        for cube_coord, lt in catan_map.land_tiles.items():
            self._tile_id_to_cube[lt.id] = cube_coord

        # Track building dedup (set of (col_coord, color, btype) tuples)
        self._built = set()

    def apply_settlement(self, col_coord, owner_color: int, is_setup: bool):
        key = (col_coord, owner_color, SETTLEMENT)
        if key in self._built:
            return
        self._built.add(key)

        node_id = self.mapper.colonist_vertex_to_catan.get(col_coord)
        if node_id is None:
            return
        catan_color = self.colonist_to_catan.get(owner_color)
        if catan_color is None:
            return

        st = self.game.state
        p_idx = st.color_to_index[catan_color]
        bbc = st.buildings_by_color[catan_color]

        if node_id in bbc[SETTLEMENT] or node_id in bbc[CITY]:
            return

        try:
            st.board.build_settlement(catan_color, node_id, initial_build_phase=is_setup)
        except ValueError:
            st.board.buildings[node_id] = (catan_color, SETTLEMENT)
            st.board.board_buildable_ids.discard(node_id)
            for n in STATIC_GRAPH.neighbors(node_id):
                st.board.board_buildable_ids.discard(n)
            st.board.connected_components[catan_color].append({node_id})
            st.board.buildable_edges_cache = {}

        bbc[SETTLEMENT].append(node_id)
        st.player_state[f"P{p_idx}_SETTLEMENTS_AVAILABLE"] = max(
            0, st.player_state[f"P{p_idx}_SETTLEMENTS_AVAILABLE"] - 1
        )

    def apply_city(self, col_coord, owner_color: int):
        key = (col_coord, owner_color, CITY)
        if key in self._built:
            return
        self._built.add(key)

        node_id = self.mapper.colonist_vertex_to_catan.get(col_coord)
        if node_id is None:
            return
        catan_color = self.colonist_to_catan.get(owner_color)
        if catan_color is None:
            return

        st = self.game.state
        p_idx = st.color_to_index[catan_color]
        bbc = st.buildings_by_color[catan_color]

        if node_id in bbc[CITY]:
            return

        try:
            st.board.build_city(catan_color, node_id)
        except ValueError:
            st.board.buildings[node_id] = (catan_color, CITY)

        if node_id in bbc[SETTLEMENT]:
            bbc[SETTLEMENT].remove(node_id)
            st.player_state[f"P{p_idx}_SETTLEMENTS_AVAILABLE"] += 1
        bbc[CITY].append(node_id)
        st.player_state[f"P{p_idx}_CITIES_AVAILABLE"] = max(
            0, st.player_state[f"P{p_idx}_CITIES_AVAILABLE"] - 1
        )

    def apply_road(self, col_coord, owner_color: int):
        key = (col_coord, owner_color, ROAD)
        if key in self._built:
            return
        self._built.add(key)

        catan_edge = self.mapper.colonist_edge_to_catan.get(col_coord)
        if catan_edge is None:
            return
        catan_color = self.colonist_to_catan.get(owner_color)
        if catan_color is None:
            return

        st = self.game.state
        p_idx = st.color_to_index[catan_color]
        bbc = st.buildings_by_color[catan_color]

        if catan_edge in bbc[ROAD]:
            return

        try:
            st.board.build_road(catan_color, catan_edge)
        except ValueError:
            st.board.roads[catan_edge] = catan_color
            st.board.roads[(catan_edge[1], catan_edge[0])] = catan_color
            a, b = catan_edge
            merged = False
            for comp in st.board.connected_components[catan_color]:
                if a in comp or b in comp:
                    comp.add(a)
                    comp.add(b)
                    merged = True
                    break
            if not merged:
                st.board.connected_components[catan_color].append({a, b})
            st.board.buildable_edges_cache = {}

        bbc[ROAD].append(catan_edge)
        st.player_state[f"P{p_idx}_ROADS_AVAILABLE"] = max(
            0, st.player_state[f"P{p_idx}_ROADS_AVAILABLE"] - 1
        )

    def set_robber(self, tile_index: int):
        xy = self.mapper.colonist_tile_index_to_xy.get(tile_index)
        if xy is None:
            return
        tile = self.mapper.colonist_tile_to_catan.get(xy)
        if tile is None:
            return
        cube = self._tile_id_to_cube.get(tile.id)
        if cube is not None:
            self.game.state.board.robber_coordinate = cube

    def prepare_for_sampling(self, rstate: 'ReplayState'):
        """Update lightweight player_state fields before create_sample() calls."""
        st = self.game.state

        # Current player
        if rstate.current_turn_player in self.colonist_to_catan:
            cur_color = self.colonist_to_catan[rstate.current_turn_player]
            cur_idx = st.color_to_index[cur_color]
        else:
            cur_idx = 0
        st.current_player_index = cur_idx
        st.current_turn_index = cur_idx
        st.num_turns = rstate.turn_number
        st.is_initial_build_phase = rstate.is_setup_phase

        # Bank resources
        for res, count in rstate.bank_resources.items():
            idx = RESOURCE_FREQDECK_INDEXES.get(res)
            if idx is not None:
                st.resource_freqdeck[idx] = count

        # HAS_ROLLED
        for i in range(len(self.colors)):
            st.player_state[f"P{i}_HAS_ROLLED"] = False
        if rstate.turn_state == 2:
            st.player_state[f"P{cur_idx}_HAS_ROLLED"] = True

        # Resources, dev cards, VP for ALL players
        for col_color in self.play_order:
            catan_color = self.colonist_to_catan[col_color]
            p_idx = st.color_to_index[catan_color]
            pk = f"P{p_idx}"

            hand = rstate.resources.get(col_color, {})
            for resource in RESOURCES:
                st.player_state[f"{pk}_{resource}_IN_HAND"] = hand.get(resource, 0)

            dev_hand = rstate.dev_cards.get(col_color, {})
            dev_used = rstate.dev_cards_used.get(col_color, {})
            for dev_card in DEVELOPMENT_CARDS:
                st.player_state[f"{pk}_{dev_card}_IN_HAND"] = dev_hand.get(dev_card, 0)
                st.player_state[f"{pk}_PLAYED_{dev_card}"] = dev_used.get(dev_card, 0)

        # VP from buildings (settlements + cities) — already tracked incrementally
        # We need to recompute VP from scratch each time because longest road /
        # largest army can change, and we stored building VP incrementally.
        # Reset VP to building-based counts, then add bonuses.
        for col_color in self.play_order:
            catan_color = self.colonist_to_catan[col_color]
            p_idx = st.color_to_index[catan_color]
            bbc = st.buildings_by_color[catan_color]
            base_vp = len(bbc[SETTLEMENT]) + 2 * len(bbc[CITY])
            st.player_state[f"P{p_idx}_VICTORY_POINTS"] = base_vp
            st.player_state[f"P{p_idx}_ACTUAL_VICTORY_POINTS"] = base_vp

        # Longest road
        board = st.board
        for col_color in self.play_order:
            catan_color = self.colonist_to_catan[col_color]
            p_idx = st.color_to_index[catan_color]
            st.player_state[f"P{p_idx}_LONGEST_ROAD_LENGTH"] = board.road_lengths.get(catan_color, 0)
            st.player_state[f"P{p_idx}_HAS_ROAD"] = False

        if board.road_color is not None and board.road_length >= 5:
            p_idx = st.color_to_index[board.road_color]
            st.player_state[f"P{p_idx}_HAS_ROAD"] = True
            st.player_state[f"P{p_idx}_VICTORY_POINTS"] += 2
            st.player_state[f"P{p_idx}_ACTUAL_VICTORY_POINTS"] += 2

        # Largest army
        best_color_idx, best_count = None, 0
        for col_color in self.play_order:
            catan_color = self.colonist_to_catan[col_color]
            p_idx = st.color_to_index[catan_color]
            knights = st.player_state[f"P{p_idx}_PLAYED_KNIGHT"]
            st.player_state[f"P{p_idx}_HAS_ARMY"] = False
            if knights >= 3 and knights > best_count:
                best_count = knights
                best_color_idx = p_idx

        if best_color_idx is not None:
            st.player_state[f"P{best_color_idx}_HAS_ARMY"] = True
            st.player_state[f"P{best_color_idx}_VICTORY_POINTS"] += 2
            st.player_state[f"P{best_color_idx}_ACTUAL_VICTORY_POINTS"] += 2


@dataclass
class ReplayState:
    """Tracks mutable game state across events."""
    play_order: List[int] = field(default_factory=list)
    colonist_to_catan: Dict[int, Color] = field(default_factory=dict)

    # Index maps
    vertex_idx_to_xyz: Dict[int, Tuple] = field(default_factory=dict)
    edge_idx_to_xyz: Dict[int, Tuple] = field(default_factory=dict)
    vertex_owners: Dict[int, int] = field(default_factory=dict)

    # Per-player resources/dev cards
    resources: Dict[int, Dict[str, int]] = field(default_factory=dict)
    dev_cards: Dict[int, Dict[str, int]] = field(default_factory=dict)
    dev_cards_used: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # Game state
    robber_tile_index: int = -1
    bank_resources: Dict[str, int] = field(default_factory=dict)

    # Turn tracking
    turn_state: int = 0
    action_state: int = 0
    current_turn_player: int = -1
    is_setup_phase: bool = True
    dice1: int = 0
    dice2: int = 0
    turn_number: int = 0

    # VP tracking
    vp: Dict[int, int] = field(default_factory=dict)


def parse_game(filepath: str) -> List[Dict]:
    """Parse a single game file into training samples.

    Returns list of dicts, each with feature columns + metadata + labels.
    Returns empty list if the game is invalid (non-4-player, non-standard, etc.).
    """
    path = Path(filepath)
    game_id = path.stem

    with open(filepath, "r") as f:
        game_data = json.load(f)

    # Validate: 4 players, standard rules
    play_order = game_data["data"].get("playOrder", [])
    if len(play_order) != 4:
        return []
    settings = game_data["data"].get("gameSettings", {})
    if settings.get("victoryPointsToWin", 10) != 10:
        return []

    eh = game_data["data"]["eventHistory"]
    end_game = eh.get("endGameState", {})
    end_players = end_game.get("players", {})
    total_turns = end_game.get("totalTurnCount", 0)

    winner_color = None
    final_vps = {}
    for col_str, pdata in end_players.items():
        col_color = int(col_str)
        if pdata.get("winningPlayer"):
            winner_color = col_color
        final_vps[col_color] = sum(pdata.get("victoryPoints", {}).values())

    if winner_color is None:
        return []

    feature_keys = get_feature_ordering(num_players=4, map_type="BASE")

    # Initialize
    init_state = eh["initialState"]
    map_state = init_state["mapState"]

    colonist_to_catan = {}
    rstate = ReplayState()
    rstate.play_order = list(play_order)
    for i, col_color in enumerate(play_order):
        colonist_to_catan[col_color] = CATANATRON_COLORS[i]
        rstate.resources[col_color] = {r: 0 for r in RESOURCES}
        rstate.dev_cards[col_color] = {d: 0 for d in DEVELOPMENT_CARDS}
        rstate.dev_cards_used[col_color] = {d: 0 for d in DEVELOPMENT_CARDS}
        rstate.vp[col_color] = 0
    rstate.colonist_to_catan = colonist_to_catan

    # Build coordinate mapper
    tiles = adapt_tiles(map_state["tileHexStates"])
    port_edge_states = map_state.get("portEdgeStates", {})
    verts = adapt_vertices(map_state["tileCornerStates"], port_edge_states)
    edges_list = adapt_edges(map_state["tileEdgeStates"])
    hp = extract_harbor_pairs(port_edge_states)
    mapper = build_coordinate_mapper(tiles, verts, edges_list, harbor_pairs=hp)

    rstate.vertex_idx_to_xyz, rstate.edge_idx_to_xyz = build_index_maps(
        map_state["tileCornerStates"], map_state["tileEdgeStates"],
    )

    # Create incremental game
    igame = IncrementalGame(mapper, play_order, colonist_to_catan)

    # Initial state
    robber_s = init_state.get("mechanicRobberState", {})
    rstate.robber_tile_index = robber_s.get("locationTileIndex", -1)
    if rstate.robber_tile_index >= 0:
        igame.set_robber(rstate.robber_tile_index)

    bank = init_state.get("bankState", {}).get("resourceCards", {})
    for rid_str, count in bank.items():
        res = RESOURCE_ID_TO_STR.get(int(rid_str))
        if res:
            rstate.bank_resources[res] = count

    cs = init_state.get("currentState", {})
    rstate.turn_state = cs.get("turnState", 0)
    rstate.action_state = cs.get("actionState", 0)
    rstate.current_turn_player = cs.get("currentTurnPlayerColor", -1)
    rstate.is_setup_phase = rstate.turn_state == 0

    for col_str, ps in init_state.get("playerStates", {}).items():
        col_color = int(col_str)
        if col_color not in rstate.resources:
            continue
        cards = ps.get("resourceCards", {}).get("cards", [])
        hand = {r: 0 for r in RESOURCES}
        for card_id in cards:
            res = RESOURCE_ID_TO_STR.get(card_id)
            if res:
                hand[res] += 1
        rstate.resources[col_color] = hand

    # Process events
    samples = []
    events = eh.get("events", [])

    for event_idx, event in enumerate(events):
        is_sample = _process_event(rstate, igame, event)

        if not is_sample:
            continue

        # Prepare game state once, then extract features for all 4 perspectives
        try:
            igame.prepare_for_sampling(rstate)
        except Exception as e:
            logger.debug(f"prepare_for_sampling failed for {game_id} event {event_idx}: {e}")
            continue

        game = igame.game

        # Shared metadata for this sampling point
        meta = {
            "game_id": game_id,
            "event_index": event_idx,
            "turn_number": rstate.turn_number,
            "total_turns": total_turns,
            "game_progress": rstate.turn_number / total_turns if total_turns > 0 else 0.0,
            "acting_player_color": rstate.current_turn_player,
            "dice_roll": rstate.dice1 + rstate.dice2,
        }

        for col_color in play_order:
            catan_color = colonist_to_catan[col_color]

            try:
                sample = create_sample(game, catan_color)
            except Exception as e:
                logger.debug(f"create_sample failed for {game_id} event {event_idx}: {e}")
                continue

            row = {}
            for key in feature_keys:
                val = sample.get(key)
                if val is None:
                    row[f"F_{key}"] = 0.0
                elif isinstance(val, bool):
                    row[f"F_{key}"] = 1.0 if val else 0.0
                else:
                    row[f"F_{key}"] = float(val)

            row.update(meta)
            row["perspective_color"] = col_color
            row["winner"] = 1.0 if col_color == winner_color else 0.0
            row["final_vp"] = final_vps.get(col_color, 0)
            row["vp_at_snapshot"] = rstate.vp.get(col_color, 0)

            samples.append(row)

    return samples


def _process_event(rstate: ReplayState, igame: IncrementalGame, event: Dict) -> bool:
    """Process a single event. Returns True at sampling points."""
    sc = event.get("stateChange", {})
    if not sc:
        return False

    sample_point = False

    # 1. currentState
    cs = sc.get("currentState", {})
    if cs:
        new_ts = cs.get("turnState")
        if new_ts is not None:
            old_ts = rstate.turn_state
            rstate.turn_state = new_ts
            if new_ts >= 1:
                rstate.is_setup_phase = False
            if new_ts == 2 and old_ts != 2:
                sample_point = True
        new_as = cs.get("actionState")
        if new_as is not None:
            rstate.action_state = new_as
        new_tp = cs.get("currentTurnPlayerColor")
        if new_tp is not None:
            if not rstate.is_setup_phase and new_tp != rstate.current_turn_player:
                rstate.turn_number += 1
            rstate.current_turn_player = new_tp

    # 2. Buildings — applied to board incrementally
    map_sc = sc.get("mapState", {})
    for idx_str, corner in map_sc.get("tileCornerStates", {}).items():
        idx = int(idx_str)
        building_type = corner.get("buildingType")
        if building_type is None:
            continue
        owner = corner.get("owner")
        if owner is None:
            owner = rstate.vertex_owners.get(idx)
            if owner is None:
                continue
        rstate.vertex_owners[idx] = owner
        xyz = rstate.vertex_idx_to_xyz.get(idx)
        if xyz is None:
            continue
        if building_type == 2:
            igame.apply_city(xyz, owner)
        else:
            igame.apply_settlement(xyz, owner, rstate.is_setup_phase)

    for idx_str, edge in map_sc.get("tileEdgeStates", {}).items():
        owner = edge.get("owner")
        if owner is None or owner < 0:
            continue
        xyz = rstate.edge_idx_to_xyz.get(int(idx_str))
        if xyz is None:
            continue
        igame.apply_road(xyz, owner)

    # 3. Resources
    ps_sc = sc.get("playerStates", {})
    for col_str, ps_data in ps_sc.items():
        col_color = int(col_str)
        if col_color not in rstate.resources:
            continue
        rc = ps_data.get("resourceCards")
        if rc is None:
            continue
        cards = rc.get("cards")
        if cards is None:
            continue
        hand = {r: 0 for r in RESOURCES}
        for card_id in cards:
            res = RESOURCE_ID_TO_STR.get(card_id)
            if res:
                hand[res] += 1
        rstate.resources[col_color] = hand

    # 4. Dev cards
    dev_sc = sc.get("mechanicDevelopmentCardsState", {})
    for col_str, pdev in dev_sc.get("players", {}).items():
        col_color = int(col_str)
        if col_color not in rstate.dev_cards:
            continue
        dc = pdev.get("developmentCards")
        if dc is not None:
            hand = {d: 0 for d in DEVELOPMENT_CARDS}
            for card_id in dc.get("cards", []):
                if card_id == 10:
                    continue
                dev_str = DEVCARD_ID_TO_STR.get(card_id)
                if dev_str:
                    hand[dev_str] += 1
            rstate.dev_cards[col_color] = hand
        dcu = pdev.get("developmentCardsUsed")
        if dcu is not None:
            used = {d: 0 for d in DEVELOPMENT_CARDS}
            for card_id in dcu:
                dev_str = DEVCARD_ID_TO_STR.get(card_id)
                if dev_str:
                    used[dev_str] += 1
            rstate.dev_cards_used[col_color] = used

    # 5. Robber
    loc = sc.get("mechanicRobberState", {}).get("locationTileIndex")
    if loc is not None:
        rstate.robber_tile_index = loc
        igame.set_robber(loc)

    # 6. Bank
    for rid_str, count in sc.get("bankState", {}).get("resourceCards", {}).items():
        res = RESOURCE_ID_TO_STR.get(int(rid_str))
        if res:
            rstate.bank_resources[res] = count

    # 7. Dice
    dice_sc = sc.get("diceState", {})
    if "dice1" in dice_sc:
        rstate.dice1 = dice_sc["dice1"]
    if "dice2" in dice_sc:
        rstate.dice2 = dice_sc["dice2"]

    # 8. VP
    for col_str, ps_data in ps_sc.items():
        col_color = int(col_str)
        vps = ps_data.get("victoryPointsState")
        if vps is not None and col_color in rstate.vp:
            rstate.vp[col_color] = sum(vps.values())

    return sample_point
