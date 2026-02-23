"""Conversion between Catanatron Game and BitboardState.

game_to_bitboard(game) — create BitboardState from a Game (for init + verification)
"""

import numpy as np
from catanatron.models.enums import (
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT,
    SETTLEMENT, CITY,
)
from robottler.bitboard.masks import (
    NUM_NODES, NUM_EDGES, NUM_TILES, NUM_PORT_TYPES,
    NODE_BIT, NEIGHBOR_MASK,
    EDGE_TO_INDEX, INCIDENT_EDGES, EDGE_ENDPOINTS,
    RESOURCE_INDEX, PORT_TYPE_INDEX,
    build_tile_data, build_port_masks,
    popcount64, bitscan,
)
from catanatron.models.enums import ActionPrompt
from robottler.bitboard.state import (
    BitboardState, PS_VP, PS_ROADS_AVAIL, PS_SETTLE_AVAIL, PS_CITY_AVAIL,
    PS_HAS_ROAD, PS_HAS_ARMY, PS_HAS_ROLLED, PS_HAS_PLAYED_DEV,
    PS_ACTUAL_VP, PS_LONGEST_ROAD,
    PS_KNIGHT_START, PS_MONO_START, PS_YOP_START, PS_RB_START,
    PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
    PS_KNIGHT_HAND, PS_PLAYED_KNIGHT,
    PS_YOP_HAND, PS_PLAYED_YOP,
    PS_MONO_HAND, PS_PLAYED_MONO,
    PS_RB_HAND, PS_PLAYED_RB,
    PS_VP_HAND, PS_PLAYED_VP,
    PS_SIZE, PS_RESOURCE_START,
    DEV_NAME_TO_IDX,
    PROMPT_BUILD_INITIAL_SETTLEMENT, PROMPT_BUILD_INITIAL_ROAD,
    PROMPT_PLAY_TURN, PROMPT_DISCARD, PROMPT_MOVE_ROBBER,
    PROMPT_DECIDE_TRADE, PROMPT_DECIDE_ACCEPTEES,
)

# Map ActionPrompt enum → int PROMPT_* constant
_ACTION_PROMPT_TO_INT = {
    ActionPrompt.BUILD_INITIAL_SETTLEMENT: PROMPT_BUILD_INITIAL_SETTLEMENT,
    ActionPrompt.BUILD_INITIAL_ROAD: PROMPT_BUILD_INITIAL_ROAD,
    ActionPrompt.PLAY_TURN: PROMPT_PLAY_TURN,
    ActionPrompt.DISCARD: PROMPT_DISCARD,
    ActionPrompt.MOVE_ROBBER: PROMPT_MOVE_ROBBER,
    ActionPrompt.DECIDE_TRADE: PROMPT_DECIDE_TRADE,
    ActionPrompt.DECIDE_ACCEPTEES: PROMPT_DECIDE_ACCEPTEES,
}

# Mapping from Catanatron dev card string to (hand_idx, played_idx) in player_state
_DEV_HAND_MAP = {
    KNIGHT: PS_KNIGHT_HAND,
    YEAR_OF_PLENTY: PS_YOP_HAND,
    MONOPOLY: PS_MONO_HAND,
    ROAD_BUILDING: PS_RB_HAND,
    VICTORY_POINT: PS_VP_HAND,
}
_DEV_PLAYED_MAP = {
    KNIGHT: PS_PLAYED_KNIGHT,
    YEAR_OF_PLENTY: PS_PLAYED_YOP,
    MONOPOLY: PS_PLAYED_MONO,
    ROAD_BUILDING: PS_PLAYED_RB,
    VICTORY_POINT: PS_PLAYED_VP,
}
_DEV_START_MAP = {
    KNIGHT: PS_KNIGHT_START,
    MONOPOLY: PS_MONO_START,
    YEAR_OF_PLENTY: PS_YOP_START,
    ROAD_BUILDING: PS_RB_START,
}

_RESOURCE_PS_MAP = {
    WOOD: PS_WOOD,
    BRICK: PS_BRICK,
    SHEEP: PS_SHEEP,
    WHEAT: PS_WHEAT,
    ORE: PS_ORE,
}

# Dev card string → int8 ID for dev_deck array
DEV_CARD_ID = {
    KNIGHT: 0, YEAR_OF_PLENTY: 1, MONOPOLY: 2, ROAD_BUILDING: 3, VICTORY_POINT: 4,
}
DEV_ID_TO_NAME = {v: k for k, v in DEV_CARD_ID.items()}


def game_to_bitboard(game):
    """Convert a Catanatron Game to a BitboardState.

    This reads the full game state and constructs a BitboardState that
    mirrors it exactly. Used for initialization and verification.
    """
    state = game.state
    n = len(state.colors)
    bb = BitboardState(n)

    # Shared immutables
    bb.colors = state.colors
    bb.color_to_index = dict(state.color_to_index)
    bb.catan_map = state.board.map
    bb.discard_limit = state.discard_limit

    # Build map-specific data
    tr, tn, tc2i, ti2c = build_tile_data(state.board.map)
    bb.tile_resource = tr
    bb.tile_number = tn
    bb.tile_coord_to_id = tc2i
    bb.tile_id_to_coord = ti2c
    bb.port_nodes = build_port_masks(state.board.map)

    # Robber
    robber_coord = state.board.robber_coordinate
    if robber_coord in tc2i:
        bb.robber_tile = np.int8(tc2i[robber_coord])

    # Buildings
    for node_id, (color, btype) in state.board.buildings.items():
        pidx = state.color_to_index[color]
        if btype == SETTLEMENT:
            bb.settlement_bb[pidx] |= NODE_BIT[node_id]
        elif btype == CITY:
            bb.city_bb[pidx] |= NODE_BIT[node_id]

    # Roads
    seen_edges = set()
    for edge, color in state.board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical in seen_edges:
            continue
        seen_edges.add(canonical)
        eidx = EDGE_TO_INDEX[canonical]
        pidx = state.color_to_index[color]
        bb.set_edge(pidx, eidx)

    # Board buildable
    bb.board_buildable = np.uint64(0)
    for nid in state.board.board_buildable_ids:
        bb.board_buildable |= NODE_BIT[nid]

    # Bank
    for i in range(5):
        bb.bank[i] = np.int8(state.resource_freqdeck[i])

    # Dev deck
    # Catanatron draws with .pop() (from end of list). Our code reads from dev_deck[dev_deck_idx]
    # forward. Store remaining cards reversed so our draw order matches Catanatron's pop order.
    n_remaining = len(state.development_listdeck)
    bb.dev_deck_idx = 25 - n_remaining
    for i, card in enumerate(reversed(state.development_listdeck)):
        bb.dev_deck[bb.dev_deck_idx + i] = np.int8(DEV_CARD_ID[card])

    # Player state
    for color, pidx in state.color_to_index.items():
        key = f"P{pidx}"
        ps = bb.player_state[pidx]

        ps[PS_VP] = state.player_state[f"{key}_VICTORY_POINTS"]
        ps[PS_ROADS_AVAIL] = state.player_state[f"{key}_ROADS_AVAILABLE"]
        ps[PS_SETTLE_AVAIL] = state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
        ps[PS_CITY_AVAIL] = state.player_state[f"{key}_CITIES_AVAILABLE"]
        ps[PS_HAS_ROAD] = int(state.player_state[f"{key}_HAS_ROAD"])
        ps[PS_HAS_ARMY] = int(state.player_state[f"{key}_HAS_ARMY"])
        ps[PS_HAS_ROLLED] = int(state.player_state[f"{key}_HAS_ROLLED"])
        ps[PS_HAS_PLAYED_DEV] = int(state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])
        ps[PS_ACTUAL_VP] = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        ps[PS_LONGEST_ROAD] = state.player_state[f"{key}_LONGEST_ROAD_LENGTH"]

        # Dev card start-of-turn flags
        ps[PS_KNIGHT_START] = int(state.player_state[f"{key}_KNIGHT_OWNED_AT_START"])
        ps[PS_MONO_START] = int(state.player_state[f"{key}_MONOPOLY_OWNED_AT_START"])
        ps[PS_YOP_START] = int(state.player_state[f"{key}_YEAR_OF_PLENTY_OWNED_AT_START"])
        ps[PS_RB_START] = int(state.player_state[f"{key}_ROAD_BUILDING_OWNED_AT_START"])

        # Resources
        ps[PS_WOOD] = state.player_state[f"{key}_WOOD_IN_HAND"]
        ps[PS_BRICK] = state.player_state[f"{key}_BRICK_IN_HAND"]
        ps[PS_SHEEP] = state.player_state[f"{key}_SHEEP_IN_HAND"]
        ps[PS_WHEAT] = state.player_state[f"{key}_WHEAT_IN_HAND"]
        ps[PS_ORE] = state.player_state[f"{key}_ORE_IN_HAND"]

        # Dev cards
        ps[PS_KNIGHT_HAND] = state.player_state[f"{key}_KNIGHT_IN_HAND"]
        ps[PS_PLAYED_KNIGHT] = state.player_state[f"{key}_PLAYED_KNIGHT"]
        ps[PS_YOP_HAND] = state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
        ps[PS_PLAYED_YOP] = state.player_state[f"{key}_PLAYED_YEAR_OF_PLENTY"]
        ps[PS_MONO_HAND] = state.player_state[f"{key}_MONOPOLY_IN_HAND"]
        ps[PS_PLAYED_MONO] = state.player_state[f"{key}_PLAYED_MONOPOLY"]
        ps[PS_RB_HAND] = state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"]
        ps[PS_PLAYED_RB] = state.player_state[f"{key}_PLAYED_ROAD_BUILDING"]
        ps[PS_VP_HAND] = state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
        ps[PS_PLAYED_VP] = state.player_state[f"{key}_PLAYED_VICTORY_POINT"]

    # Game machine
    bb.current_player_idx = state.current_player_index
    bb.current_turn_idx = state.current_turn_index
    bb.num_turns = state.num_turns
    bb.is_initial_build_phase = state.is_initial_build_phase
    bb.is_discarding = state.is_discarding
    bb.is_moving_knight = state.is_moving_knight
    bb.is_road_building = state.is_road_building
    bb.free_roads = state.free_roads_available
    bb.is_resolving_trade = state.is_resolving_trade
    bb.friendly_robber = getattr(state, "friendly_robber", False)
    bb.current_prompt = _ACTION_PROMPT_TO_INT[state.current_prompt]

    # Rebuild connected components from scratch
    _rebuild_connected_components(bb)

    # Rebuild port access
    for pidx in range(n):
        for node in bitscan(bb.settlement_bb[pidx] | bb.city_bb[pidx]):
            bb._update_port_access(pidx, node)

    # Road holder and army holder
    _rebuild_road_holder(bb)
    _rebuild_army_holder(bb, state)

    # Zobrist hash
    from robottler.zobrist import ZobristTracker
    tracker = ZobristTracker()
    tracker._num_players = n
    tracker._color_to_idx = dict(state.color_to_index)
    bb.zobrist_hash = tracker.compute_full(game)

    return bb


def _rebuild_connected_components(bb):
    """Rebuild connected components from road and building bitmasks."""
    for pidx in range(bb.num_players):
        bb.component_ids[pidx, :] = -1
        bb.reachable_bb[pidx] = np.uint64(0)
        bb.num_components[pidx] = 0

        # Find all nodes connected by this player's roads or buildings
        all_nodes = set()
        for eidx in range(NUM_EDGES):
            if bb.has_edge(pidx, eidx):
                a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                all_nodes.add(a)
                all_nodes.add(b)
        for node in bitscan(bb.settlement_bb[pidx] | bb.city_bb[pidx]):
            all_nodes.add(node)

        if not all_nodes:
            continue

        visited = set()
        comp_id = 0

        for start in sorted(all_nodes):
            if start in visited:
                continue

            # BFS
            queue = [start]
            component_nodes = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component_nodes.append(node)

                if bb.is_enemy_node(node, pidx):
                    continue  # don't expand through enemy nodes

                for eidx in INCIDENT_EDGES[node]:
                    if not bb.has_edge(pidx, eidx):
                        continue
                    a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                    neighbor = b if a == node else a
                    if neighbor not in visited:
                        queue.append(neighbor)

            for node in component_nodes:
                if not bb.is_enemy_node(node, pidx):
                    bb.component_ids[pidx, node] = comp_id
                    bb.reachable_bb[pidx] |= NODE_BIT[node]
            comp_id += 1

        bb.num_components[pidx] = comp_id

        # Compute longest road
        bb._recompute_longest_road(pidx)


def _rebuild_road_holder(bb):
    """Set road_holder from road_lengths and player_state."""
    bb.road_holder = -1
    bb.road_holder_length = 0
    for p in range(bb.num_players):
        length = int(bb.road_lengths[p])
        if length >= 5 and length > bb.road_holder_length:
            bb.road_holder = p
            bb.road_holder_length = length


def _rebuild_army_holder(bb, game_state):
    """Set army_holder from player_state."""
    bb.army_holder = -1
    bb.army_holder_size = 0
    for p in range(bb.num_players):
        if bb.player_state[p, PS_HAS_ARMY]:
            bb.army_holder = p
            bb.army_holder_size = int(bb.player_state[p, PS_PLAYED_KNIGHT])


def compare_states(game, bb):
    """Compare a Catanatron Game state with a BitboardState.

    Returns a list of differences (empty if they match).
    """
    diffs = []
    state = game.state

    # Buildings
    for node_id, (color, btype) in state.board.buildings.items():
        pidx = state.color_to_index[color]
        if btype == SETTLEMENT:
            if not bb.has_settlement(pidx, node_id):
                diffs.append(f"Missing settlement: P{pidx} at node {node_id}")
        elif btype == CITY:
            if not bb.has_city(pidx, node_id):
                diffs.append(f"Missing city: P{pidx} at node {node_id}")

    # Check no extra buildings in bitboard
    for pidx in range(bb.num_players):
        for node in bitscan(bb.settlement_bb[pidx]):
            if node not in state.board.buildings:
                diffs.append(f"Extra settlement in bb: P{pidx} at node {node}")
            elif state.board.buildings[node][0] != bb.colors[pidx]:
                diffs.append(f"Wrong settlement owner: P{pidx} at node {node}")
        for node in bitscan(bb.city_bb[pidx]):
            if node not in state.board.buildings:
                diffs.append(f"Extra city in bb: P{pidx} at node {node}")

    # Roads
    seen_edges = set()
    for edge, color in state.board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical in seen_edges:
            continue
        seen_edges.add(canonical)
        eidx = EDGE_TO_INDEX[canonical]
        pidx = state.color_to_index[color]
        if not bb.has_edge(pidx, eidx):
            diffs.append(f"Missing road: P{pidx} at edge {canonical} (idx {eidx})")

    # Board buildable
    ref_buildable = np.uint64(0)
    for nid in state.board.board_buildable_ids:
        ref_buildable |= NODE_BIT[nid]
    if bb.board_buildable != ref_buildable:
        diffs.append(f"board_buildable mismatch: bb={bb.board_buildable:#x} ref={ref_buildable:#x}")

    # Player state
    for color, pidx in state.color_to_index.items():
        key = f"P{pidx}"

        checks = [
            ("VP", PS_VP, f"{key}_VICTORY_POINTS"),
            ("ROADS_AVAIL", PS_ROADS_AVAIL, f"{key}_ROADS_AVAILABLE"),
            ("SETTLE_AVAIL", PS_SETTLE_AVAIL, f"{key}_SETTLEMENTS_AVAILABLE"),
            ("CITY_AVAIL", PS_CITY_AVAIL, f"{key}_CITIES_AVAILABLE"),
            ("HAS_ROLLED", PS_HAS_ROLLED, f"{key}_HAS_ROLLED"),
            ("HAS_PLAYED_DEV", PS_HAS_PLAYED_DEV, f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"),
            ("ACTUAL_VP", PS_ACTUAL_VP, f"{key}_ACTUAL_VICTORY_POINTS"),
            ("WOOD", PS_WOOD, f"{key}_WOOD_IN_HAND"),
            ("BRICK", PS_BRICK, f"{key}_BRICK_IN_HAND"),
            ("SHEEP", PS_SHEEP, f"{key}_SHEEP_IN_HAND"),
            ("WHEAT", PS_WHEAT, f"{key}_WHEAT_IN_HAND"),
            ("ORE", PS_ORE, f"{key}_ORE_IN_HAND"),
            ("KNIGHT_HAND", PS_KNIGHT_HAND, f"{key}_KNIGHT_IN_HAND"),
            ("PLAYED_KNIGHT", PS_PLAYED_KNIGHT, f"{key}_PLAYED_KNIGHT"),
            ("YOP_HAND", PS_YOP_HAND, f"{key}_YEAR_OF_PLENTY_IN_HAND"),
            ("PLAYED_YOP", PS_PLAYED_YOP, f"{key}_PLAYED_YEAR_OF_PLENTY"),
            ("MONO_HAND", PS_MONO_HAND, f"{key}_MONOPOLY_IN_HAND"),
            ("PLAYED_MONO", PS_PLAYED_MONO, f"{key}_PLAYED_MONOPOLY"),
            ("RB_HAND", PS_RB_HAND, f"{key}_ROAD_BUILDING_IN_HAND"),
            ("PLAYED_RB", PS_PLAYED_RB, f"{key}_PLAYED_ROAD_BUILDING"),
            ("VP_HAND", PS_VP_HAND, f"{key}_VICTORY_POINT_IN_HAND"),
            ("PLAYED_VP", PS_PLAYED_VP, f"{key}_PLAYED_VICTORY_POINT"),
        ]
        for name, ps_idx, state_key in checks:
            bb_val = int(bb.player_state[pidx, ps_idx])
            ref_val = int(state.player_state[state_key])
            if bb_val != ref_val:
                diffs.append(f"P{pidx} {name}: bb={bb_val} ref={ref_val}")

    # Bank
    for i in range(5):
        if int(bb.bank[i]) != state.resource_freqdeck[i]:
            rname = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE'][i]
            diffs.append(f"Bank {rname}: bb={int(bb.bank[i])} ref={state.resource_freqdeck[i]}")

    # Game machine
    if bb.current_player_idx != state.current_player_index:
        diffs.append(f"current_player_idx: bb={bb.current_player_idx} ref={state.current_player_index}")
    if bb.current_turn_idx != state.current_turn_index:
        diffs.append(f"current_turn_idx: bb={bb.current_turn_idx} ref={state.current_turn_index}")
    if bb.num_turns != state.num_turns:
        diffs.append(f"num_turns: bb={bb.num_turns} ref={state.num_turns}")
    if bb.is_initial_build_phase != state.is_initial_build_phase:
        diffs.append(f"is_initial_build_phase: bb={bb.is_initial_build_phase} ref={state.is_initial_build_phase}")
    if bb.is_discarding != state.is_discarding:
        diffs.append(f"is_discarding: bb={bb.is_discarding} ref={state.is_discarding}")
    if bb.is_moving_knight != state.is_moving_knight:
        diffs.append(f"is_moving_knight: bb={bb.is_moving_knight} ref={state.is_moving_knight}")
    if bb.is_road_building != state.is_road_building:
        diffs.append(f"is_road_building: bb={bb.is_road_building} ref={state.is_road_building}")
    if bb.free_roads != state.free_roads_available:
        diffs.append(f"free_roads: bb={bb.free_roads} ref={state.free_roads_available}")

    ref_prompt = _ACTION_PROMPT_TO_INT[state.current_prompt]
    if bb.current_prompt != ref_prompt:
        diffs.append(f"current_prompt: bb={bb.current_prompt} ref={ref_prompt}")

    # Dev deck remaining
    bb_remaining = 25 - bb.dev_deck_idx
    ref_remaining = len(state.development_listdeck)
    if bb_remaining != ref_remaining:
        diffs.append(f"dev_deck_remaining: bb={bb_remaining} ref={ref_remaining}")

    # Robber
    robber_coord = state.board.robber_coordinate
    if robber_coord in bb.tile_coord_to_id:
        ref_robber = bb.tile_coord_to_id[robber_coord]
        if bb.robber_tile != ref_robber:
            diffs.append(f"robber_tile: bb={bb.robber_tile} ref={ref_robber}")

    return diffs
