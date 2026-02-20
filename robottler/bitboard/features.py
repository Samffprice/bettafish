"""Feature extraction from BitboardState for neural network evaluation.

Produces the same feature dict as catanatron.features.create_sample(),
but reads from BitboardState instead of Game objects.
"""

import functools
import numpy as np
from collections import Counter

from catanatron.models.enums import (
    RESOURCES, DEVELOPMENT_CARDS, SETTLEMENT, CITY, ROAD,
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    VICTORY_POINT, ActionType,
)
from catanatron.models.board import get_edges
from catanatron.models.map import number_probability

from robottler.bitboard.masks import (
    NUM_NODES, NUM_EDGES, NUM_TILES, NUM_PORT_TYPES,
    NODE_BIT, ADJACENT_NODES, EDGE_LIST, EDGE_TO_INDEX,
    INCIDENT_EDGES, EDGE_ENDPOINTS, TILE_NODES,
    RESOURCE_INDEX, RESOURCE_NAMES, DICE_PROBA,
    popcount64, bitscan, HAS_CYTHON,
    NODE_NEIGHBOR_NODES, NODE_NEIGHBOR_EDGES, NODE_NEIGHBOR_COUNT,
)
from robottler.bitboard.state import (
    PS_VP, PS_ROADS_AVAIL, PS_SETTLE_AVAIL, PS_CITY_AVAIL,
    PS_HAS_ROAD, PS_HAS_ARMY, PS_HAS_ROLLED, PS_HAS_PLAYED_DEV,
    PS_ACTUAL_VP, PS_LONGEST_ROAD,
    PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
    PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
    PS_PLAYED_KNIGHT, PS_PLAYED_YOP, PS_PLAYED_MONO, PS_PLAYED_RB,
    PS_RESOURCE_START, PS_RESOURCE_END,
    PROMPT_MOVE_ROBBER, PROMPT_DISCARD,
)

# Dev card hand indices mapping
_DEV_HAND_IDX = {
    "KNIGHT": PS_KNIGHT_HAND,
    "YEAR_OF_PLENTY": PS_YOP_HAND,
    "MONOPOLY": PS_MONO_HAND,
    "ROAD_BUILDING": PS_RB_HAND,
    "VICTORY_POINT": PS_VP_HAND,
}
_DEV_PLAYED_IDX = {
    "KNIGHT": PS_PLAYED_KNIGHT,
    "YEAR_OF_PLENTY": PS_PLAYED_YOP,
    "MONOPOLY": PS_PLAYED_MONO,
    "ROAD_BUILDING": PS_PLAYED_RB,
}

# Resource name to PS index
_RES_PS = {
    WOOD: PS_WOOD, BRICK: PS_BRICK, SHEEP: PS_SHEEP,
    WHEAT: PS_WHEAT, ORE: PS_ORE,
}


def _iter_players(state, p0_idx):
    """Iterate players in turn order starting from p0_idx."""
    n = state.num_players
    for i in range(n):
        pidx = (p0_idx + i) % n
        yield i, pidx


# ---------------------------------------------------------------------------
# Cached static feature templates
# ---------------------------------------------------------------------------

@functools.lru_cache(4)
def _graph_template(num_players, catan_map):
    """Pre-build the all-False graph features dict (cached per player count + map)."""
    features = {}
    for i in range(num_players):
        for node_id in range(len(catan_map.land_nodes)):
            features[f"NODE{node_id}_P{i}_{SETTLEMENT}"] = False
            features[f"NODE{node_id}_P{i}_{CITY}"] = False
        for edge in get_edges(catan_map.land_nodes):
            features[f"EDGE{edge}_P{i}_{ROAD}"] = False
    return features


@functools.lru_cache(1)
def _static_port_features(catan_map):
    """Port features are static per map. Cache them."""
    features = {}
    for port_id, port in catan_map.ports_by_id.items():
        for resource in RESOURCES:
            features[f"PORT{port_id}_IS_{resource}"] = port.resource == resource
        features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port.resource is None
    return features


@functools.lru_cache(38)  # 19 tiles × 2 robber positions (approx)
def _static_tile_features(catan_map, robber_coordinate):
    """Tile features are static per map + robber position. Cache them."""
    features = {}
    for tile_id, tile in catan_map.tiles_by_id.items():
        for resource in RESOURCES:
            features[f"TILE{tile_id}_IS_{resource}"] = tile.resource == resource
        features[f"TILE{tile_id}_IS_DESERT"] = tile.resource is None
        features[f"TILE{tile_id}_PROBA"] = (
            0 if tile.resource is None else number_probability(tile.number)
        )
        features[f"TILE{tile_id}_HAS_ROBBER"] = (
            catan_map.tiles[robber_coordinate] == tile
        )
    return features


# Precompute per-node neighbor list: _NODE_NEIGHBORS[node] = tuple of (neighbor, edge_word, edge_bitmask)
# where edge_word is 0 or 1 (which uint64 word) and edge_bitmask is the bit within that word.
_NODE_NEIGHBORS = [[] for _ in range(NUM_NODES)]
for _eidx in range(NUM_EDGES):
    _a, _b = int(EDGE_ENDPOINTS[_eidx, 0]), int(EDGE_ENDPOINTS[_eidx, 1])
    _eword = _eidx >> 6  # // 64
    _ebit = 1 << (_eidx & 63)  # % 64, plain int
    _NODE_NEIGHBORS[_a].append((_b, _eword, _ebit))
    _NODE_NEIGHBORS[_b].append((_a, _eword, _ebit))
_NODE_NEIGHBORS = tuple(tuple(n) for n in _NODE_NEIGHBORS)

# Plain-int versions of NODE_BIT for BFS (avoid np.uint64 overhead)
_INODE_BIT = tuple(1 << i for i in range(NUM_NODES))


def bb_create_sample(state, p0_color):
    """Extract features from BitboardState, matching create_sample(game, p0_color)."""
    p0_idx = state.color_to_index[p0_color]
    features = {}

    _player_features(state, p0_idx, features)
    _resource_hand_features(state, p0_idx, features)
    _production_features(state, p0_idx, features, consider_robber=True, prefix="EFFECTIVE_")
    _production_features(state, p0_idx, features, consider_robber=False, prefix="TOTAL_")
    _reachability_features(state, p0_idx, features)

    # Tile/port: use cached static features
    robber_coord = state.tile_id_to_coord[state.robber_tile]
    features.update(_static_tile_features(state.catan_map, robber_coord))
    features.update(_static_port_features(state.catan_map))

    _graph_features(state, p0_idx, features)
    _game_features(state, features)

    return features


def bb_create_sample_vector(state, p0_color, feature_names):
    """Extract features as a vector in the given feature order."""
    sample = bb_create_sample(state, p0_color)
    return [float(sample.get(f, 0.0)) for f in feature_names]


# ---------------------------------------------------------------------------
# Player features
# ---------------------------------------------------------------------------

def _player_features(state, p0_idx, features):
    for i, pidx in _iter_players(state, p0_idx):
        ps = state.player_state[pidx]
        if pidx == p0_idx:
            features["P0_ACTUAL_VPS"] = int(ps[PS_ACTUAL_VP])
        features[f"P{i}_PUBLIC_VPS"] = int(ps[PS_VP])
        features[f"P{i}_HAS_ARMY"] = bool(ps[PS_HAS_ARMY])
        features[f"P{i}_HAS_ROAD"] = bool(ps[PS_HAS_ROAD])
        features[f"P{i}_ROADS_LEFT"] = int(ps[PS_ROADS_AVAIL])
        features[f"P{i}_SETTLEMENTS_LEFT"] = int(ps[PS_SETTLE_AVAIL])
        features[f"P{i}_CITIES_LEFT"] = int(ps[PS_CITY_AVAIL])
        features[f"P{i}_HAS_ROLLED"] = bool(ps[PS_HAS_ROLLED])
        features[f"P{i}_LONGEST_ROAD_LENGTH"] = int(ps[PS_LONGEST_ROAD])


# ---------------------------------------------------------------------------
# Resource hand features
# ---------------------------------------------------------------------------

def _resource_hand_features(state, p0_idx, features):
    for i, pidx in _iter_players(state, p0_idx):
        ps = state.player_state[pidx]
        if pidx == p0_idx:
            for resource in RESOURCES:
                features[f"P0_{resource}_IN_HAND"] = int(ps[_RES_PS[resource]])
            for card in DEVELOPMENT_CARDS:
                features[f"P0_{card}_IN_HAND"] = int(ps[_DEV_HAND_IDX[card]])
            features["P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = bool(ps[PS_HAS_PLAYED_DEV])

        for card in DEVELOPMENT_CARDS:
            if card == VICTORY_POINT:
                continue
            features[f"P{i}_{card}_PLAYED"] = int(ps[_DEV_PLAYED_IDX[card]])

        total_res = int(np.sum(ps[PS_RESOURCE_START:PS_RESOURCE_END]))
        features[f"P{i}_NUM_RESOURCES_IN_HAND"] = total_res

        total_dev = 0
        for card in DEVELOPMENT_CARDS:
            total_dev += int(ps[_DEV_HAND_IDX[card]])
        features[f"P{i}_NUM_DEVS_IN_HAND"] = total_dev


# ---------------------------------------------------------------------------
# Production features
# ---------------------------------------------------------------------------

def _production_features(state, p0_idx, features, consider_robber, prefix):
    """Compute resource production per player.

    Matches Catanatron: buildings ON the robber tile get zero production.
    """
    robbed_nodes_mask = np.uint64(0)
    if consider_robber:
        robbed_nodes_mask = TILE_NODES[state.robber_tile]

    node_production = state.catan_map.node_production

    for resource in RESOURCES:
        for i, pidx in _iter_players(state, p0_idx):
            production = 0.0
            for node in bitscan(state.settlement_bb[pidx]):
                if consider_robber and (NODE_BIT[node] & robbed_nodes_mask):
                    continue
                production += node_production.get(node, {}).get(resource, 0.0)
            for node in bitscan(state.city_bb[pidx]):
                if consider_robber and (NODE_BIT[node] & robbed_nodes_mask):
                    continue
                production += 2 * node_production.get(node, {}).get(resource, 0.0)
            features[f"{prefix}P{i}_{resource}_PRODUCTION"] = production


# ---------------------------------------------------------------------------
# Reachability features (optimized with bitmask BFS)
# ---------------------------------------------------------------------------

def _reachability_features(state, p0_idx, features, max_level=2):
    """Road reachability production using bitmask-based BFS (plain-int fast path)."""
    board_buildable = int(state.board_buildable)
    node_production = state.catan_map.node_production

    for i, pidx in _iter_players(state, p0_idx):
        buildings_mask = int(state.settlement_bb[pidx] | state.city_bb[pidx])
        zero_nodes_mask = int(state.reachable_bb[pidx]) | buildings_mask
        owned_or_buildable = buildings_mask | board_buildable

        # Level 0
        level_0_nodes = zero_nodes_mask & owned_or_buildable
        production_0 = _count_production_int(node_production, level_0_nodes)
        for resource in RESOURCES:
            features[f"P{i}_0_ROAD_REACHABLE_{resource}"] = production_0.get(resource, 0.0)

        # Enemy masks (plain int for speed)
        enemy_nodes = 0
        opp_road = [0, 0]
        for opp in range(state.num_players):
            if opp == pidx:
                continue
            enemy_nodes |= int(state.settlement_bb[opp]) | int(state.city_bb[opp])
            opp_road[0] |= int(state.road_bb[opp, 0])
            opp_road[1] |= int(state.road_bb[opp, 1])
        opp_road_0, opp_road_1 = opp_road

        # BFS expansion using precomputed neighbor lists
        last_layer = zero_nodes_mask
        for level in range(1, max_level + 1):
            next_layer = last_layer
            frontier = last_layer
            while frontier:
                bit = frontier & (-frontier)
                node = bit.bit_length() - 1
                frontier ^= bit
                if bit & enemy_nodes:
                    continue
                for neighbor, eword, ebit in _NODE_NEIGHBORS[node]:
                    if eword == 0:
                        if opp_road_0 & ebit:
                            continue
                    else:
                        if opp_road_1 & ebit:
                            continue
                    next_layer |= _INODE_BIT[neighbor]

            level_nodes = next_layer & owned_or_buildable
            production = _count_production_int(node_production, level_nodes)
            for resource in RESOURCES:
                features[f"P{i}_{level}_ROAD_REACHABLE_{resource}"] = production.get(resource, 0.0)
            last_layer = next_layer


def _count_production_from_mask(state, node_mask):
    """Sum production for all nodes in mask, by resource."""
    production = {}
    node_production = state.catan_map.node_production
    for node in bitscan(node_mask):
        node_prod = node_production.get(node)
        if node_prod:
            for resource, prob in node_prod.items():
                production[resource] = production.get(resource, 0.0) + prob
    return production


def _count_production_int(node_production, node_mask_int):
    """Sum production for all nodes in mask (plain int bitscan)."""
    production = {}
    mask = node_mask_int
    while mask:
        bit = mask & (-mask)
        node = bit.bit_length() - 1
        mask ^= bit
        node_prod = node_production.get(node)
        if node_prod:
            for resource, prob in node_prod.items():
                production[resource] = production.get(resource, 0.0) + prob
    return production


# ---------------------------------------------------------------------------
# Graph features (optimized: template copy + sparse updates)
# ---------------------------------------------------------------------------

def _graph_features(state, p0_idx, features):
    """Node/edge ownership features — sparse, no template copy.

    Missing keys default to False/0.0 via bb_create_sample_vector's .get(f, 0.0).
    """
    for i, pidx in _iter_players(state, p0_idx):
        for node in bitscan(state.settlement_bb[pidx]):
            features[f"NODE{node}_P{i}_{SETTLEMENT}"] = True
        for node in bitscan(state.city_bb[pidx]):
            features[f"NODE{node}_P{i}_{CITY}"] = True
        # Bitscan road words instead of checking all 72 edges
        for eidx in bitscan(state.road_bb[pidx, 0]):
            features[f"EDGE{EDGE_LIST[eidx]}_P{i}_{ROAD}"] = True
        for eidx in bitscan(state.road_bb[pidx, 1]):
            features[f"EDGE{EDGE_LIST[eidx + 64]}_P{i}_{ROAD}"] = True


# ---------------------------------------------------------------------------
# Game features
# ---------------------------------------------------------------------------

def _game_features(state, features):
    features["BANK_DEV_CARDS"] = 25 - state.dev_deck_idx
    for ri, resource in enumerate(RESOURCES):
        features[f"BANK_{resource}"] = int(state.bank[ri])
    features["IS_MOVING_ROBBER"] = (state.current_prompt == PROMPT_MOVE_ROBBER)
    features["IS_DISCARDING"] = (state.current_prompt == PROMPT_DISCARD)


# ---------------------------------------------------------------------------
# Direct vector extraction (Optimization 1): bypasses dict construction
# ---------------------------------------------------------------------------

# PS indices for dev hand (ordered: KNIGHT, YOP, MONO, RB, VP)
_PS_DEV_HAND = (PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND)
# PS indices for dev played (ordered: KNIGHT, YOP, MONO, RB)
_PS_DEV_PLAYED = (PS_PLAYED_KNIGHT, PS_PLAYED_YOP, PS_PLAYED_MONO, PS_PLAYED_RB)
# Dev card names in order (for feature name generation)
_DEV_NAMES = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
_DEV_PLAYED_NAMES = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING")


def _idx_arr(fim, n, template):
    """Build int32 array of feature indices for n player slots."""
    return np.array([fim.get(template.format(i), -1) for i in range(n)],
                    dtype=np.int32)


def _build_node_prod(catan_map):
    """Build [54, 5] node production array from a CatanMap."""
    node_prod = np.zeros((NUM_NODES, 5), dtype=np.float64)
    for node_id, counter in catan_map.node_production.items():
        for resource, prob in counter.items():
            node_prod[node_id, RESOURCE_INDEX[resource]] = prob
    return node_prod


class FeatureIndexer:
    """Precomputes integer indices for direct vector fill, bypassing dict construction.

    Built once per (checkpoint, map) pair. Maps feature names to integer positions
    in the neural net input vector so bb_fill_feature_vector() can write values
    directly to a pre-allocated numpy array.
    """

    __slots__ = [
        'num_model_players', 'node_prod', '_catan_map',
        # P0-only
        'p0_actual_vps', 'p0_has_played_dev', 'p0_res_hand', 'p0_dev_hand',
        # Per-slot
        'public_vps', 'has_army', 'has_road', 'roads_left',
        'settlements_left', 'cities_left', 'has_rolled', 'longest_road',
        'num_resources', 'num_devs', 'dev_played',
        # Production
        'eff_prod', 'total_prod',
        # Reachability
        'reach',
        # Game
        'bank_dev_cards', 'bank_res', 'is_moving_robber', 'is_discarding',
    ]

    def __init__(self, feature_index_map, catan_map):
        fim = feature_index_map

        # Detect number of player slots in model
        self.num_model_players = 0
        for i in range(8):
            if f"P{i}_PUBLIC_VPS" in fim:
                self.num_model_players = i + 1
        N = self.num_model_players

        # --- Precompute node production [54, 5] (map-specific) ---
        self._catan_map = catan_map
        self.node_prod = _build_node_prod(catan_map)

        # --- P0-only features ---
        self.p0_actual_vps = fim.get("P0_ACTUAL_VPS", -1)
        self.p0_has_played_dev = fim.get(
            "P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", -1)

        # P0 resource hand: [5] indices
        self.p0_res_hand = np.array([
            fim.get(f"P0_{RESOURCE_NAMES[ri]}_IN_HAND", -1)
            for ri in range(5)
        ], dtype=np.int32)

        # P0 dev hand: [5] indices
        self.p0_dev_hand = np.array([
            fim.get(f"P0_{d}_IN_HAND", -1) for d in _DEV_NAMES
        ], dtype=np.int32)

        # --- Per-player-slot features ---
        self.public_vps = _idx_arr(fim, N, "P{}_PUBLIC_VPS")
        self.has_army = _idx_arr(fim, N, "P{}_HAS_ARMY")
        self.has_road = _idx_arr(fim, N, "P{}_HAS_ROAD")
        self.roads_left = _idx_arr(fim, N, "P{}_ROADS_LEFT")
        self.settlements_left = _idx_arr(fim, N, "P{}_SETTLEMENTS_LEFT")
        self.cities_left = _idx_arr(fim, N, "P{}_CITIES_LEFT")
        self.has_rolled = _idx_arr(fim, N, "P{}_HAS_ROLLED")
        self.longest_road = _idx_arr(fim, N, "P{}_LONGEST_ROAD_LENGTH")
        self.num_resources = _idx_arr(fim, N, "P{}_NUM_RESOURCES_IN_HAND")
        self.num_devs = _idx_arr(fim, N, "P{}_NUM_DEVS_IN_HAND")

        # Dev played: [N, 4] (KNIGHT, YOP, MONO, RB — no VP)
        self.dev_played = np.full((N, 4), -1, dtype=np.int32)
        for slot in range(N):
            for di, d in enumerate(_DEV_PLAYED_NAMES):
                self.dev_played[slot, di] = fim.get(f"P{slot}_{d}_PLAYED", -1)

        # Production: [N, 5] for effective and total
        self.eff_prod = np.full((N, 5), -1, dtype=np.int32)
        self.total_prod = np.full((N, 5), -1, dtype=np.int32)
        for slot in range(N):
            for ri in range(5):
                r = RESOURCE_NAMES[ri]
                self.eff_prod[slot, ri] = fim.get(
                    f"EFFECTIVE_P{slot}_{r}_PRODUCTION", -1)
                self.total_prod[slot, ri] = fim.get(
                    f"TOTAL_P{slot}_{r}_PRODUCTION", -1)

        # Reachability: [N, 3, 5]
        self.reach = np.full((N, 3, 5), -1, dtype=np.int32)
        for slot in range(N):
            for level in range(3):
                for ri in range(5):
                    r = RESOURCE_NAMES[ri]
                    self.reach[slot, level, ri] = fim.get(
                        f"P{slot}_{level}_ROAD_REACHABLE_{r}", -1)

        # Game features
        self.bank_dev_cards = fim.get("BANK_DEV_CARDS", -1)
        self.bank_res = np.array([
            fim.get(f"BANK_{RESOURCE_NAMES[ri]}", -1) for ri in range(5)
        ], dtype=np.int32)
        self.is_moving_robber = fim.get("IS_MOVING_ROBBER", -1)
        self.is_discarding = fim.get("IS_DISCARDING", -1)

    def update_map(self, catan_map):
        """Update node_prod for a new game map. Index arrays are unchanged."""
        if catan_map is not self._catan_map:
            self._catan_map = catan_map
            self.node_prod = _build_node_prod(catan_map)


def bb_fill_feature_vector(state, p0_color, vec, fi):
    """Fill feature vector directly from BitboardState, bypassing dict construction.

    Args:
        state: BitboardState
        p0_color: Color of the player from whose perspective features are extracted
        vec: Pre-zeroed numpy array of length n_features (float32 or float64)
        fi: FeatureIndexer instance
    """
    p0_idx = state.color_to_index[p0_color]
    n_game = state.num_players
    n_slots = min(n_game, fi.num_model_players)

    # --- Player features ---
    for slot in range(n_slots):
        pidx = (p0_idx + slot) % n_game
        ps = state.player_state[pidx]

        idx = int(fi.public_vps[slot])
        if idx >= 0: vec[idx] = float(ps[PS_VP])
        idx = int(fi.has_army[slot])
        if idx >= 0: vec[idx] = float(ps[PS_HAS_ARMY])
        idx = int(fi.has_road[slot])
        if idx >= 0: vec[idx] = float(ps[PS_HAS_ROAD])
        idx = int(fi.roads_left[slot])
        if idx >= 0: vec[idx] = float(ps[PS_ROADS_AVAIL])
        idx = int(fi.settlements_left[slot])
        if idx >= 0: vec[idx] = float(ps[PS_SETTLE_AVAIL])
        idx = int(fi.cities_left[slot])
        if idx >= 0: vec[idx] = float(ps[PS_CITY_AVAIL])
        idx = int(fi.has_rolled[slot])
        if idx >= 0: vec[idx] = float(ps[PS_HAS_ROLLED])
        idx = int(fi.longest_road[slot])
        if idx >= 0: vec[idx] = float(ps[PS_LONGEST_ROAD])

    # P0-only features
    p0_ps = state.player_state[p0_idx]
    idx = fi.p0_actual_vps
    if idx >= 0: vec[idx] = float(p0_ps[PS_ACTUAL_VP])

    # --- Resource/dev hand (P0 only) ---
    for ri in range(5):
        idx = int(fi.p0_res_hand[ri])
        if idx >= 0: vec[idx] = float(p0_ps[PS_RESOURCE_START + ri])

    for di in range(5):
        idx = int(fi.p0_dev_hand[di])
        if idx >= 0: vec[idx] = float(p0_ps[_PS_DEV_HAND[di]])

    idx = fi.p0_has_played_dev
    if idx >= 0: vec[idx] = float(p0_ps[PS_HAS_PLAYED_DEV])

    # --- Dev played + num resources/devs (all players) ---
    for slot in range(n_slots):
        pidx = (p0_idx + slot) % n_game
        ps = state.player_state[pidx]

        for di in range(4):
            idx = int(fi.dev_played[slot, di])
            if idx >= 0: vec[idx] = float(ps[_PS_DEV_PLAYED[di]])

        idx = int(fi.num_resources[slot])
        if idx >= 0:
            vec[idx] = float(int(np.sum(ps[PS_RESOURCE_START:PS_RESOURCE_END])))

        idx = int(fi.num_devs[slot])
        if idx >= 0:
            vec[idx] = float(
                int(ps[PS_KNIGHT_HAND]) + int(ps[PS_YOP_HAND]) +
                int(ps[PS_MONO_HAND]) + int(ps[PS_RB_HAND]) + int(ps[PS_VP_HAND])
            )

    # --- Production features ---
    robbed_mask = TILE_NODES[state.robber_tile]
    node_prod = fi.node_prod

    if HAS_CYTHON:
        from robottler.bitboard._fast import fill_production_c
        for slot in range(n_slots):
            pidx = (p0_idx + slot) % n_game
            eff_p = np.zeros(5, dtype=np.float64)
            total_p = np.zeros(5, dtype=np.float64)
            fill_production_c(
                state.settlement_bb[pidx], state.city_bb[pidx],
                robbed_mask, node_prod, eff_p, total_p,
            )
            for ri in range(5):
                idx = int(fi.eff_prod[slot, ri])
                if idx >= 0: vec[idx] = eff_p[ri]
                idx = int(fi.total_prod[slot, ri])
                if idx >= 0: vec[idx] = total_p[ri]
    else:
        for slot in range(n_slots):
            pidx = (p0_idx + slot) % n_game

            # Compute total and robbed production in single pass
            total_p = np.zeros(5, dtype=np.float64)
            robbed_p = np.zeros(5, dtype=np.float64)

            mask = int(state.settlement_bb[pidx])
            while mask:
                bit = mask & (-mask)
                node = bit.bit_length() - 1
                mask ^= bit
                total_p += node_prod[node]
                if NODE_BIT[node] & robbed_mask:
                    robbed_p += node_prod[node]

            mask = int(state.city_bb[pidx])
            while mask:
                bit = mask & (-mask)
                node = bit.bit_length() - 1
                mask ^= bit
                contrib = 2.0 * node_prod[node]
                total_p += contrib
                if NODE_BIT[node] & robbed_mask:
                    robbed_p += contrib

            eff_p = total_p - robbed_p

            for ri in range(5):
                idx = int(fi.eff_prod[slot, ri])
                if idx >= 0: vec[idx] = eff_p[ri]
                idx = int(fi.total_prod[slot, ri])
                if idx >= 0: vec[idx] = total_p[ri]

    # --- Reachability features ---
    _fill_reachability_vec(state, p0_idx, n_game, n_slots, fi, node_prod, vec)

    # --- Game features ---
    idx = fi.bank_dev_cards
    if idx >= 0: vec[idx] = float(25 - state.dev_deck_idx)

    for ri in range(5):
        idx = int(fi.bank_res[ri])
        if idx >= 0: vec[idx] = float(state.bank[ri])

    idx = fi.is_moving_robber
    if idx >= 0: vec[idx] = 1.0 if state.current_prompt == PROMPT_MOVE_ROBBER else 0.0

    idx = fi.is_discarding
    if idx >= 0: vec[idx] = 1.0 if state.current_prompt == PROMPT_DISCARD else 0.0


def _fill_reachability_vec(state, p0_idx, n_game, n_slots, fi, node_prod, vec):
    """Fill reachability features directly into vec."""
    if HAS_CYTHON:
        from robottler.bitboard._fast import fill_reachability_c
        for slot in range(n_slots):
            pidx = (p0_idx + slot) % n_game
            out = np.zeros((3, 5), dtype=np.float64)
            fill_reachability_c(
                pidx, n_game, 2,
                state.settlement_bb, state.city_bb,
                state.road_bb, state.reachable_bb,
                state.board_buildable,
                NODE_NEIGHBOR_NODES, NODE_NEIGHBOR_EDGES, NODE_NEIGHBOR_COUNT,
                node_prod, out,
            )
            for level in range(3):
                for ri in range(5):
                    idx = int(fi.reach[slot, level, ri])
                    if idx >= 0: vec[idx] = out[level, ri]
    else:
        _fill_reachability_vec_py(state, p0_idx, n_game, n_slots, fi, node_prod, vec)


def _fill_reachability_vec_py(state, p0_idx, n_game, n_slots, fi, node_prod, vec):
    """Python fallback for reachability feature fill."""
    board_buildable = int(state.board_buildable)

    for slot in range(n_slots):
        pidx = (p0_idx + slot) % n_game
        buildings_mask = int(state.settlement_bb[pidx] | state.city_bb[pidx])
        zero_nodes_mask = int(state.reachable_bb[pidx]) | buildings_mask
        owned_or_buildable = buildings_mask | board_buildable

        # Level 0
        level_0_nodes = zero_nodes_mask & owned_or_buildable
        prod = np.zeros(5, dtype=np.float64)
        _sum_production_np(node_prod, level_0_nodes, prod)
        for ri in range(5):
            idx = int(fi.reach[slot, 0, ri])
            if idx >= 0: vec[idx] = prod[ri]

        # Enemy masks
        enemy_nodes = 0
        opp_road_0 = 0
        opp_road_1 = 0
        for opp in range(n_game):
            if opp == pidx:
                continue
            enemy_nodes |= int(state.settlement_bb[opp]) | int(state.city_bb[opp])
            opp_road_0 |= int(state.road_bb[opp, 0])
            opp_road_1 |= int(state.road_bb[opp, 1])

        # BFS expansion
        last_layer = zero_nodes_mask
        for level in range(1, 3):
            next_layer = last_layer
            frontier = last_layer
            while frontier:
                bit = frontier & (-frontier)
                node = bit.bit_length() - 1
                frontier ^= bit
                if bit & enemy_nodes:
                    continue
                for neighbor, eword, ebit in _NODE_NEIGHBORS[node]:
                    if eword == 0:
                        if opp_road_0 & ebit:
                            continue
                    else:
                        if opp_road_1 & ebit:
                            continue
                    next_layer |= _INODE_BIT[neighbor]

            level_nodes = next_layer & owned_or_buildable
            prod = np.zeros(5, dtype=np.float64)
            _sum_production_np(node_prod, level_nodes, prod)
            for ri in range(5):
                idx = int(fi.reach[slot, level, ri])
                if idx >= 0: vec[idx] = prod[ri]
            last_layer = next_layer


def _sum_production_np(node_prod, node_mask_int, out):
    """Sum production into out[5] for all nodes in bitmask (plain int bitscan)."""
    mask = node_mask_int
    while mask:
        bit = mask & (-mask)
        node = bit.bit_length() - 1
        mask ^= bit
        out += node_prod[node]


# ---------------------------------------------------------------------------
# Numpy helpers for heuristic (avoid dict construction)
# ---------------------------------------------------------------------------

def compute_production_np(state, pidx, node_prod, consider_robber):
    """Compute [5] array of resource production for one player.

    Args:
        state: BitboardState
        pidx: player index
        node_prod: precomputed [54, 5] node production array (from FeatureIndexer)
        consider_robber: if True, skip buildings on robber tile
    """
    robbed_mask = TILE_NODES[state.robber_tile] if consider_robber else np.uint64(0)

    if HAS_CYTHON:
        from robottler.bitboard._fast import fill_production_c
        eff = np.zeros(5, dtype=np.float64)
        total = np.zeros(5, dtype=np.float64)
        fill_production_c(
            state.settlement_bb[pidx], state.city_bb[pidx],
            robbed_mask, node_prod, eff, total,
        )
        return eff if consider_robber else total

    prod = np.zeros(5, dtype=np.float64)

    mask = int(state.settlement_bb[pidx])
    while mask:
        bit = mask & (-mask)
        node = bit.bit_length() - 1
        mask ^= bit
        if consider_robber and (NODE_BIT[node] & robbed_mask):
            continue
        prod += node_prod[node]

    mask = int(state.city_bb[pidx])
    while mask:
        bit = mask & (-mask)
        node = bit.bit_length() - 1
        mask ^= bit
        if consider_robber and (NODE_BIT[node] & robbed_mask):
            continue
        prod += 2.0 * node_prod[node]

    return prod


def compute_reachability_np(state, pidx, node_prod, max_level=2):
    """Compute [max_level+1, 5] array of reachability production for one player.

    Args:
        state: BitboardState
        pidx: player index
        node_prod: precomputed [54, 5] node production array
        max_level: maximum BFS depth (default 2, levels 0/1/2)
    """
    out = np.zeros((max_level + 1, 5), dtype=np.float64)

    if HAS_CYTHON:
        from robottler.bitboard._fast import fill_reachability_c
        fill_reachability_c(
            pidx, state.num_players, max_level,
            state.settlement_bb, state.city_bb,
            state.road_bb, state.reachable_bb,
            state.board_buildable,
            NODE_NEIGHBOR_NODES, NODE_NEIGHBOR_EDGES, NODE_NEIGHBOR_COUNT,
            node_prod, out,
        )
        return out

    board_buildable = int(state.board_buildable)

    buildings_mask = int(state.settlement_bb[pidx] | state.city_bb[pidx])
    zero_nodes_mask = int(state.reachable_bb[pidx]) | buildings_mask
    owned_or_buildable = buildings_mask | board_buildable

    # Level 0
    level_0_nodes = zero_nodes_mask & owned_or_buildable
    _sum_production_np(node_prod, level_0_nodes, out[0])

    # Enemy masks
    enemy_nodes = 0
    opp_road_0 = 0
    opp_road_1 = 0
    for opp in range(state.num_players):
        if opp == pidx:
            continue
        enemy_nodes |= int(state.settlement_bb[opp]) | int(state.city_bb[opp])
        opp_road_0 |= int(state.road_bb[opp, 0])
        opp_road_1 |= int(state.road_bb[opp, 1])

    # BFS expansion
    last_layer = zero_nodes_mask
    for level in range(1, max_level + 1):
        next_layer = last_layer
        frontier = last_layer
        while frontier:
            bit = frontier & (-frontier)
            node = bit.bit_length() - 1
            frontier ^= bit
            if bit & enemy_nodes:
                continue
            for neighbor, eword, ebit in _NODE_NEIGHBORS[node]:
                if eword == 0:
                    if opp_road_0 & ebit:
                        continue
                else:
                    if opp_road_1 & ebit:
                        continue
                next_layer |= _INODE_BIT[neighbor]

        level_nodes = next_layer & owned_or_buildable
        _sum_production_np(node_prod, level_nodes, out[level])
        last_layer = next_layer

    return out
