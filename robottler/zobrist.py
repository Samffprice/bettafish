"""Zobrist hashing for Catan game states.

External tracker that observes a Catanatron Game and maintains an incremental
Zobrist hash. Does NOT modify any Catanatron classes.

Usage:
    tracker = ZobristTracker()
    tracker.init(game)               # compute full hash
    game.execute(action)
    tracker.update(game, action)     # incremental update
    assert tracker.zobrist_hash == tracker.compute_full(game)  # always true
"""

import numpy as np
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import (
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT,
    SETTLEMENT, CITY,
    ActionType,
)

# ---------------------------------------------------------------------------
# Edge indexing: canonical sorted edge list from STATIC_GRAPH (land nodes only)
# ---------------------------------------------------------------------------
from catanatron.models.map import NUM_NODES as _NUM_LAND_NODES

_land_edges_set = set()
for a, b in STATIC_GRAPH.edges():
    if a < _NUM_LAND_NODES and b < _NUM_LAND_NODES:
        _land_edges_set.add(tuple(sorted((a, b))))
EDGE_LIST = sorted(_land_edges_set)

EDGE_TO_INDEX = {}
for idx, (a, b) in enumerate(EDGE_LIST):
    EDGE_TO_INDEX[(a, b)] = idx
    EDGE_TO_INDEX[(b, a)] = idx

NUM_NODES = 54
NUM_EDGES = len(EDGE_LIST)  # should be 72
NUM_TILES = 19
MAX_PLAYERS = 4
NUM_RESOURCES = 5
NUM_DEV_TYPES = 5
MAX_COUNT = 32  # max resource/dev card count per player

# Resource and dev card index mappings
RESOURCE_INDEX = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}
RESOURCE_NAMES = [WOOD, BRICK, SHEEP, WHEAT, ORE]
DEV_INDEX = {KNIGHT: 0, YEAR_OF_PLENTY: 1, MONOPOLY: 2, ROAD_BUILDING: 3, VICTORY_POINT: 4}
DEV_NAMES = [KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT]

# ---------------------------------------------------------------------------
# Precomputed random 64-bit Zobrist keys (deterministic seed for reproducibility)
# ---------------------------------------------------------------------------
_rng = np.random.RandomState(0xCA7A_2025)


def _rand_table(*shape):
    """Generate a table of random uint64 values."""
    # np.random.RandomState doesn't have a uint64 method; combine two uint32s
    lo = _rng.randint(0, 2**32, size=shape, dtype=np.uint64)
    hi = _rng.randint(0, 2**32, size=shape, dtype=np.uint64)
    return (hi << np.uint64(32)) | lo


# Z_SETTLEMENT[player_idx][node_id]
Z_SETTLEMENT = _rand_table(MAX_PLAYERS, NUM_NODES)
# Z_CITY[player_idx][node_id]
Z_CITY = _rand_table(MAX_PLAYERS, NUM_NODES)
# Z_ROAD[player_idx][edge_index]
Z_ROAD = _rand_table(MAX_PLAYERS, NUM_EDGES)
# Z_ROBBER[tile_id]  (0..18)
Z_ROBBER = _rand_table(NUM_TILES)
# Z_RESOURCE[player_idx][resource_type][count]
Z_RESOURCE = _rand_table(MAX_PLAYERS, NUM_RESOURCES, MAX_COUNT)
# Z_DEV_HAND[player_idx][dev_type][count]
Z_DEV_HAND = _rand_table(MAX_PLAYERS, NUM_DEV_TYPES, MAX_COUNT)
# Z_PLAYER_TO_MOVE[player_idx]
Z_PLAYER_TO_MOVE = _rand_table(MAX_PLAYERS)


# ---------------------------------------------------------------------------
# Tile coordinate to tile ID mapping (built once from a reference map)
# ---------------------------------------------------------------------------
def _build_tile_coord_to_id():
    """Build mapping from robber_coordinate (cube tuple) -> tile.id."""
    from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
    ref_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    return {coord: tile.id for coord, tile in ref_map.land_tiles.items()}


# We can't use a fixed mapping since maps are randomized (tile resources/numbers
# change, but coordinates and tile IDs are stable for a given template).
# Instead, we build the mapping per-game from the game's own map.
def _get_tile_id_from_coord(catan_map, coord):
    """Get tile ID from a coordinate using the game's map."""
    tile = catan_map.land_tiles.get(coord)
    if tile is not None:
        return tile.id
    return None


# ---------------------------------------------------------------------------
# ZobristTracker
# ---------------------------------------------------------------------------
class ZobristTracker:
    """Maintains an incremental Zobrist hash for a Catanatron Game.

    The tracker is external to the Game — it never modifies Game internals.
    Call `init(game)` once, then `update(game, action)` after each
    `game.execute(action)`.
    """

    __slots__ = ('zobrist_hash', '_num_players', '_color_to_idx',
                 '_prev_robber_tile_id', '_prev_resources', '_prev_dev_hands',
                 '_prev_player_idx')

    def __init__(self):
        self.zobrist_hash = np.uint64(0)
        self._num_players = 0
        self._color_to_idx = {}
        self._prev_robber_tile_id = 0
        self._prev_resources = None  # shape (N, 5) int
        self._prev_dev_hands = None  # shape (N, 5) int
        self._prev_player_idx = 0

    def init(self, game):
        """Compute full hash and set up tracker state."""
        self._num_players = len(game.state.colors)
        self._color_to_idx = dict(game.state.color_to_index)
        self.zobrist_hash = self.compute_full(game)
        self._snapshot_state(game)

    def _snapshot_state(self, game):
        """Capture mutable state that we diff against on update."""
        state = game.state
        n = self._num_players

        # Robber
        tile_id = _get_tile_id_from_coord(state.board.map, state.board.robber_coordinate)
        self._prev_robber_tile_id = tile_id if tile_id is not None else 0

        # Resources per player
        self._prev_resources = np.zeros((n, 5), dtype=np.int32)
        for color, idx in self._color_to_idx.items():
            key = f"P{idx}"
            for ri, rname in enumerate(RESOURCE_NAMES):
                self._prev_resources[idx, ri] = state.player_state[f"{key}_{rname}_IN_HAND"]

        # Dev cards in hand per player
        self._prev_dev_hands = np.zeros((n, 5), dtype=np.int32)
        for color, idx in self._color_to_idx.items():
            key = f"P{idx}"
            for di, dname in enumerate(DEV_NAMES):
                self._prev_dev_hands[idx, di] = state.player_state[f"{key}_{dname}_IN_HAND"]

        # Current player
        self._prev_player_idx = state.current_player_index

    def compute_full(self, game):
        """Recompute hash from scratch. Used for init and verification."""
        h = np.uint64(0)
        state = game.state

        # Buildings
        for node_id, (color, btype) in state.board.buildings.items():
            pidx = self._color_to_idx[color]
            if btype == SETTLEMENT:
                h ^= Z_SETTLEMENT[pidx, node_id]
            elif btype == CITY:
                h ^= Z_CITY[pidx, node_id]

        # Roads
        seen_edges = set()
        for edge, color in state.board.roads.items():
            canonical = tuple(sorted(edge))
            if canonical in seen_edges:
                continue
            seen_edges.add(canonical)
            eidx = EDGE_TO_INDEX[canonical]
            pidx = self._color_to_idx[color]
            h ^= Z_ROAD[pidx, eidx]

        # Robber
        tile_id = _get_tile_id_from_coord(state.board.map, state.board.robber_coordinate)
        if tile_id is not None:
            h ^= Z_ROBBER[tile_id]

        # Resources
        for color, idx in self._color_to_idx.items():
            key = f"P{idx}"
            for ri, rname in enumerate(RESOURCE_NAMES):
                count = state.player_state[f"{key}_{rname}_IN_HAND"]
                if count > 0:
                    h ^= Z_RESOURCE[idx, ri, min(count, MAX_COUNT - 1)]

        # Dev cards in hand
        for color, idx in self._color_to_idx.items():
            key = f"P{idx}"
            for di, dname in enumerate(DEV_NAMES):
                count = state.player_state[f"{key}_{dname}_IN_HAND"]
                if count > 0:
                    h ^= Z_DEV_HAND[idx, di, min(count, MAX_COUNT - 1)]

        # Current player to move
        h ^= Z_PLAYER_TO_MOVE[state.current_player_index]

        return h

    def update(self, game, action):
        """Incrementally update hash by diffing state before/after action.

        Must be called AFTER game.execute(action) has been applied.
        """
        state = game.state
        h = self.zobrist_hash

        # --- XOR out old current player, XOR in new ---
        h ^= Z_PLAYER_TO_MOVE[self._prev_player_idx]
        h ^= Z_PLAYER_TO_MOVE[state.current_player_index]

        atype = action.action_type
        color = action.color
        pidx = self._color_to_idx[color]

        if atype == ActionType.BUILD_SETTLEMENT:
            node_id = action.value
            h ^= Z_SETTLEMENT[pidx, node_id]

        elif atype == ActionType.BUILD_CITY:
            node_id = action.value
            # Settlement removed, city added
            h ^= Z_SETTLEMENT[pidx, node_id]
            h ^= Z_CITY[pidx, node_id]

        elif atype == ActionType.BUILD_ROAD:
            edge = action.value
            eidx = EDGE_TO_INDEX[tuple(sorted(edge))]
            h ^= Z_ROAD[pidx, eidx]

        elif atype == ActionType.MOVE_ROBBER:
            coord = action.value[0]
            new_tile_id = _get_tile_id_from_coord(state.board.map, coord)
            old_tile_id = self._prev_robber_tile_id
            if old_tile_id is not None:
                h ^= Z_ROBBER[old_tile_id]
            if new_tile_id is not None:
                h ^= Z_ROBBER[new_tile_id]

        # --- Diff resources for ALL players (handles yields, trades, steals, etc.) ---
        for idx in range(self._num_players):
            key = f"P{idx}"
            for ri, rname in enumerate(RESOURCE_NAMES):
                old_count = int(self._prev_resources[idx, ri])
                new_count = state.player_state[f"{key}_{rname}_IN_HAND"]
                if old_count != new_count:
                    if old_count > 0:
                        h ^= Z_RESOURCE[idx, ri, min(old_count, MAX_COUNT - 1)]
                    if new_count > 0:
                        h ^= Z_RESOURCE[idx, ri, min(new_count, MAX_COUNT - 1)]

        # --- Diff dev cards for ALL players ---
        for idx in range(self._num_players):
            key = f"P{idx}"
            for di, dname in enumerate(DEV_NAMES):
                old_count = int(self._prev_dev_hands[idx, di])
                new_count = state.player_state[f"{key}_{dname}_IN_HAND"]
                if old_count != new_count:
                    if old_count > 0:
                        h ^= Z_DEV_HAND[idx, di, min(old_count, MAX_COUNT - 1)]
                    if new_count > 0:
                        h ^= Z_DEV_HAND[idx, di, min(new_count, MAX_COUNT - 1)]

        self.zobrist_hash = h
        self._snapshot_state(game)

    def copy(self):
        """Create a copy of this tracker (for search tree branching)."""
        t = ZobristTracker.__new__(ZobristTracker)
        t.zobrist_hash = self.zobrist_hash
        t._num_players = self._num_players
        t._color_to_idx = self._color_to_idx  # immutable after init
        t._prev_robber_tile_id = self._prev_robber_tile_id
        t._prev_resources = self._prev_resources.copy()
        t._prev_dev_hands = self._prev_dev_hands.copy()
        t._prev_player_idx = self._prev_player_idx
        return t


# ---------------------------------------------------------------------------
# TranspositionTable
# ---------------------------------------------------------------------------
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class TranspositionTable:
    """Simple hash table for alpha-beta search transpositions.

    Stores (depth, score, flag, best_action) keyed by Zobrist hash.
    Uses Python dict with replacement policy: always replace if new
    entry has >= depth.
    """

    __slots__ = ('table', 'hits', 'misses', 'stores')

    def __init__(self):
        self.table = {}
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def probe(self, zhash, depth, alpha, beta):
        """Look up a position in the table.

        Returns:
            (hit: bool, score: float, best_action: Action|None)
            If hit=True, score is usable. best_action may be set even on miss
            (for move ordering).
        """
        entry = self.table.get(zhash)
        if entry is None:
            self.misses += 1
            return False, 0.0, None

        e_depth, e_score, e_flag, e_action = entry

        if e_depth >= depth:
            if e_flag == EXACT:
                self.hits += 1
                return True, e_score, e_action
            elif e_flag == LOWERBOUND and e_score >= beta:
                self.hits += 1
                return True, e_score, e_action
            elif e_flag == UPPERBOUND and e_score <= alpha:
                self.hits += 1
                return True, e_score, e_action

        # Not a usable hit, but return best_action for move ordering
        self.misses += 1
        return False, 0.0, e_action

    def store(self, zhash, depth, score, flag, best_action):
        """Store or replace an entry."""
        existing = self.table.get(zhash)
        if existing is None or existing[0] <= depth:
            self.table[zhash] = (depth, score, flag, best_action)
            self.stores += 1

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def __len__(self):
        return len(self.table)
