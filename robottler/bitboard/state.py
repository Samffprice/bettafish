"""BitboardState: compact Catan game state for fast search.

All board state is represented using numpy arrays and uint64 bitmasks.
Designed for O(1) copy and fast action application.
"""

import numpy as np
from robottler.bitboard.masks import (
    NUM_NODES, NUM_EDGES, NUM_TILES, NUM_PORT_TYPES,
    NODE_BIT, ADJACENT_NODES, NEIGHBOR_MASK,
    EDGE_LIST, EDGE_TO_INDEX, EDGE_ENDPOINTS, INCIDENT_EDGES,
    ADJACENT_EDGE_NODES, TILE_NODES, RESOURCE_INDEX,
    PORT_TYPE_INDEX, popcount64, bitscan,
    HAS_CYTHON, INCIDENT_EDGES_FLAT, INCIDENT_EDGES_COUNT,
)

# Player state array indices
PS_VP = 0
PS_ROADS_AVAIL = 1
PS_SETTLE_AVAIL = 2
PS_CITY_AVAIL = 3
PS_HAS_ROAD = 4          # longest road bonus
PS_HAS_ARMY = 5          # largest army bonus
PS_HAS_ROLLED = 6
PS_HAS_PLAYED_DEV = 7
PS_ACTUAL_VP = 8
PS_LONGEST_ROAD = 9
PS_KNIGHT_START = 10      # owned at start of turn (can play)
PS_MONO_START = 11
PS_YOP_START = 12
PS_RB_START = 13
PS_WOOD = 14
PS_BRICK = 15
PS_SHEEP = 16
PS_WHEAT = 17
PS_ORE = 18
PS_KNIGHT_HAND = 19
PS_PLAYED_KNIGHT = 20
PS_YOP_HAND = 21
PS_PLAYED_YOP = 22
PS_MONO_HAND = 23
PS_PLAYED_MONO = 24
PS_RB_HAND = 25
PS_PLAYED_RB = 26
PS_VP_HAND = 27
PS_PLAYED_VP = 28
PS_SIZE = 29

# Resource offsets in player_state
PS_RESOURCE_START = PS_WOOD  # 14
PS_RESOURCE_END = PS_ORE + 1  # 19

# Dev card hand/played pairs: (hand_idx, played_idx)
# Order: KNIGHT, YOP, MONO, RB, VP
DEV_HAND_INDICES = [PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND]
DEV_PLAYED_INDICES = [PS_PLAYED_KNIGHT, PS_PLAYED_YOP, PS_PLAYED_MONO, PS_PLAYED_RB, PS_PLAYED_VP]
DEV_START_INDICES = [PS_KNIGHT_START, PS_YOP_START, PS_MONO_START, PS_RB_START]

# Mapping from Catanatron dev card names to our indices
DEV_NAME_TO_IDX = {
    "KNIGHT": 0, "YEAR_OF_PLENTY": 1, "MONOPOLY": 2, "ROAD_BUILDING": 3, "VICTORY_POINT": 4,
}

# Prompt constants (mirrors ActionPrompt enum as ints for fast comparison)
PROMPT_BUILD_INITIAL_SETTLEMENT = 0
PROMPT_BUILD_INITIAL_ROAD = 1
PROMPT_PLAY_TURN = 2
PROMPT_DISCARD = 3
PROMPT_MOVE_ROBBER = 4
PROMPT_DECIDE_TRADE = 5
PROMPT_DECIDE_ACCEPTEES = 6


class BitboardState:
    """Compact Catan game state using numpy arrays and bitmasks.

    All mutable state is in numpy arrays for fast copy().
    Shared immutables (catan_map, colors, etc.) are referenced, not copied.
    """

    __slots__ = [
        # Board bitmasks — per player
        'settlement_bb',    # uint64[N]
        'city_bb',          # uint64[N]
        'road_bb',          # uint64[N, 2] — 2 words for 72 edges
        'board_buildable',  # uint64 — nodes where settlements can go
        'robber_tile',      # int8

        # Player state
        'player_state',     # int16[N, PS_SIZE]
        'num_players',      # int

        # Bank & deck
        'bank',             # int8[5]
        'dev_deck',         # int8[25] — full ordered deck
        'dev_deck_idx',     # int — next card to draw

        # Game machine
        'current_player_idx',   # int
        'current_turn_idx',     # int
        'num_turns',            # int
        'is_initial_build_phase',  # bool
        'is_discarding',       # bool
        'is_moving_knight',    # bool
        'is_road_building',    # bool
        'free_roads',          # int
        'current_prompt',      # int (PROMPT_* constants)

        # Connected components
        'reachable_bb',    # uint64[N] — all nodes reachable via roads
        'component_ids',   # int8[N, 54] — node → component_id or -1
        'num_components',  # int8[N]
        'road_lengths',    # int16[N] — longest road per player
        'road_holder',     # int (-1 if nobody)
        'road_holder_length',  # int
        'army_holder',     # int (-1 if nobody)
        'army_holder_size',  # int

        # Port access cache
        'port_access',     # uint8[N] — bitfield: bit i = has port type i

        # Shared immutables (references only, not copied)
        'colors',          # tuple of Color
        'color_to_index',  # dict Color → int
        'catan_map',       # CatanMap reference
        'tile_resource',   # int8[19] — per-game
        'tile_number',     # int8[19] — per-game
        'tile_coord_to_id',  # dict
        'tile_id_to_coord',  # dict
        'port_nodes',      # uint64[6] — per-game port masks
        'discard_limit',   # int

        # Zobrist
        'zobrist_hash',    # uint64

        # Trade state (simplified — search doesn't do domestic trades)
        'is_resolving_trade',  # bool

        # Friendly robber (1v1: can't rob players with < 3 visible VP)
        'friendly_robber',  # bool
    ]

    def __init__(self, num_players=2):
        """Create an empty BitboardState. Use convert.game_to_bitboard() to populate."""
        n = num_players
        self.num_players = n

        self.settlement_bb = np.zeros(n, dtype=np.uint64)
        self.city_bb = np.zeros(n, dtype=np.uint64)
        self.road_bb = np.zeros((n, 2), dtype=np.uint64)
        self.board_buildable = np.uint64(0)
        self.robber_tile = np.int8(0)

        self.player_state = np.zeros((n, PS_SIZE), dtype=np.int16)
        self.bank = np.zeros(5, dtype=np.int8)
        self.dev_deck = np.zeros(25, dtype=np.int8)
        self.dev_deck_idx = 0

        self.current_player_idx = 0
        self.current_turn_idx = 0
        self.num_turns = 0
        self.is_initial_build_phase = True
        self.is_discarding = False
        self.is_moving_knight = False
        self.is_road_building = False
        self.free_roads = 0
        self.current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT

        self.reachable_bb = np.zeros(n, dtype=np.uint64)
        self.component_ids = np.full((n, NUM_NODES), -1, dtype=np.int8)
        self.num_components = np.zeros(n, dtype=np.int8)
        self.road_lengths = np.zeros(n, dtype=np.int16)
        self.road_holder = -1
        self.road_holder_length = 0
        self.army_holder = -1
        self.army_holder_size = 0

        self.port_access = np.zeros(n, dtype=np.uint8)

        self.colors = ()
        self.color_to_index = {}
        self.catan_map = None
        self.tile_resource = np.full(NUM_TILES, -1, dtype=np.int8)
        self.tile_number = np.zeros(NUM_TILES, dtype=np.int8)
        self.tile_coord_to_id = {}
        self.tile_id_to_coord = {}
        self.port_nodes = np.zeros(NUM_PORT_TYPES, dtype=np.uint64)
        self.discard_limit = 7

        self.zobrist_hash = np.uint64(0)
        self.is_resolving_trade = False
        self.friendly_robber = False

    def copy(self):
        """Fast copy — numpy .copy() for arrays, scalars assigned directly."""
        s = BitboardState.__new__(BitboardState)

        s.settlement_bb = self.settlement_bb.copy()
        s.city_bb = self.city_bb.copy()
        s.road_bb = self.road_bb.copy()
        s.board_buildable = self.board_buildable
        s.robber_tile = self.robber_tile

        s.player_state = self.player_state.copy()
        s.num_players = self.num_players
        s.bank = self.bank.copy()
        s.dev_deck = self.dev_deck.copy()
        s.dev_deck_idx = self.dev_deck_idx

        s.current_player_idx = self.current_player_idx
        s.current_turn_idx = self.current_turn_idx
        s.num_turns = self.num_turns
        s.is_initial_build_phase = self.is_initial_build_phase
        s.is_discarding = self.is_discarding
        s.is_moving_knight = self.is_moving_knight
        s.is_road_building = self.is_road_building
        s.free_roads = self.free_roads
        s.current_prompt = self.current_prompt

        s.reachable_bb = self.reachable_bb.copy()
        s.component_ids = self.component_ids.copy()
        s.num_components = self.num_components.copy()
        s.road_lengths = self.road_lengths.copy()
        s.road_holder = self.road_holder
        s.road_holder_length = self.road_holder_length
        s.army_holder = self.army_holder
        s.army_holder_size = self.army_holder_size

        s.port_access = self.port_access.copy()

        # Shared immutables — reference only
        s.colors = self.colors
        s.color_to_index = self.color_to_index
        s.catan_map = self.catan_map
        s.tile_resource = self.tile_resource
        s.tile_number = self.tile_number
        s.tile_coord_to_id = self.tile_coord_to_id
        s.tile_id_to_coord = self.tile_id_to_coord
        s.port_nodes = self.port_nodes
        s.discard_limit = self.discard_limit

        s.zobrist_hash = self.zobrist_hash
        s.is_resolving_trade = self.is_resolving_trade
        s.friendly_robber = self.friendly_robber

        return s

    def current_color(self):
        return self.colors[self.current_player_idx]

    def has_edge(self, player_idx, edge_idx):
        """Check if a player has a road at edge_idx."""
        word = edge_idx // 64
        bit = np.uint64(1) << np.uint64(edge_idx % 64)
        return bool(self.road_bb[player_idx, word] & bit)

    def set_edge(self, player_idx, edge_idx):
        """Set a road bit for a player."""
        word = edge_idx // 64
        bit = np.uint64(1) << np.uint64(edge_idx % 64)
        self.road_bb[player_idx, word] |= bit

    def has_settlement(self, player_idx, node_id):
        return bool(self.settlement_bb[player_idx] & NODE_BIT[node_id])

    def has_city(self, player_idx, node_id):
        return bool(self.city_bb[player_idx] & NODE_BIT[node_id])

    def any_building_at(self, node_id):
        """Check if any player has a building at node_id."""
        bit = NODE_BIT[node_id]
        for p in range(self.num_players):
            if (self.settlement_bb[p] | self.city_bb[p]) & bit:
                return True
        return False

    def building_owner(self, node_id):
        """Return (player_idx, 'S'|'C') or (None, None)."""
        bit = NODE_BIT[node_id]
        for p in range(self.num_players):
            if self.settlement_bb[p] & bit:
                return p, 'S'
            if self.city_bb[p] & bit:
                return p, 'C'
        return None, None

    def is_enemy_node(self, node_id, player_idx):
        """True if a different player has a building at node_id."""
        bit = NODE_BIT[node_id]
        for p in range(self.num_players):
            if p == player_idx:
                continue
            if (self.settlement_bb[p] | self.city_bb[p]) & bit:
                return True
        return False

    # --- Connected Components ---

    def build_road(self, player_idx, edge_idx):
        """Add a road and update connected components. Returns (prev_road_holder, new_road_holder)."""
        self.set_edge(player_idx, edge_idx)

        a, b = int(EDGE_ENDPOINTS[edge_idx, 0]), int(EDGE_ENDPOINTS[edge_idx, 1])
        a_comp = self.component_ids[player_idx, a]
        b_comp = self.component_ids[player_idx, b]
        a_enemy = self.is_enemy_node(a, player_idx)
        b_enemy = self.is_enemy_node(b, player_idx)

        prev_road_holder = self.road_holder

        if a_comp == -1 and b_comp == -1:
            # Both nodes new — shouldn't happen if build is valid
            # (unless initial placement creates isolated road)
            cid = self.num_components[player_idx]
            self.num_components[player_idx] += 1
            if not a_enemy:
                self.component_ids[player_idx, a] = cid
                self.reachable_bb[player_idx] |= NODE_BIT[a]
            if not b_enemy:
                self.component_ids[player_idx, b] = cid
                self.reachable_bb[player_idx] |= NODE_BIT[b]
        elif a_comp == -1 and not a_enemy:
            self.component_ids[player_idx, a] = b_comp
            self.reachable_bb[player_idx] |= NODE_BIT[a]
        elif b_comp == -1 and not b_enemy:
            self.component_ids[player_idx, b] = a_comp
            self.reachable_bb[player_idx] |= NODE_BIT[b]
        elif a_comp != -1 and b_comp != -1 and a_comp != b_comp:
            # Merge: relabel the smaller component into the larger
            keep, kill = (a_comp, b_comp) if a_comp < b_comp else (b_comp, a_comp)
            for node in range(NUM_NODES):
                if self.component_ids[player_idx, node] == kill:
                    self.component_ids[player_idx, node] = keep
            self.num_components[player_idx] -= 1
            # Re-number components > kill to fill the gap
            for node in range(NUM_NODES):
                if self.component_ids[player_idx, node] > kill:
                    self.component_ids[player_idx, node] -= 1
        # else: a_comp == b_comp, cycle — no component changes needed

        # Recompute longest road for this player (only the affected component)
        self._recompute_longest_road(player_idx)

        # Only take over longest road if STRICTLY exceeding current holder's length
        # (matching Catanatron: candidate_length >= 5 and candidate_length > self.road_length)
        my_length = int(self.road_lengths[player_idx])
        if my_length >= 5 and my_length > self.road_holder_length:
            self.road_holder = player_idx
            self.road_holder_length = my_length

        return prev_road_holder, self.road_holder

    def build_settlement_initial(self, player_idx, node_id):
        """Build initial settlement: new component, update buildable."""
        self.settlement_bb[player_idx] |= NODE_BIT[node_id]

        # Create new component
        cid = self.num_components[player_idx]
        self.num_components[player_idx] += 1
        self.component_ids[player_idx, node_id] = cid
        self.reachable_bb[player_idx] |= NODE_BIT[node_id]

        # Remove node and neighbors from buildable
        self.board_buildable &= ~NEIGHBOR_MASK[node_id]

        # Update port access
        self._update_port_access(player_idx, node_id)

        return -1, self.road_holder  # no plowing in initial phase

    def build_settlement_normal(self, player_idx, node_id):
        """Build normal settlement: plow opponents, update buildable."""
        self.settlement_bb[player_idx] |= NODE_BIT[node_id]

        prev_road_holder = self.road_holder
        plowed = False

        # Check for plowing: does any opponent have 2+ road edges through this node?
        for opp in range(self.num_players):
            if opp == player_idx:
                continue
            incident_opp_edges = []
            for eidx in INCIDENT_EDGES[node_id]:
                if self.has_edge(opp, eidx):
                    incident_opp_edges.append(eidx)

            if len(incident_opp_edges) >= 2:
                # Plowing! Recompute connected components for opponent
                plowed = True
                self._recompute_components_after_plow(opp, node_id)
                self._recompute_longest_road(opp)

        # Remove node and neighbors from buildable
        self.board_buildable &= ~NEIGHBOR_MASK[node_id]

        # Update port access
        self._update_port_access(player_idx, node_id)

        if plowed:
            # After plowing, re-evaluate road holder matching Catanatron:
            # max(road_lengths.items(), key=...) — no >= 5 threshold
            best_player = 0
            best_length = int(self.road_lengths[0])
            for p in range(1, self.num_players):
                length = int(self.road_lengths[p])
                if length > best_length:
                    best_length = length
                    best_player = p
            self.road_holder = best_player
            self.road_holder_length = best_length

        return prev_road_holder, self.road_holder

    def build_city(self, player_idx, node_id):
        """Upgrade settlement to city."""
        self.settlement_bb[player_idx] &= ~NODE_BIT[node_id]
        self.city_bb[player_idx] |= NODE_BIT[node_id]

    def _update_port_access(self, player_idx, node_id):
        """Update port_access bitfield when a settlement/city is built."""
        bit = NODE_BIT[node_id]
        for pt in range(NUM_PORT_TYPES):
            if self.port_nodes[pt] & bit:
                self.port_access[player_idx] |= np.uint8(1 << pt)

    def has_port(self, player_idx, port_type_idx):
        """Check if player has access to a specific port type."""
        return bool(self.port_access[player_idx] & (1 << port_type_idx))

    # --- Longest Road ---

    def _recompute_longest_road(self, player_idx):
        """Recompute longest road for a player using DFS with edge bitmask."""
        longest = 0

        # Find all nodes that are road endpoints for this player
        road_nodes = np.uint64(0)
        for eidx in range(NUM_EDGES):
            if self.has_edge(player_idx, eidx):
                a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                road_nodes |= NODE_BIT[a] | NODE_BIT[b]

        if not road_nodes:
            self.road_lengths[player_idx] = 0
            return

        # DFS from each road node
        for start_node in bitscan(road_nodes):
            length = self._dfs_longest_path(player_idx, start_node)
            if length > longest:
                longest = length

        self.road_lengths[player_idx] = longest

    def _dfs_longest_path(self, player_idx, start_node):
        """DFS to find longest acyclic path from start_node using player's roads.

        Uses Cython fast path when available, falling back to Python implementation.
        Matches Catanatron's longest_acyclic_path: edges to enemy nodes are NOT
        counted and enemy nodes are not expanded through.
        """
        if HAS_CYTHON:
            from robottler.bitboard._fast import dfs_longest_path as _cy_dfs
            return _cy_dfs(
                self.settlement_bb, self.city_bb, self.road_bb,
                player_idx, self.num_players, start_node,
                INCIDENT_EDGES_FLAT, INCIDENT_EDGES_COUNT, EDGE_ENDPOINTS,
            )
        return self._dfs_longest_path_py(player_idx, start_node)

    def _dfs_longest_path_py(self, player_idx, start_node):
        """Python fallback for DFS longest path."""
        # Don't start DFS from enemy-occupied nodes
        if self.is_enemy_node(start_node, player_idx):
            return 0

        best = 0
        # Stack: (node, visited_w0, visited_w1, depth)
        stack = [(start_node, np.uint64(0), np.uint64(0), 0)]

        while stack:
            node, vis_w0, vis_w1, depth = stack.pop()

            expanded = False
            for eidx in INCIDENT_EDGES[node]:
                if not self.has_edge(player_idx, eidx):
                    continue

                # Check if edge already visited
                word = eidx // 64
                bit = np.uint64(1) << np.uint64(eidx % 64)
                if word == 0:
                    if vis_w0 & bit:
                        continue
                    new_w0, new_w1 = vis_w0 | bit, vis_w1
                else:
                    if vis_w1 & bit:
                        continue
                    new_w0, new_w1 = vis_w0, vis_w1 | bit

                # Find the other endpoint
                a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                neighbor = b if a == node else a

                # Can't expand past enemy nodes — edge doesn't count
                # (matching Catanatron's longest_acyclic_path)
                if self.is_enemy_node(neighbor, player_idx):
                    continue

                stack.append((neighbor, new_w0, new_w1, depth + 1))
                expanded = True

            if not expanded and depth > best:
                best = depth

        return best

    def _update_road_holder(self):
        """Update road_holder based on all players' road_lengths."""
        best_player = -1
        best_length = 0
        for p in range(self.num_players):
            length = int(self.road_lengths[p])
            if length >= 5 and length > best_length:
                best_length = length
                best_player = p
        self.road_holder = best_player
        self.road_holder_length = best_length

    def _recompute_components_after_plow(self, player_idx, plow_node):
        """Recompute connected components for player_idx after an enemy settlement
        is placed at plow_node, potentially splitting their road network.

        Full BFS recomputation for this player.
        """
        # Clear existing components
        self.component_ids[player_idx, :] = -1
        self.reachable_bb[player_idx] = np.uint64(0)
        self.num_components[player_idx] = 0

        # Find all nodes that have roads from this player
        road_node_set = set()
        for eidx in range(NUM_EDGES):
            if self.has_edge(player_idx, eidx):
                a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                road_node_set.add(a)
                road_node_set.add(b)

        # Also include settlement/city nodes
        for node in bitscan(self.settlement_bb[player_idx] | self.city_bb[player_idx]):
            road_node_set.add(node)

        visited = set()
        comp_id = 0

        for start in sorted(road_node_set):
            if start in visited:
                continue

            # BFS from start
            queue = [start]
            component_nodes = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component_nodes.append(node)

                if self.is_enemy_node(node, player_idx):
                    continue  # don't expand through enemy nodes

                for eidx in INCIDENT_EDGES[node]:
                    if not self.has_edge(player_idx, eidx):
                        continue
                    a, b = int(EDGE_ENDPOINTS[eidx, 0]), int(EDGE_ENDPOINTS[eidx, 1])
                    neighbor = b if a == node else a
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Assign component
            for node in component_nodes:
                if not self.is_enemy_node(node, player_idx):
                    self.component_ids[player_idx, node] = comp_id
                    self.reachable_bb[player_idx] |= NODE_BIT[node]
            comp_id += 1

        self.num_components[player_idx] = comp_id

    # --- Resource queries ---

    def player_resource_count(self, player_idx, resource_idx):
        """Get count of a specific resource for a player."""
        return int(self.player_state[player_idx, PS_RESOURCE_START + resource_idx])

    def player_total_resources(self, player_idx):
        """Get total resource card count for a player."""
        return int(np.sum(self.player_state[player_idx, PS_RESOURCE_START:PS_RESOURCE_END]))

    def winning_player(self, vps_to_win=10):
        """Return player index of winner, or -1 if no winner."""
        for p in range(self.num_players):
            if self.player_state[p, PS_ACTUAL_VP] >= vps_to_win:
                return p
        return -1
