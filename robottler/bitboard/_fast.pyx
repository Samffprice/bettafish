# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython hot-path implementations for the bitboard Catan engine.

Replaces inner loops that dominate profiling:
  - bitscan / popcount: eliminate Python overhead
  - DFS longest path: eliminate np.uint64 boxing + Python stack frames
  - yield_resources: tight C loop for dice rolls
  - production / reachability fills: C loops for feature extraction
"""

from libc.stdint cimport uint64_t, int16_t, int8_t, int32_t, uint8_t
import numpy as np
cimport numpy as np

cdef extern from *:
    """
    #ifdef _MSC_VER
    #include <intrin.h>
    static inline int ctz64(unsigned long long x) {
        unsigned long idx;
        _BitScanForward64(&idx, x);
        return (int)idx;
    }
    static inline int popcount64(unsigned long long x) {
        return (int)__popcnt64(x);
    }
    #else
    static inline int ctz64(unsigned long long x) {
        return __builtin_ctzll(x);
    }
    static inline int popcount64(unsigned long long x) {
        return __builtin_popcountll(x);
    }
    #endif
    """
    int ctz64(unsigned long long x) nogil
    int popcount64(unsigned long long x) nogil


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def cy_popcount64(uint64_t x):
    """Count set bits in a uint64 value."""
    return popcount64(x)


def bitscan_list(uint64_t x):
    """Return list of set bit indices (lowest first). Replaces bitscan() generator."""
    cdef list result = []
    cdef int idx
    while x:
        idx = ctz64(x)
        result.append(idx)
        x &= x - 1  # clear lowest set bit
    return result


# ---------------------------------------------------------------------------
# DFS Longest Path
# ---------------------------------------------------------------------------

DEF MAX_STACK = 1024

def dfs_longest_path(
    uint64_t[::1] settlement_bb,
    uint64_t[::1] city_bb,
    uint64_t[:, ::1] road_bb,
    int player_idx, int num_players, int start_node,
    int[:, ::1] incident_edges_flat,
    int[::1] incident_edges_count,
    signed char[:, ::1] edge_endpoints
):
    """Stack-based DFS with 2-word uint64 visited bitmask.

    Returns the longest acyclic path length from start_node.
    Matches Catanatron: edges to enemy nodes are NOT counted.
    """
    cdef uint64_t node_bit
    cdef int p

    # Check if start node is enemy-occupied
    node_bit = <uint64_t>1 << start_node
    for p in range(num_players):
        if p == player_idx:
            continue
        if (settlement_bb[p] | city_bb[p]) & node_bit:
            return 0

    # Stack arrays
    cdef int s_node[MAX_STACK]
    cdef uint64_t s_w0[MAX_STACK]
    cdef uint64_t s_w1[MAX_STACK]
    cdef int s_depth[MAX_STACK]
    cdef int top = 0

    # Push initial state
    s_node[0] = start_node
    s_w0[0] = 0
    s_w1[0] = 0
    s_depth[0] = 0
    top = 1

    cdef int best = 0
    cdef int node, depth, eidx, word, neighbor, a, b, i, deg
    cdef uint64_t vis_w0, vis_w1, edge_bit, new_w0, new_w1
    cdef bint expanded, is_enemy

    while top > 0:
        top -= 1
        node = s_node[top]
        vis_w0 = s_w0[top]
        vis_w1 = s_w1[top]
        depth = s_depth[top]

        expanded = False
        deg = incident_edges_count[node]

        for i in range(deg):
            eidx = incident_edges_flat[node, i]
            if eidx < 0:
                break

            # Check if player has this road
            word = eidx >> 6  # // 64
            edge_bit = <uint64_t>1 << (eidx & 63)
            if not (road_bb[player_idx, word] & edge_bit):
                continue

            # Check if edge already visited
            if word == 0:
                if vis_w0 & edge_bit:
                    continue
                new_w0 = vis_w0 | edge_bit
                new_w1 = vis_w1
            else:
                if vis_w1 & edge_bit:
                    continue
                new_w0 = vis_w0
                new_w1 = vis_w1 | edge_bit

            # Find neighbor
            a = edge_endpoints[eidx, 0]
            b = edge_endpoints[eidx, 1]
            neighbor = b if a == node else a

            # Check if neighbor is enemy-occupied
            node_bit = <uint64_t>1 << neighbor
            is_enemy = False
            for p in range(num_players):
                if p == player_idx:
                    continue
                if (settlement_bb[p] | city_bb[p]) & node_bit:
                    is_enemy = True
                    break

            if is_enemy:
                continue

            # Push to stack
            if top < MAX_STACK:
                s_node[top] = neighbor
                s_w0[top] = new_w0
                s_w1[top] = new_w1
                s_depth[top] = depth + 1
                top += 1
                expanded = True

        if not expanded and depth > best:
            best = depth

    return best


# ---------------------------------------------------------------------------
# Yield Resources
# ---------------------------------------------------------------------------

def yield_resources_c(
    int number, int num_players, int robber_tile,
    signed char[::1] tile_number,
    signed char[::1] tile_resource,
    uint64_t[::1] tile_nodes,
    uint64_t[::1] settlement_bb,
    uint64_t[::1] city_bb,
    int16_t[:, ::1] player_state,
    signed char[::1] bank,
    int ps_resource_start
):
    """Distribute resources for a dice roll. Modifies player_state and bank in-place."""
    cdef int intended[4][5]  # max 4 players
    cdef int tid, p, res, ri
    cdef int s_count, c_count
    cdef uint64_t tile_mask
    cdef int total_demand, amount

    # Zero intended
    for p in range(num_players):
        for ri in range(5):
            intended[p][ri] = 0

    # Compute intended yields
    for tid in range(19):
        if tile_number[tid] != number:
            continue
        if tid == robber_tile:
            continue
        res = tile_resource[tid]
        if res < 0:
            continue

        tile_mask = tile_nodes[tid]
        for p in range(num_players):
            s_count = popcount64(settlement_bb[p] & tile_mask)
            c_count = popcount64(city_bb[p] & tile_mask)
            intended[p][res] += s_count + 2 * c_count

    # Check depletion
    for ri in range(5):
        total_demand = 0
        for p in range(num_players):
            total_demand += intended[p][ri]
        if total_demand > bank[ri]:
            for p in range(num_players):
                intended[p][ri] = 0

    # Apply payouts
    for p in range(num_players):
        for ri in range(5):
            amount = intended[p][ri]
            if amount > 0:
                player_state[p, ps_resource_start + ri] += <int16_t>amount
                bank[ri] -= <signed char>amount


# ---------------------------------------------------------------------------
# Production Fill (for feature extraction)
# ---------------------------------------------------------------------------

def fill_production_c(
    uint64_t settle_bb,
    uint64_t city_bb,
    uint64_t robbed_mask,
    double[:, ::1] node_prod,  # [54, 5]
    double[::1] out_eff,       # [5] effective production output
    double[::1] out_total,     # [5] total production output
):
    """Compute effective and total production for one player in a single pass.

    out_eff and out_total must be pre-zeroed.
    """
    cdef uint64_t mask, bit, node_bit
    cdef int node, ri

    # Settlements
    mask = settle_bb
    while mask:
        node = ctz64(mask)
        mask &= mask - 1
        node_bit = <uint64_t>1 << node
        for ri in range(5):
            out_total[ri] += node_prod[node, ri]
        if not (node_bit & robbed_mask):
            for ri in range(5):
                out_eff[ri] += node_prod[node, ri]

    # Cities (2x production)
    mask = city_bb
    while mask:
        node = ctz64(mask)
        mask &= mask - 1
        node_bit = <uint64_t>1 << node
        for ri in range(5):
            out_total[ri] += 2.0 * node_prod[node, ri]
        if not (node_bit & robbed_mask):
            for ri in range(5):
                out_eff[ri] += 2.0 * node_prod[node, ri]


# ---------------------------------------------------------------------------
# Reachability BFS (for feature extraction)
# ---------------------------------------------------------------------------

def fill_reachability_c(
    int pidx, int num_players, int max_level,
    uint64_t[::1] settlement_bb,
    uint64_t[::1] city_bb,
    uint64_t[:, ::1] road_bb,
    uint64_t[::1] reachable_bb,
    uint64_t board_buildable,
    int[:, ::1] nn_nodes,   # [54, max_degree] neighbor node IDs
    int[:, ::1] nn_edges,   # [54, max_degree] neighbor edge indices
    int[::1] nn_count,      # [54] neighbor count per node
    double[:, ::1] node_prod,  # [54, 5]
    double[:, ::1] out      # [max_level+1, 5] output
):
    """BFS reachability with bitmask expansion. Fills per-level production into out.

    out must be pre-zeroed.
    """
    cdef uint64_t buildings_mask = settlement_bb[pidx] | city_bb[pidx]
    cdef uint64_t zero_nodes_mask = reachable_bb[pidx] | buildings_mask
    cdef uint64_t owned_or_buildable = buildings_mask | board_buildable

    # Compute enemy masks
    cdef uint64_t enemy_nodes = 0
    cdef uint64_t opp_road_0 = 0, opp_road_1 = 0
    cdef int opp
    for opp in range(num_players):
        if opp == pidx:
            continue
        enemy_nodes |= settlement_bb[opp] | city_bb[opp]
        opp_road_0 |= road_bb[opp, 0]
        opp_road_1 |= road_bb[opp, 1]

    # Level 0
    cdef uint64_t level_nodes = zero_nodes_mask & owned_or_buildable
    _sum_prod_bitmask(node_prod, level_nodes, out, 0)

    cdef uint64_t last_layer = zero_nodes_mask
    cdef uint64_t next_layer, frontier, bit, ebit
    cdef int node, nb, eidx, eword, i, level, ri

    for level in range(1, max_level + 1):
        next_layer = last_layer
        frontier = last_layer
        while frontier:
            bit = frontier & ((<uint64_t>0) - frontier)  # lowest set bit
            node = ctz64(bit)
            frontier ^= bit

            if bit & enemy_nodes:
                continue

            for i in range(nn_count[node]):
                nb = nn_nodes[node, i]
                eidx = nn_edges[node, i]
                if eidx < 0:
                    break
                eword = eidx >> 6
                ebit = <uint64_t>1 << (eidx & 63)
                if eword == 0:
                    if opp_road_0 & ebit:
                        continue
                else:
                    if opp_road_1 & ebit:
                        continue
                next_layer |= <uint64_t>1 << nb

        level_nodes = next_layer & owned_or_buildable
        _sum_prod_bitmask(node_prod, level_nodes, out, level)
        last_layer = next_layer


cdef void _sum_prod_bitmask(
    double[:, ::1] node_prod,
    uint64_t mask,
    double[:, ::1] out,
    int level
):
    """Sum production for all nodes in bitmask into out[level, :]."""
    cdef int node, ri
    while mask:
        node = ctz64(mask)
        mask &= mask - 1
        for ri in range(5):
            out[level, ri] += node_prod[node, ri]
