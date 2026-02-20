"""Precomputed topology masks for the bitboard Catan engine.

All constants are built once at import from STATIC_GRAPH and a reference CatanMap.
They are module-level constants, immutable after initialization.

Node masks use uint64 (54 nodes fit in 64 bits).
Edge masks use a pair of uint64 (72 edges need 2 words).
"""

import numpy as np
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.map import (
    NUM_NODES, NUM_EDGES, NUM_TILES,
    CatanMap, BASE_MAP_TEMPLATE, LandTile,
)
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE

# Resource index mapping (same as freqdeck order)
RESOURCE_INDEX = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}
RESOURCE_NAMES = [WOOD, BRICK, SHEEP, WHEAT, ORE]

# ---------------------------------------------------------------------------
# Reference map (used only during initialization)
# ---------------------------------------------------------------------------
# Note: the reference map is random — topology is fixed but tile resources/numbers
# vary. For masks that depend on topology only (nodes, edges, adjacency), this is fine.
# For tile resource/number masks, we build from a specific game's map at runtime.
# But we CAN build topology-only masks from ANY map since they share the same graph.

_ref_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

# ---------------------------------------------------------------------------
# Node masks (54 nodes → uint64)
# ---------------------------------------------------------------------------

def _build_node_bit():
    """NODE_BIT[i] = uint64 with only bit i set."""
    arr = np.zeros(NUM_NODES, dtype=np.uint64)
    for i in range(NUM_NODES):
        arr[i] = np.uint64(1) << np.uint64(i)
    return arr

NODE_BIT = _build_node_bit()


def _build_adjacent_nodes():
    """ADJACENT_NODES[i] = uint64 bitmask of nodes adjacent to node i (land nodes only)."""
    arr = np.zeros(NUM_NODES, dtype=np.uint64)
    for node in range(NUM_NODES):
        mask = np.uint64(0)
        for neighbor in STATIC_GRAPH.neighbors(node):
            if neighbor < NUM_NODES:  # land nodes only
                mask |= np.uint64(1) << np.uint64(neighbor)
        arr[node] = mask
    return arr

ADJACENT_NODES = _build_adjacent_nodes()


def _build_neighbor_mask():
    """NEIGHBOR_MASK[i] = node i itself + all its adjacent nodes (for distance rule)."""
    arr = np.zeros(NUM_NODES, dtype=np.uint64)
    for i in range(NUM_NODES):
        arr[i] = ADJACENT_NODES[i] | NODE_BIT[i]
    return arr

NEIGHBOR_MASK = _build_neighbor_mask()


# ---------------------------------------------------------------------------
# Edge indexing and masks (72 land edges)
# ---------------------------------------------------------------------------

def _build_edge_list():
    """Sorted list of 72 canonical (a, b) land edge tuples with a < b."""
    edges = set()
    for a, b in STATIC_GRAPH.edges():
        if a < NUM_NODES and b < NUM_NODES:
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)

EDGE_LIST = _build_edge_list()
assert len(EDGE_LIST) == NUM_EDGES, f"Expected {NUM_EDGES} edges, got {len(EDGE_LIST)}"

# Bidirectional lookup: (a,b) → index and (b,a) → index
EDGE_TO_INDEX = {}
for idx, (a, b) in enumerate(EDGE_LIST):
    EDGE_TO_INDEX[(a, b)] = idx
    EDGE_TO_INDEX[(b, a)] = idx

# Which uint64 word and which bit within it for each edge
EDGE_BIT_WORD = np.array([idx // 64 for idx in range(NUM_EDGES)], dtype=np.int8)
EDGE_BIT_MASK = np.array([np.uint64(1) << np.uint64(idx % 64) for idx in range(NUM_EDGES)],
                          dtype=np.uint64)

# Edge endpoints: EDGE_ENDPOINTS[edge_idx] = (node_a, node_b)
EDGE_ENDPOINTS = np.array(EDGE_LIST, dtype=np.int8)

# INCIDENT_EDGES[node_id] = list of edge indices incident to that node
INCIDENT_EDGES = [[] for _ in range(NUM_NODES)]
for idx, (a, b) in enumerate(EDGE_LIST):
    INCIDENT_EDGES[a].append(idx)
    INCIDENT_EDGES[b].append(idx)
# Convert to tuple for immutability
INCIDENT_EDGES = tuple(tuple(edges) for edges in INCIDENT_EDGES)

# ADJACENT_EDGE_NODES[edge_idx] = uint64 bitmask of the two endpoint nodes
ADJACENT_EDGE_NODES = np.zeros(NUM_EDGES, dtype=np.uint64)
for idx, (a, b) in enumerate(EDGE_LIST):
    ADJACENT_EDGE_NODES[idx] = NODE_BIT[a] | NODE_BIT[b]


# ---------------------------------------------------------------------------
# Tile masks
# ---------------------------------------------------------------------------

def _build_tile_masks(catan_map):
    """Build tile-related masks from a specific CatanMap instance.

    Returns:
        tile_nodes: uint64[19] - bitmask of 6 nodes per tile
        node_tiles: list[list[int]] - tile IDs per node
        tile_resource: int8[19] - resource index per tile (-1 for desert)
        tile_number: int8[19] - dice number per tile (0 for desert)
        tile_coord_to_id: dict - coordinate → tile ID
        tile_id_to_coord: dict - tile ID → coordinate
    """
    tile_nodes = np.zeros(NUM_TILES, dtype=np.uint64)
    node_tiles = [[] for _ in range(NUM_NODES)]
    tile_resource = np.full(NUM_TILES, -1, dtype=np.int8)
    tile_number = np.zeros(NUM_TILES, dtype=np.int8)
    tile_coord_to_id = {}
    tile_id_to_coord = {}

    for coord, tile in catan_map.land_tiles.items():
        tid = tile.id
        tile_coord_to_id[coord] = tid
        tile_id_to_coord[tid] = coord

        # Build node bitmask for this tile
        mask = np.uint64(0)
        for nid in tile.nodes.values():
            if nid < NUM_NODES:
                mask |= np.uint64(1) << np.uint64(nid)
                node_tiles[nid].append(tid)
        tile_nodes[tid] = mask

        # Resource and number
        if tile.resource is not None:
            tile_resource[tid] = RESOURCE_INDEX[tile.resource]
            tile_number[tid] = tile.number
        # else: desert, stays -1 / 0

    return (tile_nodes, [tuple(t) for t in node_tiles],
            tile_resource, tile_number, tile_coord_to_id, tile_id_to_coord)

# Build from reference map — topology masks. Note: tile_resource and tile_number
# are map-specific and must be rebuilt per game. But the topology (which nodes
# belong to which tile) is FIXED for the BASE template.
(TILE_NODES, NODE_TILES, _REF_TILE_RESOURCE, _REF_TILE_NUMBER,
 _REF_TILE_COORD_TO_ID, _REF_TILE_ID_TO_COORD) = _build_tile_masks(_ref_map)


def build_tile_data(catan_map):
    """Build map-specific tile resource/number arrays. Call once per game."""
    tile_resource = np.full(NUM_TILES, -1, dtype=np.int8)
    tile_number = np.zeros(NUM_TILES, dtype=np.int8)
    tile_coord_to_id = {}
    tile_id_to_coord = {}

    for coord, tile in catan_map.land_tiles.items():
        tid = tile.id
        tile_coord_to_id[coord] = tid
        tile_id_to_coord[tid] = coord
        if tile.resource is not None:
            tile_resource[tid] = RESOURCE_INDEX[tile.resource]
            tile_number[tid] = tile.number

    return tile_resource, tile_number, tile_coord_to_id, tile_id_to_coord


# ---------------------------------------------------------------------------
# Port masks
# ---------------------------------------------------------------------------

# PORT_TYPE_INDEX: resource → port type index (0-4 for resources, 5 for 3:1)
PORT_TYPE_INDEX = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4, None: 5}
NUM_PORT_TYPES = 6


def _build_port_masks(catan_map):
    """PORT_NODES[port_type] = uint64 bitmask of nodes with that port access."""
    port_nodes = np.zeros(NUM_PORT_TYPES, dtype=np.uint64)
    for resource, node_ids in catan_map.port_nodes.items():
        ptype = PORT_TYPE_INDEX[resource]
        mask = np.uint64(0)
        for nid in node_ids:
            if nid < NUM_NODES:
                mask |= np.uint64(1) << np.uint64(nid)
        port_nodes[ptype] = mask
    return port_nodes

# Port topology is also map-specific (port positions are randomized)
# Build from reference for now; actual game should call build_port_masks()
PORT_NODES = _build_port_masks(_ref_map)


def build_port_masks(catan_map):
    """Build map-specific port masks. Call once per game."""
    return _build_port_masks(catan_map)


# ---------------------------------------------------------------------------
# Dice probability lookup
# ---------------------------------------------------------------------------
DICE_PROBA = np.zeros(13, dtype=np.float64)
for i in range(1, 7):
    for j in range(1, 7):
        DICE_PROBA[i + j] += 1.0 / 36.0

# TILE_PROBA[tile_id] = probability of that tile's number being rolled (0 for desert)
def _build_tile_proba(tile_number):
    proba = np.zeros(NUM_TILES, dtype=np.float64)
    for tid in range(NUM_TILES):
        num = tile_number[tid]
        if num > 0:
            proba[tid] = DICE_PROBA[num]
    return proba


# ---------------------------------------------------------------------------
# Flat topology arrays for Cython (C-compatible int32 arrays)
# ---------------------------------------------------------------------------

# INCIDENT_EDGES as padded 2D int32 array
_max_ie_degree = max(len(ie) for ie in INCIDENT_EDGES)
INCIDENT_EDGES_FLAT = np.full((NUM_NODES, _max_ie_degree), -1, dtype=np.int32)
INCIDENT_EDGES_COUNT = np.zeros(NUM_NODES, dtype=np.int32)
for _node, _edges in enumerate(INCIDENT_EDGES):
    INCIDENT_EDGES_COUNT[_node] = len(_edges)
    for _i, _eidx in enumerate(_edges):
        INCIDENT_EDGES_FLAT[_node, _i] = _eidx

# Node neighbors via edges (for BFS in features/reachability)
# NODE_NEIGHBOR_NODES[node, i] = neighbor node id
# NODE_NEIGHBOR_EDGES[node, i] = edge index connecting node to neighbor
_nn_raw = [[] for _ in range(NUM_NODES)]
for _eidx, (_a, _b) in enumerate(EDGE_LIST):
    _nn_raw[_a].append((_b, _eidx))
    _nn_raw[_b].append((_a, _eidx))
_max_nn_degree = max(len(nn) for nn in _nn_raw)
NODE_NEIGHBOR_NODES = np.full((NUM_NODES, _max_nn_degree), -1, dtype=np.int32)
NODE_NEIGHBOR_EDGES = np.full((NUM_NODES, _max_nn_degree), -1, dtype=np.int32)
NODE_NEIGHBOR_COUNT = np.zeros(NUM_NODES, dtype=np.int32)
for _node, _neighbors in enumerate(_nn_raw):
    NODE_NEIGHBOR_COUNT[_node] = len(_neighbors)
    for _i, (_nb, _eidx) in enumerate(_neighbors):
        NODE_NEIGHBOR_NODES[_node, _i] = _nb
        NODE_NEIGHBOR_EDGES[_node, _i] = _eidx
del _nn_raw, _node, _neighbors, _nb, _eidx, _i, _edges, _a, _b


# ---------------------------------------------------------------------------
# Popcount / bitscan utilities (with Cython fast-path when available)
# ---------------------------------------------------------------------------

try:
    from robottler.bitboard._fast import cy_popcount64, bitscan_list
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


def _py_popcount64(x):
    """Count set bits in a uint64 value (pure Python fallback)."""
    x = int(x)
    return bin(x).count('1')


def _py_bitscan(x):
    """Yield indices of set bits in a uint64 value, lowest first (pure Python)."""
    x = int(x)
    while x:
        bit = x & (-x)
        yield bit.bit_length() - 1
        x ^= bit


if HAS_CYTHON:
    def popcount64(x):
        return cy_popcount64(int(x))

    def bitscan(x):
        return iter(bitscan_list(int(x)))
else:
    popcount64 = _py_popcount64
    bitscan = _py_bitscan
