"""Bidirectional coordinate mapping between colonist.io and Catanatron.

COORDINATE SYSTEMS:
- colonist.io tiles: hexFace(x, y) offset coordinates
- colonist.io vertices: hexCorner(x, y, z) where z=0 top, z=1 bottom
- colonist.io edges: hexEdge(x, y, z) where z=0,1,2 for three orientations
- Catanatron tiles: cube coordinates (x, y, z) where x+y+z=0
- Catanatron vertices: integer node IDs 0-53
- Catanatron edges: tuple (node_id_a, node_id_b)

ALGORITHM:
1. Find the center tile (has all 6 hex-face neighbors) to establish the
   colonist.io origin offset relative to Catanatron cube (0,0,0).
2. Map each colonist.io offset (x, y) to a Catanatron cube coordinate.
3. Build a CatanMap matching the colonist.io board's resources and numbers.
4. Build vertex mapping: for each colonist tile, match its 6 hexCorner
   coordinates to the 6 NodeRefs of the corresponding Catanatron tile.
   The offsets were verified empirically against Catanatron adjacency data.
5. Build edge mapping: use the verified edge endpoint formula from board.py
   and the shared vertex-to-node mapping from step 4.
6. Port detection: group harbor vertices by type, emit one resource per pair.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from catanatron.models.coordinate_system import Direction, UNIT_VECTORS
from catanatron.models.enums import NodeRef
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    Port,
    initialize_tiles,
)

logger = logging.getLogger(__name__)


@dataclass
class CoordinateMapper:
    """Bidirectional mapping between colonist.io and Catanatron coordinates.

    Attributes:
        colonist_tile_to_catan: (x, y) -> Catanatron tile id
        catan_tile_to_colonist: catanatron tile id -> (x, y) colonist offset
        colonist_vertex_to_catan: (x, y, z) -> catanatron node_id
        catan_vertex_to_colonist: catanatron node_id -> (x, y, z)
        colonist_edge_to_catan: (x, y, z) -> catanatron edge_id (tuple)
        catan_edge_to_colonist: frozenset(edge) -> (x, y, z)
        catan_map: the CatanMap built from colonist.io board data
        colonist_tile_index_to_colonist_xy: tile_index -> (x, y)
        colonist_xy_to_tile_index: (x, y) -> tile_index
    """
    colonist_tile_to_catan: Dict      # (x, y) -> catanatron LandTile
    catan_tile_to_colonist: Dict      # LandTile.id -> (x, y)
    colonist_vertex_to_catan: Dict    # (x, y, z) -> node_id
    catan_vertex_to_colonist: Dict    # node_id -> (x, y, z)
    colonist_edge_to_catan: Dict      # (x, y, z) -> edge tuple
    catan_edge_to_colonist: Dict      # frozenset(edge) -> (x, y, z)
    catan_map: CatanMap
    colonist_tile_index_to_xy: Dict   # tile_list_index -> (x, y)
    colonist_xy_to_tile_index: Dict   # (x, y) -> tile_list_index


def build_coordinate_mapper(
    tiles: List[Dict],
    vertices: List[Dict],
    edges: List[Dict],
    harbor_pairs: Optional[List] = None,
) -> CoordinateMapper:
    """Build a bidirectional coordinate mapper from colonist.io board data.

    Algorithm:
    1. Parse tile coordinates from colonist.io format
    2. Map colonist.io axial coords to Catanatron cube coords: (q,r) -> (q,r,-q-r)
    3. Build CatanMap with matching resources/numbers (with list reversal for .pop())
    4. Build vertex mapping by matching NodeRefs geometrically
    5. Build edge mapping by matching EdgeRefs geometrically

    Args:
        tiles: List of tile dicts from colonist.io tileState
        vertices: List of vertex dicts from colonist.io tileState
        edges: List of edge dicts from colonist.io tileState

    Returns:
        CoordinateMapper with all mappings populated.
    """
    # Step 1: Parse colonist.io tile positions
    col_tile_xy = {}   # tile_index -> (x, y)
    col_xy_to_idx = {} # (x, y) -> tile_index
    col_xy_to_tile = {} # (x, y) -> tile dict

    for i, tile in enumerate(tiles):
        face = _get_attr(tile, "hexFace")
        x, y = int(_get_attr(face, "x")), int(_get_attr(face, "y"))
        col_tile_xy[i] = (x, y)
        col_xy_to_idx[(x, y)] = i
        col_xy_to_tile[(x, y)] = tile

    # Step 2: Build tile mapping colonist_xy -> catanatron cube coord
    # Colonist.io uses AXIAL coordinates (q, r), not offset coordinates.
    # Axial to cube: (q, r) -> (q, r, -q-r)   [x + y + z = 0]
    col_xy_to_cube = {}
    for xy in col_xy_to_tile:
        x, y = xy
        col_xy_to_cube[xy] = (x, y, -x - y)

    center_xy = (0, 0)  # axial origin is always the center
    logger.info(f"Center tile at colonist axial {center_xy}")

    # Step 4: Build CatanMap matching colonist.io board layout
    catan_map = _build_catan_map(tiles, col_xy_to_cube, vertices, harbor_pairs)

    # Step 5: Build tile cross-reference (colonist xy <-> catanatron LandTile)
    col_tile_to_catan, catan_tile_to_col = _build_tile_mapping(
        col_xy_to_tile, col_xy_to_cube, catan_map
    )

    # Step 6: Build vertex mapping
    col_vertex_to_catan, catan_vertex_to_col = _build_vertex_mapping(
        col_xy_to_tile, col_tile_to_catan, catan_map, vertices
    )

    # Step 7: Build edge mapping (pass vertex mapping to ensure consistent resolution)
    col_edge_to_catan, catan_edge_to_col = _build_edge_mapping(
        col_xy_to_tile, col_tile_to_catan, catan_map, edges, col_vertex_to_catan
    )

    logger.info(
        f"Coordinate mapper built: {len(col_vertex_to_catan)} vertices, "
        f"{len(col_edge_to_catan)} edges, {len(col_tile_to_catan)} tiles"
    )

    return CoordinateMapper(
        colonist_tile_to_catan=col_tile_to_catan,
        catan_tile_to_colonist=catan_tile_to_col,
        colonist_vertex_to_catan=col_vertex_to_catan,
        catan_vertex_to_colonist=catan_vertex_to_col,
        colonist_edge_to_catan=col_edge_to_catan,
        catan_edge_to_colonist=catan_edge_to_col,
        catan_map=catan_map,
        colonist_tile_index_to_xy=col_tile_xy,
        colonist_xy_to_tile_index=col_xy_to_idx,
    )


def _get_attr(obj, key):
    """Get attribute from dict or object-with-attributes."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _get_attr_safe(obj, key, default=None):
    """Get attribute from dict or object, returning default if missing."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _find_center_tile(col_xy_to_tile: Dict) -> Tuple[int, int]:
    """Find the tile with 6 hex-face neighbors (the center tile).

    In a 19-tile standard board, exactly one tile (the center) has
    all 6 neighbors present. Uses E/W/NE/NW/SE/SW neighbor offsets
    in the colonist.io offset system.

    For offset coordinates, hex neighbors at (x, y) are:
    - Even rows: (x+1,y), (x-1,y), (x,y-1), (x+1,y-1), (x,y+1), (x+1,y+1)
    - But colonist.io uses a specific offset system.
    Actually in colonist.io's system, from board.py:
    - find_tile_index_by_coordinates(loc.x, loc.y-1), find_tile_index_by_coordinates(loc.x+1, loc.y-1)
    This implies neighbors are offset differently.

    The 6 neighbors in colonist.io's system (from build_vertex_tiles):
    For vertex (x, y, 0): tiles at (x, y), (x, y-1), (x+1, y-1)
    For vertex (x, y, 1): tiles at (x, y), (x-1, y+1), (x, y+1)

    So a tile at (x, y) shares vertices with tiles at:
    (x-1, y), (x+1, y), (x, y-1), (x-1, y+1), (x, y+1), (x+1, y-1)
    (These are the 6 hex neighbors in colonist.io's offset system)
    """
    NEIGHBOR_OFFSETS = [(1, 0), (-1, 0), (0, -1), (1, -1), (0, 1), (-1, 1)]

    best_xy = None
    best_count = -1

    for xy in col_xy_to_tile:
        x, y = xy
        count = sum(1 for dx, dy in NEIGHBOR_OFFSETS if (x + dx, y + dy) in col_xy_to_tile)
        if count > best_count:
            best_count = count
            best_xy = xy

    if best_count < 6:
        # Fallback: use the tile closest to the median coordinates
        logger.warning(
            f"Could not find tile with 6 neighbors (max={best_count}). "
            "Using median-closest tile as center."
        )
        xs = [xy[0] for xy in col_xy_to_tile]
        ys = [xy[1] for xy in col_xy_to_tile]
        med_x = sorted(xs)[len(xs) // 2]
        med_y = sorted(ys)[len(ys) // 2]
        # Find closest tile to (med_x, med_y)
        best_xy = min(col_xy_to_tile.keys(), key=lambda xy: abs(xy[0]-med_x) + abs(xy[1]-med_y))

    return best_xy


def _build_catan_map(
    tiles: List[Dict],
    col_xy_to_cube: Dict,
    vertices: List[Dict],
    harbor_pairs: Optional[List] = None,
) -> CatanMap:
    """Build a CatanMap from colonist.io board data.

    CRITICAL: initialize_tiles() uses .pop() which consumes from the END of the list.
    So we must REVERSE the resource and number lists before passing them.

    Two-pass approach for ports:
    Pass 1: Build with default port ordering to establish vertex mapping.
    Pass 2: Read harborType from vertices to determine actual port resources.
    Actually we just detect port resources from vertex harborType directly.
    """
    from catanatron.models.map import (
        BASE_MAP_TEMPLATE,
        initialize_tiles,
    )
    from bridge.config import COLONIST_TILE_TO_RESOURCE

    # Parse tiles: build ordered resource and number lists matching topology order
    # We need to map cube coords -> (resource, number)
    cube_to_tile_data = {}
    for tile in tiles:
        face = _get_attr(tile, "hexFace")
        x, y = int(_get_attr(face, "x")), int(_get_attr(face, "y"))
        xy = (x, y)
        if xy not in col_xy_to_cube:
            continue
        cube = col_xy_to_cube[xy]

        tile_type = int(_get_attr_safe(tile, "tileType", 0))
        resource = COLONIST_TILE_TO_RESOURCE.get(tile_type)

        # Try various attribute names for dice number
        dice_num = _get_attr_safe(tile, "diceNum", None)
        if dice_num is None:
            dice_num = _get_attr_safe(tile, "diceNumber", None)
        if dice_num is None:
            dice_num = _get_attr_safe(tile, "_diceNum", None)
        if dice_num is not None:
            dice_num = int(dice_num)

        cube_to_tile_data[cube] = (resource, dice_num)

    # Build resource and number lists in topology order
    # BASE_MAP_TEMPLATE.topology iterates in the order we need
    tile_resources_ordered = []
    numbers_ordered = []

    for coordinate, tile_type in BASE_MAP_TEMPLATE.topology.items():
        if tile_type != LandTile:
            continue
        data = cube_to_tile_data.get(coordinate)
        if data is None:
            # Tile not in colonist data - use desert as fallback
            logger.warning(f"No colonist tile data for cube coord {coordinate}, using desert")
            tile_resources_ordered.append(None)
            continue
        resource, number = data
        tile_resources_ordered.append(resource)
        if resource is not None and number is not None:
            numbers_ordered.append(number)

    # Build port resources from vertices harborType, matched to template port order
    port_resources = _determine_port_resources(vertices, col_xy_to_cube, harbor_pairs)

    # CRITICAL: Reverse lists because initialize_tiles() uses .pop() (consumes from END)
    tile_resources_ordered.reverse()
    numbers_ordered.reverse()
    port_resources.reverse()

    tiles_dict = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=numbers_ordered,
        shuffled_port_resources_param=port_resources,
        shuffled_tile_resources_param=tile_resources_ordered,
    )

    return CatanMap.from_tiles(tiles_dict)


def _vdist(a, b):
    """Manhattan distance between two colonist vertex coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _determine_port_resources(
    vertices: List[Dict],
    col_xy_to_cube: Dict,
    harbor_pairs: Optional[List] = None,
) -> List:
    """Extract port resources from colonist.io vertex harborType values,
    matched to BASE_MAP_TEMPLATE port topology order.

    For each PORT tile in the template (in topology iteration order):
    1. Find the adjacent land tile using the port's direction + UNIT_VECTORS
    2. Determine the two NodeRefs on the land tile that face the port
    3. Convert those to colonist vertex coordinates
    4. Look up the harborType at those vertex positions
    5. Convert to Catanatron resource

    colonist.io harborType: 0=no harbor, 1=3:1, 2=WOOD, 3=BRICK, 4=SHEEP, 5=WHEAT, 6=ORE

    Returns list of 9 port resources in BASE_MAP_TEMPLATE port topology order.
    Falls back to default if insufficient port data.
    """
    from bridge.config import COLONIST_HARBOR_TO_RESOURCE

    # Build colonist vertex coord -> harborType lookup
    vertex_harbor = {}
    for v in vertices:
        harbor_type = int(_get_attr_safe(v, "harborType", 0))
        if harbor_type > 0:
            corner = _get_attr_safe(v, "hexCorner")
            if corner is not None:
                coord = (
                    int(_get_attr(corner, "x")),
                    int(_get_attr(corner, "y")),
                    int(_get_attr(corner, "z")),
                )
                vertex_harbor[coord] = harbor_type

    # Use explicit harbor pairs if provided (dataset path), otherwise
    # fall back to individual vertex lookups (live game path).
    if harbor_pairs is None:
        harbor_pairs = []

    # Build reverse: cube coord -> colonist (x, y)
    cube_to_col_xy = {cube: xy for xy, cube in col_xy_to_cube.items()}

    # NodeRef -> colonist vertex offset (inverse of VERTEX_OFFSET_TO_NODEREF)
    NODEREF_TO_VERTEX_OFFSET = {
        NodeRef.NORTH:     (0,  1, 0),
        NodeRef.NORTHEAST: (1, -1, 1),
        NodeRef.SOUTHEAST: (0,  0, 0),
        NodeRef.SOUTH:     (0, -1, 1),
        NodeRef.SOUTHWEST: (-1, 1, 0),
        NodeRef.NORTHWEST: (0,  0, 1),
    }

    # Port direction -> the two NodeRefs on the ADJACENT LAND tile's shared edge.
    # If port faces direction D, land is at port+UNIT_VECTORS[D], and port is on
    # the opposite side from the land tile's perspective.
    PORT_DIR_TO_LAND_NODEREFS = {
        Direction.WEST:      (NodeRef.NORTHEAST, NodeRef.SOUTHEAST),
        Direction.EAST:      (NodeRef.NORTHWEST, NodeRef.SOUTHWEST),
        Direction.NORTHWEST: (NodeRef.SOUTHEAST, NodeRef.SOUTH),
        Direction.NORTHEAST: (NodeRef.SOUTH,     NodeRef.SOUTHWEST),
        Direction.SOUTHEAST: (NodeRef.NORTHWEST, NodeRef.NORTH),
        Direction.SOUTHWEST: (NodeRef.NORTH,     NodeRef.NORTHEAST),
    }

    # Collect all PORT entries with their expected vertex pairs
    port_entries = []  # [(coordinate, direction, v1, v2), ...]
    for coordinate, tile_type in BASE_MAP_TEMPLATE.topology.items():
        if not isinstance(tile_type, tuple):
            continue
        _, direction = tile_type
        dx, dy, dz = UNIT_VECTORS[direction]
        land_cube = (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz)
        col_xy = cube_to_col_xy.get(land_cube)
        if col_xy is None:
            port_entries.append((coordinate, direction, None, None))
            continue
        nr1, nr2 = PORT_DIR_TO_LAND_NODEREFS[direction]
        off1, off2 = NODEREF_TO_VERTEX_OFFSET[nr1], NODEREF_TO_VERTEX_OFFSET[nr2]
        tx, ty = col_xy
        v1 = (tx + off1[0], ty + off1[1], off1[2])
        v2 = (tx + off2[0], ty + off2[1], off2[2])
        port_entries.append((coordinate, direction, v1, v2))

    port_results = [None] * len(port_entries)
    used_pairs = set()

    if harbor_pairs:
        # Dataset path: explicit harbor pairs from port edge data.
        # Two-pass matching — exact first, then nearest — to prevent
        # greedy mis-assignment across template iteration order.
        for port_i, (coord, direction, v1, v2) in enumerate(port_entries):
            if v1 is None:
                continue
            for pi, (pa, pb, p_ht) in enumerate(harbor_pairs):
                if pi in used_pairs:
                    continue
                d = min(_vdist(pa, v1) + _vdist(pb, v2),
                        _vdist(pa, v2) + _vdist(pb, v1))
                if d == 0:
                    port_results[port_i] = p_ht
                    used_pairs.add(pi)
                    break
        for port_i, (coord, direction, v1, v2) in enumerate(port_entries):
            if port_results[port_i] is not None or v1 is None:
                continue
            best_dist, best_ht, best_pi = 999, 0, -1
            for pi, (pa, pb, p_ht) in enumerate(harbor_pairs):
                if pi in used_pairs:
                    continue
                d = min(_vdist(pa, v1) + _vdist(pb, v2),
                        _vdist(pa, v2) + _vdist(pb, v1))
                if d < best_dist:
                    best_dist, best_ht, best_pi = d, p_ht, pi
            if best_ht > 0 and best_dist <= 6:
                port_results[port_i] = best_ht
                used_pairs.add(best_pi)
                logger.debug(
                    f"Port at {coord} dir={direction}: "
                    f"matched harbor pair (dist={best_dist}) "
                    f"for expected vertices {v1}, {v2}"
                )
            else:
                logger.warning(
                    f"Port at {coord} dir={direction}: "
                    f"no harbor type at vertices {v1}, {v2}"
                )
    else:
        # Live game path: harbors are already on vertex coordinates.
        for port_i, (coord, direction, v1, v2) in enumerate(port_entries):
            if v1 is None:
                continue
            ht = vertex_harbor.get(v1, 0) or vertex_harbor.get(v2, 0)
            if ht > 0:
                port_results[port_i] = ht
            else:
                logger.warning(
                    f"Port at {coord} dir={direction}: "
                    f"no harbor type at vertices {v1}, {v2}"
                )

    port_resources = []
    for port_i, (coord, direction, v1, v2) in enumerate(port_entries):
        ht = port_results[port_i]
        if ht is None or ht == 0:
            port_resources.append(None)
            continue

        resource = COLONIST_HARBOR_TO_RESOURCE.get(ht, None)
        if resource == "NONE":
            resource = None
        port_resources.append(resource)

    # Validate: should have exactly 9 ports
    expected = list(BASE_MAP_TEMPLATE.port_resources)
    if len(port_resources) != 9:
        logger.warning(
            f"Found {len(port_resources)} ports, expected 9. Using default ordering."
        )
        return expected

    return port_resources


def _build_tile_mapping(
    col_xy_to_tile: Dict,
    col_xy_to_cube: Dict,
    catan_map: CatanMap,
) -> Tuple[Dict, Dict]:
    """Build bidirectional mapping between colonist xy and Catanatron tiles.

    Returns:
        (col_tile_to_catan, catan_tile_to_col):
        col_tile_to_catan: (x, y) -> LandTile
        catan_tile_to_col: LandTile.id -> (x, y)
    """
    col_tile_to_catan = {}
    catan_tile_to_col = {}

    for xy, cube in col_xy_to_cube.items():
        catan_tile = catan_map.tiles.get(cube)
        if catan_tile is None:
            logger.warning(f"No Catanatron tile at cube coord {cube} (from colonist {xy})")
            continue
        if isinstance(catan_tile, LandTile):
            col_tile_to_catan[xy] = catan_tile
            catan_tile_to_col[catan_tile.id] = xy

    return col_tile_to_catan, catan_tile_to_col


def _build_vertex_mapping(
    col_xy_to_tile: Dict,
    col_tile_to_catan: Dict,
    catan_map: CatanMap,
    vertices: List[Dict],
) -> Tuple[Dict, Dict]:
    """Build bidirectional mapping between colonist vertex coords and Catanatron node IDs.

    For each colonist.io tile, we know which Catanatron tile it corresponds to.
    We can then map each of the 6 colonist vertices of that tile to the 6 NodeRefs
    of the Catanatron tile.

    Colonist.io tile at (tx, ty) has vertices:
    - (tx,   ty,   0)   -> NodeRef.SOUTHEAST
    - (tx+1, ty-1, 1)   -> NodeRef.NORTHEAST
    - (tx,   ty-1, 1)   -> NodeRef.SOUTH
    - (tx,   ty,   1)   -> NodeRef.NORTHWEST
    - (tx,   ty+1, 0)   -> NodeRef.NORTH
    - (tx-1, ty+1, 0)   -> NodeRef.SOUTHWEST

    Derived from first principles: for each pair of adjacent tiles in the axial
    coordinate system, two colonist vertices are shared. Catanatron's
    get_nodes_and_edges() defines which NodeRef pairs are shared between neighbors
    in each Direction. Matching shared colonist vertex offsets to shared NodeRef
    pairs across all 6 directions gives a unique consistent assignment.
    """
    # Colonist vertex offset (dx, dy, z) relative to tile (tx, ty) -> NodeRef
    # Derived from first principles: for each pair of adjacent tiles in the axial
    # coordinate system, the two shared colonist vertices must map to the same
    # Catanatron node_id. The Catanatron sharing rules (get_nodes_and_edges) define
    # which NodeRefs are shared between neighbors in each Direction. By matching
    # shared colonist vertex offsets to shared NodeRef pairs across all 6 directions,
    # the unique consistent assignment is:
    VERTEX_OFFSET_TO_NODEREF = {
        (0,  0,  0): NodeRef.SOUTHEAST,
        (1, -1,  1): NodeRef.NORTHEAST,
        (0, -1,  1): NodeRef.SOUTH,
        (0,  0,  1): NodeRef.NORTHWEST,
        (0,  1,  0): NodeRef.NORTH,
        (-1, 1,  0): NodeRef.SOUTHWEST,
    }

    col_vertex_to_catan = {}
    catan_vertex_to_col = {}

    for xy, col_tile in col_xy_to_tile.items():
        catan_tile = col_tile_to_catan.get(xy)
        if catan_tile is None:
            continue

        tx, ty = xy
        for (dx, dy, dz), noderef in VERTEX_OFFSET_TO_NODEREF.items():
            vx, vy, vz = tx + dx, ty + dy, dz
            col_coord = (vx, vy, vz)

            node_id = catan_tile.nodes.get(noderef)
            if node_id is None:
                continue

            if col_coord in col_vertex_to_catan:
                # Already mapped - verify consistency
                existing = col_vertex_to_catan[col_coord]
                if existing != node_id:
                    logger.debug(
                        f"Vertex {col_coord} mapped to both {existing} and {node_id}. "
                        "Keeping first mapping."
                    )
                continue

            col_vertex_to_catan[col_coord] = node_id
            catan_vertex_to_col[node_id] = col_coord

    return col_vertex_to_catan, catan_vertex_to_col


def _build_edge_mapping(
    col_xy_to_tile: Dict,
    col_tile_to_catan: Dict,
    catan_map: CatanMap,
    edges: List[Dict],
    col_vertex_to_catan: Dict,
) -> Tuple[Dict, Dict]:
    """Build bidirectional mapping between colonist edge coords and Catanatron edge IDs.

    Uses the vertex-to-catan mapping from _build_vertex_mapping (passed in as
    col_vertex_to_catan) to ensure consistent first-writer-wins vertex resolution.
    Rebuilding the vertex mapping here with last-writer-wins semantics would cause
    ~10 edges to map to incorrect node pairs.

    Edge endpoint formula from board.py get_vertices_next_to_edge:
      edge(x,y,0): vertices (x,y,0) and (x,y-1,1)
      edge(x,y,1): vertices (x,y-1,1) and (x-1,y+1,0)
      edge(x,y,2): vertices (x-1,y+1,0) and (x,y,1)
    """
    # Build: colonist edge coord -> pair of vertex node_ids using get_vertices_next_to_edge logic
    # edge(x,y,0): vertices (x,y,0) and (x,y-1,1)
    # edge(x,y,1): vertices (x,y-1,1) and (x-1,y+1,0)
    # edge(x,y,2): vertices (x-1,y+1,0) and (x,y,1)

    def edge_vertices(ex, ey, ez):
        """Return the two vertex coords for a colonist edge."""
        if ez == 0:
            return ((ex, ey, 0), (ex, ey - 1, 1))
        elif ez == 1:
            return ((ex, ey - 1, 1), (ex - 1, ey + 1, 0))
        elif ez == 2:
            return ((ex - 1, ey + 1, 0), (ex, ey, 1))
        return None

    col_edge_to_catan = {}
    catan_edge_to_col = {}

    # Process all edges from colonist data
    for edge in edges:
        he = _get_attr(edge, "hexEdge")
        ex, ey, ez = int(_get_attr(he, "x")), int(_get_attr(he, "y")), int(_get_attr(he, "z"))
        col_edge_coord = (ex, ey, ez)

        v_pair = edge_vertices(ex, ey, ez)
        if v_pair is None:
            continue

        v1_col, v2_col = v_pair
        n1 = col_vertex_to_catan.get(v1_col)
        n2 = col_vertex_to_catan.get(v2_col)

        if n1 is None or n2 is None:
            continue

        catan_edge = (min(n1, n2), max(n1, n2))
        col_edge_to_catan[col_edge_coord] = catan_edge
        catan_edge_to_col[frozenset(catan_edge)] = col_edge_coord

    return col_edge_to_catan, catan_edge_to_col
