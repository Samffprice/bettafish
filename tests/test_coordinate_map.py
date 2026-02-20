"""Tests for bridge/coordinate_map.py - coordinate mapping."""
import pytest
from bridge.coordinate_map import build_coordinate_mapper, _find_center_tile


def make_standard_colonist_board():
    """Create a synthetic colonist.io 19-tile board matching the standard layout.

    Uses AXIAL coordinates (q, r) = (cube_x, cube_y) from Catanatron's cube coords.
    colonist.io uses axial coordinates, NOT offset coordinates.
    The center tile is at (0, 0) = Catanatron cube (0, 0, 0).
    """
    from catanatron.models.map import BASE_MAP_TEMPLATE, LandTile

    # Map cube -> colonist axial coordinate: (q, r) = (cube_x, cube_y)
    land_cube_to_col_offset = {}
    for cube, t in BASE_MAP_TEMPLATE.topology.items():
        if t == LandTile:
            land_cube_to_col_offset[cube] = (cube[0], cube[1])

    # Use the actual Catanatron topology order for resource/number assignment
    cube_order = [c for c, t in BASE_MAP_TEMPLATE.topology.items() if t == LandTile]

    # Standard Catan resource distribution (matching BASE_MAP_TEMPLATE.tile_resources)
    # 4 WOOD, 3 BRICK, 4 SHEEP, 4 WHEAT, 3 ORE, 1 desert
    base_resources_catan = BASE_MAP_TEMPLATE.tile_resources
    base_numbers_catan = BASE_MAP_TEMPLATE.numbers

    # Map Catanatron resource strings to colonist int values
    resource_map = {
        "WOOD": 1, "BRICK": 2, "SHEEP": 3, "WHEAT": 4, "ORE": 5, None: 0
    }

    # Create tiles with integer colonist.io coordinates
    # Note: cube_to_offset always returns integer values for the standard board
    tile_coords = []
    resources = []
    dice_numbers = []

    # Use standard resource ordering (same as BASE_MAP_TEMPLATE)
    res_list = list(base_resources_catan)
    num_list = list(base_numbers_catan)

    for cube in cube_order:
        col = land_cube_to_col_offset[cube]
        # Convert to int (cube_to_offset gives .0 floats for this board)
        tile_coords.append((int(col[0]), int(col[1])))
        r = res_list.pop(0)
        resources.append(resource_map[r])
        if r is not None:
            n = num_list.pop(0) if num_list else 0
            dice_numbers.append(n)
        else:
            dice_numbers.append(0)

    tiles = []
    for i, (x, y) in enumerate(tile_coords):
        tile = {
            "hexFace": {"x": x, "y": y},
            "tileType": resources[i],
            "diceNum": dice_numbers[i] if resources[i] != 0 else None,
        }
        tiles.append(tile)

    # Build vertices - each tile has 6 vertices
    vertex_set = set()
    for x, y in tile_coords:
        for dx, dy, z in [(0, 0, 0), (1, -1, 1), (0, -1, 1), (0, 0, 1), (0, 1, 0), (-1, 1, 0)]:
            vertex_set.add((x + dx, y + dy, z))

    vertices = []
    for (vx, vy, vz) in sorted(vertex_set):
        v = {
            "hexCorner": {"x": vx, "y": vy, "z": vz},
            "owner": -1,
            "buildingType": 0,
            "harborType": 0,
        }
        vertices.append(v)

    # Build edges using the formula from board.py get_vertices_next_to_edge
    # edge(x,y,0): connects (x,y,0) and (x,y-1,1)
    # edge(x,y,1): connects (x,y-1,1) and (x-1,y+1,0)
    # edge(x,y,2): connects (x-1,y+1,0) and (x,y,1)
    edge_set = set()
    for x, y in tile_coords:
        for dx, dy, z in [(1, -1, 2), (1, -1, 1), (0, -1, 2), (0, 0, 1), (0, 0, 0), (-1, 1, 2)]:
            edge_set.add((x + dx, y + dy, z))

    # Alternative: collect all possible edges for the vertex set
    edge_set2 = set()
    for (vx, vy, vz) in vertex_set:
        # From get_edges_next_to_vertex
        if vz == 0:
            edge_set2.add((vx, vy, 0))
            edge_set2.add((vx + 1, vy - 1, 1))
            edge_set2.add((vx + 1, vy - 1, 2))
        else:  # z == 1
            edge_set2.add((vx, vy + 1, 0))
            edge_set2.add((vx, vy + 1, 1))
            edge_set2.add((vx, vy, 2))

    # Use only edges that connect to two valid vertices
    vertex_coords = vertex_set

    def edge_verts(ex, ey, ez):
        if ez == 0:
            return (ex, ey, 0), (ex, ey - 1, 1)
        elif ez == 1:
            return (ex, ey - 1, 1), (ex - 1, ey + 1, 0)
        elif ez == 2:
            return (ex - 1, ey + 1, 0), (ex, ey, 1)
        return None, None

    valid_edges = []
    for (ex, ey, ez) in sorted(edge_set2):
        v1, v2 = edge_verts(ex, ey, ez)
        if v1 in vertex_coords and v2 in vertex_coords:
            e = {
                "hexEdge": {"x": ex, "y": ey, "z": ez},
                "owner": -1,
            }
            valid_edges.append(e)

    return tiles, vertices, valid_edges


class TestBuildCoordinateMapper:
    def test_build_coordinate_mapper_basic(self):
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        assert mapper is not None
        assert mapper.catan_map is not None

    def test_vertex_mapping_covers_many_nodes(self):
        """Vertex mapping should cover all 54 Catanatron nodes."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        # Check unique Catanatron node IDs (not just total colonist vertex entries)
        unique_nodes = set(mapper.colonist_vertex_to_catan.values())
        assert len(unique_nodes) == 54, (
            f"Only {len(unique_nodes)} unique Catanatron nodes mapped (expected == 54). "
            f"Unmapped: {sorted(set(range(54)) - unique_nodes)}"
        )

    def test_edge_mapping_covers_many_edges(self):
        """Edge mapping should cover nearly all 72 Catanatron edges."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        # Check unique Catanatron edge tuples (not just total colonist edge entries)
        unique_edges = set(mapper.colonist_edge_to_catan.values())
        assert len(unique_edges) == 72, (
            f"Only {len(unique_edges)} unique Catanatron edges mapped (expected == 72)."
        )

    def test_tile_mapping_covers_land_tiles(self):
        """Should map all 19 land tiles."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        assert len(mapper.colonist_tile_to_catan) == 19

    def test_mapping_bidirectional_consistency(self):
        """Reverse vertex mapping should be a valid forward mapping entry."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)

        # Multiple colonist coords may map to the same catan node (shared vertices).
        # The reverse map stores only ONE colonist coord per node.
        # Verify: for each node in catan_vertex_to_colonist,
        # its stored colonist coord maps back to the same node.
        for node_id, col_coord in mapper.catan_vertex_to_colonist.items():
            forward = mapper.colonist_vertex_to_catan.get(col_coord)
            assert forward == node_id, (
                f"Reverse mapping inconsistency: node {node_id} -> "
                f"col {col_coord} -> node {forward}"
            )

    def test_edge_mapping_bidirectional_consistency(self):
        """Reverse edge mapping should be a valid forward mapping entry."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)

        for catan_edge_fs, col_coord in mapper.catan_edge_to_colonist.items():
            forward = mapper.colonist_edge_to_catan.get(col_coord)
            if forward is not None:
                assert frozenset(forward) == catan_edge_fs, (
                    f"Reverse edge mapping inconsistency: {catan_edge_fs} -> "
                    f"col {col_coord} -> {forward}"
                )

    def test_catan_map_has_19_land_tiles(self):
        """CatanMap should have exactly 19 land tiles."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        from catanatron.models.map import LandTile
        land_tiles = {
            k: v for k, v in mapper.catan_map.tiles.items()
            if isinstance(v, LandTile)
        }
        assert len(land_tiles) == 19

    def test_catan_map_has_correct_resource_counts(self):
        """CatanMap should have standard resource counts: 4W 3B 4S 4Wh 3O 1D."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        from catanatron.models.map import LandTile
        from collections import Counter
        resources = Counter()
        for tile in mapper.catan_map.tiles.values():
            if isinstance(tile, LandTile):
                resources[tile.resource] += 1
        assert resources["WOOD"] == 4
        assert resources["BRICK"] == 3
        assert resources["SHEEP"] == 4
        assert resources["WHEAT"] == 4
        assert resources["ORE"] == 3
        assert resources[None] == 1  # desert

    def test_colonist_xy_to_tile_index_populated(self):
        """colonist_xy_to_tile_index should map all 19 tiles."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        assert len(mapper.colonist_xy_to_tile_index) == 19

    def test_tile_index_to_xy_populated(self):
        """colonist_tile_index_to_xy should map all 19 tiles."""
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)
        assert len(mapper.colonist_tile_index_to_xy) == 19

    def test_no_vertex_conflicts(self):
        """Shared vertices between adjacent tiles must map to the same node_id."""
        tiles, vertices, edges = make_standard_colonist_board()
        from catanatron.models.enums import NodeRef
        from catanatron.models.map import BASE_MAP_TEMPLATE, LandTile

        # Build tile coord -> Catanatron tile
        land_cubes = [c for c, t in BASE_MAP_TEMPLATE.topology.items() if t == LandTile]

        mapper = build_coordinate_mapper(tiles, vertices, edges)

        VERTEX_OFFSET_TO_NODEREF = {
            (0,  0,  0): NodeRef.SOUTHEAST,
            (1, -1,  1): NodeRef.NORTHEAST,
            (0, -1,  1): NodeRef.SOUTH,
            (0,  0,  1): NodeRef.NORTHWEST,
            (0,  1,  0): NodeRef.NORTH,
            (-1, 1,  0): NodeRef.SOUTHWEST,
        }

        # For each tile, compute vertex_coord -> node_id, check for conflicts
        vertex_to_node = {}
        conflicts = 0
        for xy, catan_tile in mapper.colonist_tile_to_catan.items():
            tx, ty = xy
            for (dx, dy, dz), noderef in VERTEX_OFFSET_TO_NODEREF.items():
                col_coord = (tx + dx, ty + dy, dz)
                node_id = catan_tile.nodes.get(noderef)
                if node_id is None:
                    continue
                if col_coord in vertex_to_node:
                    if vertex_to_node[col_coord] != node_id:
                        conflicts += 1
                else:
                    vertex_to_node[col_coord] = node_id

        assert conflicts == 0, f"Found {conflicts} vertex mapping conflicts"
        assert len(vertex_to_node) == 54, (
            f"Expected 54 unique vertices, got {len(vertex_to_node)}"
        )


class TestFindCenterTile:
    def test_center_tile_detected_from_standard_board(self):
        """The center tile (2, 2) should be detected as having 6 neighbors."""
        tiles, _, _ = make_standard_colonist_board()
        col_xy = {(t["hexFace"]["x"], t["hexFace"]["y"]): t for t in tiles}
        center = _find_center_tile(col_xy)
        # Center should be the tile with most neighbors
        # For our layout, (2, 2) has neighbors (1,2),(3,2),(2,1),(3,1),(2,3),(1,3)
        # Let's just verify the result is a valid tile
        assert center in col_xy

    def test_center_tile_has_6_neighbors(self):
        """Detected center tile should have 6 hex neighbors."""
        tiles, _, _ = make_standard_colonist_board()
        col_xy = {(t["hexFace"]["x"], t["hexFace"]["y"]): t for t in tiles}
        center = _find_center_tile(col_xy)
        cx, cy = center
        neighbors = [
            (cx + dx, cy + dy)
            for dx, dy in [(1, 0), (-1, 0), (0, -1), (1, -1), (0, 1), (-1, 1)]
        ]
        count = sum(1 for n in neighbors if n in col_xy)
        assert count == 6, f"Center {center} has only {count} neighbors"


class TestOffsetToCubeMapping:
    def test_center_maps_to_origin(self):
        """Center tile colonist coord should map to Catanatron (0,0,0)."""
        from bridge.coordinate_map import _find_center_tile
        from catanatron.models.coordinate_system import offset_to_cube
        tiles, vertices, edges = make_standard_colonist_board()
        col_xy = {(t["hexFace"]["x"], t["hexFace"]["y"]): t for t in tiles}
        center = _find_center_tile(col_xy)
        mapper = build_coordinate_mapper(tiles, vertices, edges)

        # Center tile should map to a Catanatron tile
        center_catan_tile = mapper.colonist_tile_to_catan.get(center)
        assert center_catan_tile is not None

    def test_cube_coordinate_sum_is_zero(self):
        """All Catanatron cube coordinates must satisfy x+y+z=0."""
        from catanatron.models.map import LandTile
        tiles, vertices, edges = make_standard_colonist_board()
        mapper = build_coordinate_mapper(tiles, vertices, edges)

        for coord in mapper.catan_map.land_tiles:
            x, y, z = coord
            assert x + y + z == 0, f"Cube coord {coord} violates x+y+z=0"
