"""Convert dataset JSON map format to what build_coordinate_mapper() expects.

Dataset format (dict indexed by string keys):
  tileHexStates:    {"0": {"x":0, "y":-2, "type":4, "diceNumber":4}, ...}
  tileCornerStates: {"0": {"x":0, "y":-2, "z":0}, ...}
  tileEdgeStates:   {"0": {"x":1, "y":-3, "z":2}, ...}
  portEdgeStates:   {"0": {"x":0, "y":-2, "z":0, "type":2}, ...}

build_coordinate_mapper() expects lists of dicts:
  tiles:    [{"hexFace": {"x":0, "y":-2}, "tileType":4, "diceNumber":4}, ...]
  vertices: [{"hexCorner": {"x":0, "y":-2, "z":0}, "harborType":0}, ...]
  edges:    [{"hexEdge": {"x":1, "y":-3, "z":2}}, ...]
"""
from typing import Dict, List, Tuple


def adapt_tiles(tile_hex_states: Dict) -> List[Dict]:
    """Convert tileHexStates to tiles list for build_coordinate_mapper()."""
    tiles = []
    for idx in sorted(tile_hex_states.keys(), key=int):
        t = tile_hex_states[idx]
        tiles.append({
            "hexFace": {"x": t["x"], "y": t["y"]},
            "tileType": t.get("type", 0),
            "diceNumber": t.get("diceNumber"),
        })
    return tiles


def adapt_vertices(
    corner_states: Dict, port_edge_states: Dict
) -> List[Dict]:
    """Convert tileCornerStates + portEdgeStates to vertices list.

    Ports in the dataset are on edges, but build_coordinate_mapper() expects
    harborType on vertices. Convert using edge endpoint formulas:
      edge(x,y,0) -> vertices (x,y,0) and (x,y-1,1)
      edge(x,y,1) -> vertices (x,y-1,1) and (x-1,y+1,0)
      edge(x,y,2) -> vertices (x-1,y+1,0) and (x,y,1)
    """
    # Build harbor lookup: vertex (x,y,z) -> harbor type
    vertex_harbors: Dict[Tuple, int] = {}
    for _idx, port in port_edge_states.items():
        px, py, pz = port["x"], port["y"], port["z"]
        port_type = port.get("type", 0)
        if port_type == 0:
            continue
        # Compute the two vertex endpoints of this port edge
        if pz == 0:
            v1, v2 = (px, py, 0), (px, py - 1, 1)
        elif pz == 1:
            v1, v2 = (px, py - 1, 1), (px - 1, py + 1, 0)
        elif pz == 2:
            v1, v2 = (px - 1, py + 1, 0), (px, py, 1)
        else:
            continue
        vertex_harbors.setdefault(v1, port_type)
        vertex_harbors.setdefault(v2, port_type)

    vertices = []
    for idx in sorted(corner_states.keys(), key=int):
        c = corner_states[idx]
        xyz = (c["x"], c["y"], c["z"])
        vertices.append({
            "hexCorner": {"x": c["x"], "y": c["y"], "z": c["z"]},
            "harborType": vertex_harbors.get(xyz, 0),
        })
    return vertices


def extract_harbor_pairs(port_edge_states: Dict) -> List[Tuple]:
    """Extract explicit harbor pairs from port edge data.

    Each port edge defines exactly two vertices that form a harbor pair.
    Returns list of (v1_xyz, v2_xyz, harbor_type) tuples.
    """
    pairs = []
    for _idx, port in port_edge_states.items():
        px, py, pz = port["x"], port["y"], port["z"]
        port_type = port.get("type", 0)
        if port_type == 0:
            continue
        if pz == 0:
            v1, v2 = (px, py, 0), (px, py - 1, 1)
        elif pz == 1:
            v1, v2 = (px, py - 1, 1), (px - 1, py + 1, 0)
        elif pz == 2:
            v1, v2 = (px - 1, py + 1, 0), (px, py, 1)
        else:
            continue
        pairs.append((v1, v2, port_type))
    return pairs


def adapt_edges(edge_states: Dict) -> List[Dict]:
    """Convert tileEdgeStates to edges list for build_coordinate_mapper()."""
    edges = []
    for idx in sorted(edge_states.keys(), key=int):
        e = edge_states[idx]
        edges.append({
            "hexEdge": {"x": e["x"], "y": e["y"], "z": e["z"]},
        })
    return edges


def build_index_maps(
    corner_states: Dict, edge_states: Dict
) -> Tuple[Dict[int, Tuple], Dict[int, Tuple]]:
    """Build index -> (x,y,z) maps for vertices and edges.

    Events reference vertices/edges by their string index in the mapState dicts.
    These maps let us translate those indices to coordinate tuples.

    Returns:
        (vertex_idx_to_xyz, edge_idx_to_xyz)
    """
    vertex_idx_to_xyz = {}
    for idx_str, c in corner_states.items():
        vertex_idx_to_xyz[int(idx_str)] = (c["x"], c["y"], c["z"])

    edge_idx_to_xyz = {}
    for idx_str, e in edge_states.items():
        edge_idx_to_xyz[int(idx_str)] = (e["x"], e["y"], e["z"])

    return vertex_idx_to_xyz, edge_idx_to_xyz
