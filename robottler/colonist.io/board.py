from dev_cards import DevCards
from resources import Resources


class Board:
    tiles = None
    vertices = None
    edges = None

    robber_tile = None

    own_production = {
        Resources.WOOD: 0,
        Resources.BRICK: 0,
        Resources.SHEEP: 0,
        Resources.WHEAT: 0,
        Resources.ORE: 0
    }

    resources = {
        Resources.WOOD: 0,
        Resources.BRICK: 0,
        Resources.SHEEP: 0,
        Resources.WHEAT: 0,
        Resources.ORE: 0
    }

    bank_resources = {
        Resources.WOOD: 0,
        Resources.BRICK: 0,
        Resources.SHEEP: 0,
        Resources.WHEAT: 0,
        Resources.ORE: 0
    }

    bank_trades = {
        Resources.WOOD: 4,
        Resources.BRICK: 4,
        Resources.SHEEP: 4,
        Resources.WHEAT: 4,
        Resources.ORE: 4
    }

    bank_dev_cards = 25

    own_settlements = []
    own_cities = []
    own_harbors = set()
    own_dev_cards = {
        DevCards.ROBBER: 0,
        DevCards.VICTORY_POINT: 0,
        DevCards.YEAR_OF_PLENTY: 0,
        DevCards.MONOPOLY: 0,
        DevCards.ROAD_BUILDING: 0
    }

    adjacency_map = []
    vertex_tiles = []
    tile_vertices = []

    def __init__(self, board):
        self.tiles = board.tileState.tiles
        self.vertices = board.tileState.tileCorners
        self.edges = board.tileState.tileEdges

        self.build_adjency_map()
        self.build_vertex_tiles()
        self.build_tile_vertices()

    def build_adjency_map(self):
        for vertex in self.vertices:
            loc = vertex.hexCorner

            # TODO: clean up lambda, maybe use a function instead
            data = map(lambda edge_index: {
                "edge_index": edge_index,
                "vertex_index": self.get_other_vertex_next_to_edge(self.edges[edge_index], vertex)},
                       self.get_edges_next_to_vertex(loc.x, loc.y, loc.z))
            self.adjacency_map.append(list(data))

    def build_vertex_tiles(self):
        for vertex in self.vertices:
            loc = vertex.hexCorner

            tiles = []

            tiles.append(self.find_tile_index_by_coordinates(loc.x, loc.y))
            if loc.z == 0:
                tiles.append(self.find_tile_index_by_coordinates(loc.x, loc.y - 1))
                tiles.append(self.find_tile_index_by_coordinates(
                    loc.x + 1, loc.y - 1))
            else:
                tiles.append(self.find_tile_index_by_coordinates(
                    loc.x - 1, loc.y + 1))
                tiles.append(self.find_tile_index_by_coordinates(loc.x, loc.y + 1))

            tiles = [x for x in tiles if x is not None]

            self.vertex_tiles.append(tiles)

    def build_tile_vertices(self):
        for tile in self.tiles:
            loc = tile.hexFace

            vertices = []

            vertices.append(
                self.find_vertex_index_by_coordinates(loc.x, loc.y, 0))
            vertices.append(
                self.find_vertex_index_by_coordinates(loc.x, loc.y, 1))
            vertices.append(self.find_vertex_index_by_coordinates(
                loc.x + 1, loc.y - 1, 1))
            vertices.append(
                self.find_vertex_index_by_coordinates(loc.x, loc.y - 1, 1))
            vertices.append(
                self.find_vertex_index_by_coordinates(loc.x, loc.y + 1, 0))
            vertices.append(self.find_vertex_index_by_coordinates(
                loc.x - 1, loc.y + 1, 0))

            self.tile_vertices.append(vertices)

    def find_tile_index_by_coordinates(self, x, y):
        for i, tile in enumerate(self.tiles):
            if tile.hexFace.x == x and tile.hexFace.y == y:
                return i
        return None

    def find_edge_index_by_coordinates(self, x, y, z):
        for i, edge in enumerate(self.edges):
            if edge.hexEdge.x == x and edge.hexEdge.y == y and edge.hexEdge.z == z:
                return i
        return None

    def find_vertex_index_by_coordinates(self, x, y, z):
        for i, vertex in enumerate(self.vertices):
            if vertex.hexCorner.x == x and vertex.hexCorner.y == y and vertex.hexCorner.z == z:
                return i
        return None

    def get_edges_next_to_vertex(self, x, y, z):
        edges = []
        if z == 0:
            edges.append(self.find_edge_index_by_coordinates(x, y, 0))
            edges.append(self.find_edge_index_by_coordinates(x + 1, y - 1, 1))
            edges.append(self.find_edge_index_by_coordinates(x + 1, y - 1, 2))
        if z == 1:
            edges.append(self.find_edge_index_by_coordinates(x, y + 1, 0))
            edges.append(self.find_edge_index_by_coordinates(x, y + 1, 1))
            edges.append(self.find_edge_index_by_coordinates(x, y, 2))

        # remove None roads for vertices next to harbor tiles
        edges = [x for x in edges if x is not None]

        return edges

    def get_vertices_next_to_edge(self, edge):
        x = edge.hexEdge.x
        y = edge.hexEdge.y
        z = edge.hexEdge.z
        vertices = []

        if z == 0:
            vertices.append(self.find_vertex_index_by_coordinates(x, y, 0))
            vertices.append(self.find_vertex_index_by_coordinates(x, y - 1, 1))
        if z == 1:
            vertices.append(self.find_vertex_index_by_coordinates(x, y - 1, 1))
            vertices.append(
                self.find_vertex_index_by_coordinates(x - 1, y + 1, 0))
        if z == 2:
            vertices.append(
                self.find_vertex_index_by_coordinates(x - 1, y + 1, 0))
            vertices.append(self.find_vertex_index_by_coordinates(x, y, 1))

        return vertices

    def get_other_vertex_next_to_edge(self, edge, vertex):
        vertices = self.get_vertices_next_to_edge(edge)
        for index in vertices:
            if self.vertices[index] != vertex:
                return index
        return None

    # TODO: Store tiles in a smarter way to make this a O(1) function
    def find_tile_by_coordinates(self, x, y):
        for tile in self.tiles:
            if tile.hexFace.x == x and tile.hexFace.y == y:
                return tile
        return None
    
    def find_vertex_production_by_index(self, vertex_index):
        prod = 0
        tile_indices = self.vertex_tiles[vertex_index]
        for tile_index in tile_indices:
            tile = self.tiles[tile_index]
            if tile is None:
                continue
            if not hasattr(tile, '_diceProbability'):
                continue
            prod += tile._diceProbability
        return prod

    def find_vertex_production_per_resource_by_index(self, vertex_index):
        print("find_vertex_production_per_resource_by_index(self, {0})".format(vertex_index))

        prod = {
            Resources.WOOD: 0,
            Resources.BRICK: 0,
            Resources.SHEEP: 0,
            Resources.WHEAT: 0,
            Resources.ORE: 0
        }

        tile_indices = self.vertex_tiles[vertex_index]

        for tile_index in tile_indices:
            tile = self.tiles[tile_index]
            if tile is None:
                continue
            if not hasattr(tile, '_diceProbability'):
                continue
            prod[Resources(tile.tileType)] += tile._diceProbability

        print(prod)
        return prod
