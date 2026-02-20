"""Tests for bridge/state_translator.py - shadow state management."""
import pytest
from unittest.mock import MagicMock, patch

from catanatron.models.player import Color
from catanatron.game import Game

from bridge.state_translator import StateTranslator, TurnPhase, BuildEvent, CATANATRON_COLORS
from bridge.config import COLONIST_ACTION_STATE_SETUP_SETTLEMENT


class TestSetMyColor:
    def test_set_my_color_assigns_color_index(self):
        t = StateTranslator()
        t.set_my_color(1)
        assert t.state.my_color_index == 1

    def test_set_my_color_then_player_order_assigns_catan_color(self):
        t = StateTranslator()
        t.set_my_color(2)
        t.set_player_order([3, 2, 1, 4])
        assert t.state.my_catan_color is not None

    def test_set_player_order_populates_color_maps(self):
        t = StateTranslator()
        t.set_my_color(3)
        t.set_player_order([3, 1, 2, 4])
        assert 3 in t.state.colonist_to_catan_color
        my_catan = t.state.colonist_to_catan_color[3]
        assert t.state.catanatron_to_colonist_color[my_catan] == 3


class TestSetupBoard:
    def _make_minimal_board(self):
        """Create minimal board data for testing."""
        tile_coords = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
            (1, 3), (2, 3), (3, 3), (4, 3),
            (2, 4), (3, 4), (4, 4),
        ]
        resources = [1, 1, 1, 1, 2, 2, 2, 3, 3, 0, 3, 3, 4, 4, 4, 4, 5, 5, 5]
        dice_nums = [5, 2, 6, 8, 10, 9, 12, 11, 4, 0, 8, 10, 9, 4, 5, 6, 3, 11, 3]

        tiles = [
            {
                "hexFace": {"x": x, "y": y},
                "tileType": resources[i],
                "diceNum": dice_nums[i],
            }
            for i, (x, y) in enumerate(tile_coords)
        ]

        vertex_set = set()
        for x, y in tile_coords:
            for dx, dy, z in [(0, 0, 0), (1, -1, 1), (0, -1, 1), (0, 0, 1), (0, 1, 0), (-1, 1, 0)]:
                vertex_set.add((x + dx, y + dy, z))

        vertices = [
            {"hexCorner": {"x": vx, "y": vy, "z": vz}, "owner": -1, "buildingType": 0, "harborType": 0}
            for (vx, vy, vz) in sorted(vertex_set)
        ]

        def edge_verts(ex, ey, ez):
            if ez == 0: return (ex, ey, 0), (ex, ey - 1, 1)
            elif ez == 1: return (ex, ey - 1, 1), (ex - 1, ey + 1, 0)
            elif ez == 2: return (ex - 1, ey + 1, 0), (ex, ey, 1)

        edge_set = set()
        for (vx, vy, vz) in vertex_set:
            if vz == 0:
                edge_set.add((vx, vy, 0))
                edge_set.add((vx + 1, vy - 1, 1))
                edge_set.add((vx + 1, vy - 1, 2))
            else:
                edge_set.add((vx, vy + 1, 0))
                edge_set.add((vx, vy + 1, 1))
                edge_set.add((vx, vy, 2))

        edges = []
        for (ex, ey, ez) in sorted(edge_set):
            v1, v2 = edge_verts(ex, ey, ez)
            if v1 in vertex_set and v2 in vertex_set:
                edges.append({"hexEdge": {"x": ex, "y": ey, "z": ez}, "owner": -1})

        return tiles, vertices, edges

    def test_setup_board_creates_mapper(self):
        t = StateTranslator()
        tiles, vertices, edges = self._make_minimal_board()
        t.setup_board(tiles, vertices, edges)
        assert t.state.mapper is not None

    def test_setup_board_sets_robber_coordinate(self):
        t = StateTranslator()
        tiles, vertices, edges = self._make_minimal_board()
        t.setup_board(tiles, vertices, edges)
        # Robber should be placed on desert tile
        assert t.state.robber_coord is not None


class TestUpdateResources:
    def test_update_resources_from_distribution(self):
        t = StateTranslator()
        t.set_my_color(1)
        t.update_resources_from_distribution([
            {"owner": 1, "card": 1},  # WOOD for us
            {"owner": 2, "card": 2},  # BRICK for opponent
        ])
        assert t.state.my_resources["WOOD"] == 1
        assert t.state.my_resources["BRICK"] == 0

    def test_update_resources_accumulates(self):
        t = StateTranslator()
        t.set_my_color(1)
        t.update_resources_from_distribution([{"owner": 1, "card": 4}])
        t.update_resources_from_distribution([{"owner": 1, "card": 4}])
        assert t.state.my_resources["WHEAT"] == 2

    def test_update_resources_from_trade_giving(self):
        t = StateTranslator()
        t.set_my_color(1)
        t.state.my_resources["WOOD"] = 3
        t.update_resources_from_trade(1, 2, [1, 1], [3])  # give 2 WOOD, get 1 SHEEP
        assert t.state.my_resources["WOOD"] == 1
        assert t.state.my_resources["SHEEP"] == 1

    def test_update_resources_from_trade_receiving(self):
        t = StateTranslator()
        t.set_my_color(2)
        t.state.my_resources["WHEAT"] = 2
        t.update_resources_from_trade(1, 2, [3, 3], [4, 4])  # give 2 SHEEP, receive 2 WHEAT
        assert t.state.my_resources["SHEEP"] == 2
        assert t.state.my_resources["WHEAT"] == 0


class TestDiscoverPlayer:
    def test_discover_player_assigns_colors_in_order(self):
        t = StateTranslator()
        t._discover_player(1, seating_index=0)
        t._discover_player(2)
        t._discover_player(3)

        assert t.state.colonist_to_catan_color[1] == CATANATRON_COLORS[0]
        assert t.state.colonist_to_catan_color[2] == CATANATRON_COLORS[1]
        assert t.state.colonist_to_catan_color[3] == CATANATRON_COLORS[2]

    def test_discover_player_idempotent(self):
        t = StateTranslator()
        t._discover_player(5, seating_index=0)
        t._discover_player(5, seating_index=0)  # Second call is no-op
        assert len(t.state.player_order) == 1


class TestBuildHistory:
    def test_build_history_tracks_settlement(self):
        t = StateTranslator()
        t.set_my_color(1)
        # is_setup_phase defaults to True
        t.update_vertex(1, 2, 0, 1, 1)  # settlement
        assert len(t.state.build_history) == 1
        assert t.state.build_history[0].building_type == "SETTLEMENT"
        assert t.state.build_history[0].is_setup is True

    def test_build_history_tracks_road(self):
        t = StateTranslator()
        t.set_my_color(1)
        # is_setup_phase defaults to True
        t.update_edge(1, 2, 1, 1)
        assert len(t.state.build_history) == 1
        assert t.state.build_history[0].building_type == "ROAD"

    def test_build_history_tracks_city(self):
        t = StateTranslator()
        t.set_my_color(1)
        t.state.is_setup_phase = False  # main game
        t.update_vertex(1, 2, 0, 1, 2)  # city
        assert len(t.state.build_history) == 1
        assert t.state.build_history[0].building_type == "CITY"


class TestPendingRoads:
    def test_pending_road_added(self):
        t = StateTranslator()
        t.add_pending_road((10, 20))
        assert (10, 20) in t.state.pending_roads

    def test_pending_road_cleared_on_edge_update(self):
        t = StateTranslator()
        t.set_my_color(1)
        # Mock mapper with edge mapping
        mapper = MagicMock()
        mapper.colonist_edge_to_catan = {(1, 2, 1): (10, 20)}
        t.state.mapper = mapper
        t.add_pending_road((10, 20))
        t.update_edge(1, 2, 1, 1)
        assert (10, 20) not in t.state.pending_roads


class TestReconstructGame:
    def _setup_basic_state(self, translator):
        """Set up a basic state for reconstruction tests using real colonist.io coords."""
        from tests.test_coordinate_map import make_standard_colonist_board
        tiles, vertices, edges = make_standard_colonist_board()

        translator.set_my_color(1)
        translator.set_player_order([1, 2, 3, 4])
        translator.setup_board(tiles, vertices, edges)
        translator.update_turn_state(
            current_turn_state=1,
            current_action_state=COLONIST_ACTION_STATE_SETUP_SETTLEMENT,
            current_turn_player_color=1,
        )

    def test_reconstruct_game_returns_valid_game(self):
        t = StateTranslator()
        self._setup_basic_state(t)
        game = t.reconstruct_game()
        assert game is not None
        assert isinstance(game, Game)

    def test_reconstruct_game_has_playable_actions(self):
        t = StateTranslator()
        self._setup_basic_state(t)
        game = t.reconstruct_game()
        assert game is not None
        assert game.playable_actions is not None

    def test_reconstruct_game_bypasses_random_sample(self):
        """reconstruct_game() must produce valid results and NOT call State.__init__
        which would call random.sample. We verify by checking the game is valid."""
        import random
        t = StateTranslator()
        self._setup_basic_state(t)

        # Track calls to catanatron state random.sample
        calls = []
        original_sample = random.sample

        def tracking_sample(population, k, **kwargs):
            calls.append(k)
            return original_sample(population, k, **kwargs)

        with patch("catanatron.state.random.sample", side_effect=tracking_sample):
            game = t.reconstruct_game()

        # The game should be created successfully
        assert game is not None
        # random.sample for player ordering should NOT be called
        # (it's called in State.__init__ with k=len(players))
        # Since we use State([], None, initialize=False), no random.sample for players
        player_shuffles = [k for k in calls if k == 4]  # 4-player shuffle
        assert len(player_shuffles) == 0, (
            f"random.sample was called {len(player_shuffles)} times for 4-player shuffle, "
            "suggesting State.__init__ was called with players"
        )

    def test_reconstruct_game_with_settlement(self):
        t = StateTranslator()
        self._setup_basic_state(t)

        # Add a settlement to build history
        mapper = t.state.mapper
        if mapper and mapper.colonist_vertex_to_catan:
            first_coord = list(mapper.colonist_vertex_to_catan.keys())[0]
            t.state.is_setup_phase = True  # Mark as setup
            t.update_vertex(first_coord[0], first_coord[1], first_coord[2], 1, 1)

        game = t.reconstruct_game()
        assert game is not None
