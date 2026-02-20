"""Tests for bridge/action_translator.py - action translation."""
import pytest
from unittest.mock import MagicMock, patch

from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color

from bridge.action_translator import ActionTranslator
from bridge.config import (
    ACTION_BUILD_ROAD,
    ACTION_BUILD_SETTLEMENT,
    ACTION_BUILD_CITY,
    ACTION_BUY_DEV_CARD,
    ACTION_THROW_DICE,
    ACTION_PASS_TURN,
    ACTION_MOVE_ROBBER,
    ACTION_ROB_PLAYER,
    ACTION_SELECT_CARDS,
    ACTION_CREATE_TRADE,
    ACTION_PLAY_DEV_CARD,
    COLONIST_DEVCARD_VALUES,
)


def make_translator():
    """Create an ActionTranslator with mock mapper."""
    mapper = MagicMock()
    mapper.catan_vertex_to_colonist = {
        10: (1, 2, 0),
        20: (3, 4, 1),
        15: (2, 3, 0),
    }
    mapper.catan_edge_to_colonist = {
        frozenset([10, 20]): (1, 2, 1),
        frozenset([15, 20]): (2, 3, 2),
    }
    mapper.colonist_tile_to_catan = {}
    mapper.catan_map = MagicMock()
    mapper.catan_map.land_tiles = {}
    mapper.colonist_xy_to_tile_index = {}
    mapper.colonist_tile_index_to_xy = {}

    color_map = {
        Color.RED: 1,
        Color.BLUE: 2,
        Color.ORANGE: 3,
        Color.WHITE: 4,
    }

    translator = ActionTranslator(mapper, color_map)
    translator.set_vertex_index_map({
        (1, 2, 0): 5,
        (3, 4, 1): 12,
        (2, 3, 0): 8,
    })
    translator.set_edge_index_map({
        (1, 2, 1): 7,
        (2, 3, 2): 14,
    })
    return translator


class TestTranslateRoll:
    def test_translate_roll(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.ROLL, None)
        result = t.translate(action)
        assert result == [{"action": ACTION_THROW_DICE}]


class TestTranslateEndTurn:
    def test_translate_end_turn(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.END_TURN, None)
        result = t.translate(action)
        assert result == [{"action": ACTION_PASS_TURN}]


class TestTranslateBuildSettlement:
    def test_translate_build_settlement(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 10)
        result = t.translate(action)
        assert len(result) == 1
        assert result[0]["action"] == ACTION_BUILD_SETTLEMENT
        assert result[0]["data"] == 5  # vertex index for (1,2,0)

    def test_translate_build_settlement_unknown_node_raises(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 999)
        with pytest.raises(ValueError):
            t.translate(action)


class TestTranslateBuildRoad:
    def test_translate_build_road(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_ROAD, (10, 20))
        result = t.translate(action)
        assert len(result) == 1
        assert result[0]["action"] == ACTION_BUILD_ROAD
        assert result[0]["data"] == 7  # edge index for (1,2,1)

    def test_translate_build_road_reversed_edge(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_ROAD, (20, 10))
        result = t.translate(action)
        assert result[0]["action"] == ACTION_BUILD_ROAD

    def test_translate_build_road_unknown_edge_raises(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_ROAD, (99, 100))
        with pytest.raises(ValueError):
            t.translate(action)


class TestTranslateBuildCity:
    def test_translate_build_city(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUILD_CITY, 20)
        result = t.translate(action)
        assert len(result) == 1
        assert result[0]["action"] == ACTION_BUILD_CITY
        assert result[0]["data"] == 12  # vertex index for (3,4,1)


class TestTranslateBuyDevCard:
    def test_translate_buy_dev_card(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
        result = t.translate(action)
        assert result == [{"action": ACTION_BUY_DEV_CARD}]


class TestTranslateMoveRobber:
    def test_translate_move_robber_no_steal(self):
        t = make_translator()
        # Setup a tile mapping
        mock_tile = MagicMock()
        mock_tile.id = 5
        t.mapper.colonist_tile_to_catan = {(2, 2): mock_tile}
        mock_land_tile = MagicMock()
        mock_land_tile.id = 5
        t.mapper.catan_map.land_tiles = {(0, 0, 0): mock_land_tile}
        t.mapper.colonist_xy_to_tile_index = {(2, 2): 9}

        action = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))
        result = t.translate(action)
        assert len(result) == 1
        assert result[0]["action"] == ACTION_MOVE_ROBBER

    def test_translate_move_robber_with_steal(self):
        t = make_translator()
        mock_tile = MagicMock()
        mock_tile.id = 5
        t.mapper.colonist_tile_to_catan = {(2, 2): mock_tile}
        mock_land_tile = MagicMock()
        mock_land_tile.id = 5
        t.mapper.catan_map.land_tiles = {(0, 0, 0): mock_land_tile}
        t.mapper.colonist_xy_to_tile_index = {(2, 2): 9}

        action = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE))
        result = t.translate(action)
        assert len(result) == 2
        assert result[0]["action"] == ACTION_MOVE_ROBBER
        assert result[1]["action"] == ACTION_ROB_PLAYER
        assert result[1]["data"] == 2  # colonist color for BLUE


class TestTranslateMaritimeTrade:
    def test_translate_maritime_trade_4_to_1(self):
        """4:1 trade: offer 4 WOOD, want 1 ORE."""
        t = make_translator()
        action = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "ORE"))
        result = t.translate(action)
        assert len(result) == 1
        assert result[0]["action"] == ACTION_CREATE_TRADE
        assert result[0]["data"]["offered"] == [1, 1, 1, 1]  # 4 WOOD
        assert result[0]["data"]["wanted"] == [5]  # 1 ORE

    def test_translate_maritime_trade_port_2_to_1(self):
        """2:1 port trade: offer 2 ORE (None placeholders), want 1 WHEAT."""
        t = make_translator()
        action = Action(Color.RED, ActionType.MARITIME_TRADE, ("ORE", "ORE", None, None, "WHEAT"))
        result = t.translate(action)
        assert result[0]["action"] == ACTION_CREATE_TRADE
        assert 5 in result[0]["data"]["offered"]  # ORE = 5
        assert result[0]["data"]["wanted"] == [4]  # WHEAT = 4


class TestTranslatePlayKnight:
    def test_translate_play_knight(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
        result = t.translate(action)
        assert result == [{"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["KNIGHT"]}]


class TestTranslatePlayMonopoly:
    def test_translate_play_monopoly(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.PLAY_MONOPOLY, "WHEAT")
        result = t.translate(action)
        assert len(result) == 2
        assert result[0]["action"] == ACTION_PLAY_DEV_CARD
        assert result[0]["data"] == COLONIST_DEVCARD_VALUES["MONOPOLY"]
        assert result[1]["action"] == ACTION_SELECT_CARDS
        assert result[1]["data"] == [4]  # WHEAT = 4


class TestTranslatePlayYearOfPlenty:
    def test_translate_play_year_of_plenty(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("ORE", "WHEAT"))
        result = t.translate(action)
        assert len(result) == 2
        assert result[0]["action"] == ACTION_PLAY_DEV_CARD
        assert result[0]["data"] == COLONIST_DEVCARD_VALUES["YEAR_OF_PLENTY"]
        assert result[1]["action"] == ACTION_SELECT_CARDS
        assert 5 in result[1]["data"]  # ORE = 5
        assert 4 in result[1]["data"]  # WHEAT = 4


class TestTranslatePlayRoadBuilding:
    def test_translate_play_road_building(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
        result = t.translate(action)
        assert result == [{"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["ROAD_BUILDING"]}]


class TestTranslateDiscard:
    def test_translate_discard_via_translate_raises_error(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.DISCARD, None)
        with pytest.raises(ValueError, match="translate_discard"):
            t.translate(action)

    def test_translate_discard_via_translate_discard_method(self):
        t = make_translator()
        result = t.translate_discard(["WOOD", "BRICK", "SHEEP"])
        assert result["action"] == ACTION_SELECT_CARDS
        assert 1 in result["data"]  # WOOD = 1
        assert 2 in result["data"]  # BRICK = 2
        assert 3 in result["data"]  # SHEEP = 3


class TestTranslateAcceptRejectTrade:
    def test_translate_accept_trade(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.ACCEPT_TRADE, "trade-123")
        result = t.translate(action)
        assert result == [{"action": 6, "data": "trade-123"}]

    def test_translate_reject_trade(self):
        t = make_translator()
        action = Action(Color.RED, ActionType.REJECT_TRADE, "trade-456")
        result = t.translate(action)
        assert result == [{"action": 7, "data": "trade-456"}]


class TestTranslateHelpers:
    def test_translate_discard_empty_list(self):
        t = make_translator()
        result = t.translate_discard([])
        assert result["action"] == ACTION_SELECT_CARDS
        assert result["data"] == []

    def test_translate_monopoly_selection(self):
        t = make_translator()
        result = t.translate_monopoly_selection("ORE")
        assert result["action"] == ACTION_SELECT_CARDS
        assert result["data"] == [5]

    def test_translate_year_of_plenty_selection(self):
        t = make_translator()
        result = t.translate_year_of_plenty_selection("WOOD", "BRICK")
        assert result["action"] == ACTION_SELECT_CARDS
        assert 1 in result["data"]
        assert 2 in result["data"]


class TestMissingIndexMaps:
    """Verify ValueError is raised when vertex/edge index maps are missing."""

    def test_vertex_index_missing_raises(self):
        """Settlement with vertex not in index map should raise ValueError."""
        t = make_translator()
        t.set_vertex_index_map({})  # Empty map
        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 10)
        with pytest.raises(ValueError, match="Vertex coordinate"):
            t.translate(action)

    def test_edge_index_missing_raises(self):
        """Road with edge not in index map should raise ValueError."""
        t = make_translator()
        t.set_edge_index_map({})  # Empty map
        action = Action(Color.RED, ActionType.BUILD_ROAD, (10, 20))
        with pytest.raises(ValueError, match="Edge coordinate"):
            t.translate(action)

    def test_vertex_index_not_set_raises(self):
        """Settlement without calling set_vertex_index_map should raise ValueError."""
        mapper = MagicMock()
        mapper.catan_vertex_to_colonist = {10: (1, 2, 0)}
        mapper.catan_edge_to_colonist = {}
        translator = ActionTranslator(mapper, {})
        action = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 10)
        with pytest.raises(ValueError, match="Vertex coordinate"):
            translator.translate(action)

    def test_tile_index_missing_raises(self):
        """Move robber with unknown cube coord should raise ValueError."""
        t = make_translator()
        action = Action(Color.RED, ActionType.MOVE_ROBBER, ((99, 99, -198), None))
        with pytest.raises(ValueError, match="Could not find tile index"):
            t.translate(action)
