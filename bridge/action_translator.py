"""Action translator: converts Catanatron Actions to colonist.io JSON commands.

DISCARD is handled exclusively via translate_discard() method.
translate() raises ValueError if a DISCARD action is passed to it.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from catanatron.models.enums import ActionType

from bridge.coordinate_map import CoordinateMapper
from bridge.config import (
    CATAN_RESOURCE_TO_COLONIST,
    ACTION_BUILD_ROAD,
    ACTION_BUILD_SETTLEMENT,
    ACTION_BUILD_CITY,
    ACTION_BUY_DEV_CARD,
    ACTION_THROW_DICE,
    ACTION_PASS_TURN,
    ACTION_ACCEPT_TRADE,
    ACTION_REJECT_TRADE,
    ACTION_MOVE_ROBBER,
    ACTION_ROB_PLAYER,
    ACTION_SELECT_CARDS,
    ACTION_CREATE_TRADE,
    ACTION_PLAY_DEV_CARD,
    COLONIST_DEVCARD_VALUES,
    COLONIST_RESOURCE_TO_CATAN,
)

logger = logging.getLogger(__name__)


class ActionTranslator:
    """Converts Catanatron Action namedtuples to colonist.io JSON commands.

    Args:
        mapper: Coordinate mapper for translating node/edge IDs.
        catan_to_colonist_color: Mapping from Catanatron Color -> colonist.io color int.
    """

    def __init__(
        self,
        mapper: CoordinateMapper,
        catan_to_colonist_color: Dict,
    ):
        self.mapper = mapper
        self.catan_to_colonist_color = catan_to_colonist_color

    def translate(self, action) -> List[Dict]:
        """Translate a Catanatron Action to a list of colonist.io JSON commands.

        Most actions produce a single command. MOVE_ROBBER with a steal target
        produces two commands (action:8 then action:9).

        Dev card plays that need follow-up (monopoly, year_of_plenty) produce
        two commands: play_dev_card + select_cards.

        Args:
            action: Catanatron Action namedtuple (color, action_type, value)

        Returns:
            List of JSON dicts to send to the userscript.

        Raises:
            ValueError: If action_type is DISCARD (must use translate_discard()).
            ValueError: If action_type is unrecognized.
        """
        action_type = action.action_type

        if action_type == ActionType.DISCARD:
            raise ValueError(
                "DISCARD actions must be handled via translate_discard(), not translate()"
            )

        if action_type == ActionType.ROLL:
            return [{"action": ACTION_THROW_DICE}]

        if action_type == ActionType.END_TURN:
            return [{"action": ACTION_PASS_TURN}]

        if action_type == ActionType.BUILD_SETTLEMENT:
            node_id = action.value
            col_coord = self.mapper.catan_vertex_to_colonist.get(node_id)
            if col_coord is None:
                raise ValueError(f"Unknown node_id for settlement: {node_id}")
            vertex_index = self._find_vertex_index(col_coord)
            return [{"action": ACTION_BUILD_SETTLEMENT, "data": vertex_index}]

        if action_type == ActionType.BUILD_ROAD:
            edge = action.value
            col_coord = self.mapper.catan_edge_to_colonist.get(frozenset(edge))
            if col_coord is None:
                # Try reversed edge
                col_coord = self.mapper.catan_edge_to_colonist.get(frozenset((edge[1], edge[0])))
            if col_coord is None:
                raise ValueError(f"Unknown edge for road: {edge}")
            edge_index = self._find_edge_index(col_coord)
            return [{"action": ACTION_BUILD_ROAD, "data": edge_index}]

        if action_type == ActionType.BUILD_CITY:
            node_id = action.value
            col_coord = self.mapper.catan_vertex_to_colonist.get(node_id)
            if col_coord is None:
                raise ValueError(f"Unknown node_id for city: {node_id}")
            vertex_index = self._find_vertex_index(col_coord)
            return [{"action": ACTION_BUILD_CITY, "data": vertex_index}]

        if action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return [{"action": ACTION_BUY_DEV_CARD}]

        if action_type == ActionType.MOVE_ROBBER:
            cube_coord, steal_color = action.value
            tile_index = self._find_tile_index_by_cube(cube_coord)
            commands = [{"action": ACTION_MOVE_ROBBER, "data": tile_index}]
            if steal_color is not None:
                colonist_color = self.catan_to_colonist_color.get(steal_color)
                if colonist_color is not None:
                    commands.append({"action": ACTION_ROB_PLAYER, "data": colonist_color})
            return commands

        if action_type == ActionType.MARITIME_TRADE:
            return self._translate_maritime_trade(action.value)

        if action_type == ActionType.PLAY_KNIGHT_CARD:
            return [{"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["KNIGHT"]}]

        if action_type == ActionType.PLAY_MONOPOLY:
            resource = action.value
            resource_val = CATAN_RESOURCE_TO_COLONIST.get(resource, 0)
            return [
                {"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["MONOPOLY"]},
                {"action": ACTION_SELECT_CARDS, "data": [resource_val]},
            ]

        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            r1, r2 = action.value
            r1_val = CATAN_RESOURCE_TO_COLONIST.get(r1, 0)
            r2_val = CATAN_RESOURCE_TO_COLONIST.get(r2, 0)
            return [
                {"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["YEAR_OF_PLENTY"]},
                {"action": ACTION_SELECT_CARDS, "data": [r1_val, r2_val]},
            ]

        if action_type == ActionType.PLAY_ROAD_BUILDING:
            return [{"action": ACTION_PLAY_DEV_CARD, "data": COLONIST_DEVCARD_VALUES["ROAD_BUILDING"]}]

        if action_type == ActionType.ACCEPT_TRADE:
            trade_id = action.value if action.value is not None else "0"
            return [{"action": ACTION_ACCEPT_TRADE, "data": str(trade_id)}]

        if action_type == ActionType.REJECT_TRADE:
            trade_id = action.value if action.value is not None else "0"
            return [{"action": ACTION_REJECT_TRADE, "data": str(trade_id)}]

        raise ValueError(f"Unrecognized action type: {action_type}")

    def translate_discard(self, resources_to_discard: List[str]) -> Dict:
        """Translate a discard decision to colonist.io SELECT_CARDS command.

        Args:
            resources_to_discard: List of Catanatron resource strings to discard.

        Returns:
            JSON dict for the select_cards command.
        """
        card_values = [
            CATAN_RESOURCE_TO_COLONIST[r]
            for r in resources_to_discard
            if r in CATAN_RESOURCE_TO_COLONIST
        ]
        return {"action": ACTION_SELECT_CARDS, "data": card_values}

    def translate_monopoly_selection(self, resource: str) -> Dict:
        """Translate a monopoly resource selection to colonist.io SELECT_CARDS command."""
        resource_val = CATAN_RESOURCE_TO_COLONIST.get(resource, 0)
        return {"action": ACTION_SELECT_CARDS, "data": [resource_val]}

    def translate_year_of_plenty_selection(self, r1: str, r2: str) -> Dict:
        """Translate year of plenty resource selection to colonist.io SELECT_CARDS command."""
        r1_val = CATAN_RESOURCE_TO_COLONIST.get(r1, 0)
        r2_val = CATAN_RESOURCE_TO_COLONIST.get(r2, 0)
        return {"action": ACTION_SELECT_CARDS, "data": [r1_val, r2_val]}

    def _translate_maritime_trade(self, trade_value: Tuple) -> List[Dict]:
        """Translate a MARITIME_TRADE 5-tuple to colonist.io CREATE_TRADE command.

        The 5-tuple is: (resource, resource, resource|None, resource|None, wanted_resource)
        where None values represent placeholders in port trades.
        Resources 0-3 are offered, resource 4 is wanted.
        """
        # trade_value is a tuple of 5 resources where last is wanted
        # Offered: first 4 (with Nones for port trades), wanted: last
        offered = [r for r in trade_value[:4] if r is not None]
        wanted_resource = trade_value[4]

        offered_vals = [CATAN_RESOURCE_TO_COLONIST.get(r, 0) for r in offered]
        wanted_vals = [CATAN_RESOURCE_TO_COLONIST.get(wanted_resource, 0)]

        return [{
            "action": ACTION_CREATE_TRADE,
            "data": {
                "offered": offered_vals,
                "wanted": wanted_vals,
                "bankTrade": True,
            }
        }]

    def _find_vertex_index(self, col_coord: Tuple) -> int:
        """Find the colonist.io vertex list index for a vertex coord.

        Raises ValueError if the coordinate is not in the index map.
        """
        idx = getattr(self, '_vertex_coord_to_idx', {}).get(col_coord)
        if idx is None:
            raise ValueError(
                f"Vertex coordinate {col_coord} not found in vertex index map. "
                "Ensure set_vertex_index_map() was called with board setup data."
            )
        return idx

    def _find_edge_index(self, col_coord: Tuple) -> int:
        """Find the colonist.io edge list index for an edge coord.

        Raises ValueError if the coordinate is not in the index map.
        """
        idx = getattr(self, '_edge_coord_to_idx', {}).get(col_coord)
        if idx is None:
            raise ValueError(
                f"Edge coordinate {col_coord} not found in edge index map. "
                "Ensure set_edge_index_map() was called with board setup data."
            )
        return idx

    def _find_tile_index_by_cube(self, cube_coord: Tuple) -> int:
        """Find colonist.io tile list index from Catanatron cube coordinate.

        Raises ValueError if no matching tile is found.
        """
        for xy, catan_tile in self.mapper.colonist_tile_to_catan.items():
            if catan_tile is None:
                continue
            for coord, tile in self.mapper.catan_map.land_tiles.items():
                if tile.id == catan_tile.id and coord == cube_coord:
                    tile_idx = self.mapper.colonist_xy_to_tile_index.get(xy)
                    if tile_idx is None:
                        raise ValueError(
                            f"Tile at colonist {xy} not in colonist_xy_to_tile_index"
                        )
                    return tile_idx
        raise ValueError(f"Could not find tile index for cube coord {cube_coord}")

    def set_vertex_index_map(self, vertex_coord_to_idx: Dict) -> None:
        """Set the vertex coordinate to index mapping (from board setup)."""
        self._vertex_coord_to_idx = vertex_coord_to_idx

    def set_edge_index_map(self, edge_coord_to_idx: Dict) -> None:
        """Set the edge coordinate to index mapping (from board setup)."""
        self._edge_coord_to_idx = edge_coord_to_idx
