"""Bot interface adapter wrapping Catanatron's bot players.

Provides decide(), decide_discard(), and decide_trade_response() methods
that work with the bridge's shadow state.
"""
import logging
import random
from typing import Dict, List, Optional

from catanatron.models.enums import ActionType, RESOURCES
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

logger = logging.getLogger(__name__)

# Resource priority for discard decisions (keep high-value resources)
# Lower score = more likely to discard
RESOURCE_PRIORITY = {
    "ORE": 5,
    "WHEAT": 4,
    "SHEEP": 3,
    "WOOD": 2,
    "BRICK": 1,
}


class BotInterface:
    """Wraps Catanatron bot players for use in the bridge.

    Args:
        bot_type: "value" for ValueFunctionPlayer, "random" for WeightedRandomPlayer.
    """

    def __init__(self, bot_type: str = "value", search_depth: int = 2,
                 blend_weight: float = 1e8,
                 bc_model_path: str = "robottler/models/value_net_v2.pt"):
        self.bot_type = bot_type
        self.search_depth = search_depth
        self.blend_weight = blend_weight
        self.bc_model_path = bc_model_path
        self._player_cache: Dict[Color, object] = {}

    def _get_player(self, color: Color):
        """Get or create a bot player for the given color."""
        if color not in self._player_cache:
            if self.bot_type == "search":
                from robottler.search_player import NeuralSearchPlayer, make_blended_value_fn
                value_fn = make_blended_value_fn(
                    self.bc_model_path, blend_weight=self.blend_weight,
                )
                player = NeuralSearchPlayer(
                    color, depth=self.search_depth, prunning=False,
                    value_fn=value_fn,
                )
                self._player_cache[color] = player
                logger.info(
                    f"Search bot: depth={self.search_depth}, "
                    f"blend_weight={self.blend_weight:.0e}"
                )
            elif self.bot_type == "neural":
                from robottler.value_model import load_value_model
                value_fn = load_value_model("robottler/models/value_net.pt")
                player = ValueFunctionPlayer(color)
                player._neural_value_fn = value_fn
                self._player_cache[color] = player
            elif self.bot_type == "value":
                self._player_cache[color] = ValueFunctionPlayer(color)
            else:
                self._player_cache[color] = WeightedRandomPlayer(color)
        return self._player_cache[color]

    def decide(self, game, our_color: Color):
        """Query the bot for the best action given the current game state.

        Args:
            game: Reconstructed Catanatron Game object.
            our_color: Our Catanatron Color.

        Returns:
            Best action from playable_actions, or None if no actions available.
        """
        if not game.playable_actions:
            logger.warning("No playable actions available")
            return None

        player = self._get_player(our_color)

        try:
            action = player.decide(game, game.playable_actions)
            logger.debug(f"Bot decided: {action.action_type} {action.value}")
            return action
        except Exception as e:
            logger.error(f"Bot decision failed: {e}", exc_info=True)
            # Fallback: return first playable action
            return game.playable_actions[0]

    def decide_discard(
        self,
        my_resources: Dict[str, int],
        num_to_discard: int,
    ) -> List[str]:
        """Determine which resources to discard when hand > 7.

        Uses a simple heuristic: discard lowest-priority resources first.

        Args:
            my_resources: Current resource hand {resource_str: count}.
            num_to_discard: Number of cards to discard.

        Returns:
            List of resource strings to discard (may have duplicates).
        """
        if num_to_discard <= 0:
            return []

        # Build a flat list of resources we have, sorted by priority (lowest first)
        available = []
        for resource, count in my_resources.items():
            for _ in range(count):
                available.append(resource)

        # Sort by priority (ascending = discard first)
        available.sort(key=lambda r: RESOURCE_PRIORITY.get(r, 0))

        to_discard = available[:num_to_discard]
        logger.info(f"Discarding {num_to_discard} cards: {to_discard}")
        return to_discard

    def decide_trade_response(
        self,
        offered_resources: List[int],
        wanted_resources: List[int],
        my_resources: Dict[str, int],
        trade_id: str,
    ) -> bool:
        """Decide whether to accept or reject a trade offer.

        Simple heuristic: accept if we have the offered resources and
        the trade gives us something we have less of.

        Args:
            offered_resources: Resources the other player offers (colonist int values).
            wanted_resources: Resources they want from us (colonist int values).
            my_resources: Our current resource hand.
            trade_id: The trade ID.

        Returns:
            True to accept, False to reject.
        """
        from bridge.config import COLONIST_RESOURCE_TO_CATAN

        # Check if we have the resources they want
        wanted_catan = [COLONIST_RESOURCE_TO_CATAN.get(r) for r in wanted_resources if r in COLONIST_RESOURCE_TO_CATAN]
        for resource in wanted_catan:
            if my_resources.get(resource, 0) < wanted_resources.count(
                next(k for k, v in COLONIST_RESOURCE_TO_CATAN.items() if v == resource)
            ):
                logger.info(f"Rejecting trade {trade_id}: insufficient resources")
                return False

        # Calculate net value of trade
        offered_catan = [COLONIST_RESOURCE_TO_CATAN.get(r) for r in offered_resources if r in COLONIST_RESOURCE_TO_CATAN]

        offered_priority = sum(RESOURCE_PRIORITY.get(r, 0) for r in offered_catan if r)
        wanted_priority = sum(RESOURCE_PRIORITY.get(r, 0) for r in wanted_catan if r)

        if offered_priority >= wanted_priority:
            logger.info(f"Accepting trade {trade_id}: beneficial or neutral")
            return True

        logger.info(f"Rejecting trade {trade_id}: unfavorable")
        return False
