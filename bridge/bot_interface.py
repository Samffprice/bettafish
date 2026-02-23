"""Bot interface adapter wrapping Catanatron's bot players.

Provides decide(), decide_discard(), and decide_trade_response() methods
that work with the bridge's shadow state.

Trade evaluation strategies (set via BotInterface trade_strategy):
  "blend"     — Use the full heuristic+neural blend value function.
                Evaluates game state before/after trade, accepts if our
                position improves. Best accuracy, ~200µs per eval.
  "heuristic" — Hand-crafted scoring: production scarcity, build progress,
                harbor/surplus discounts, opponent threat, discard risk.
                No neural net needed, ~50µs per eval.
"""
import logging
from collections import Counter
from typing import Dict, List, Optional

from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from bridge.config import COLONIST_RESOURCE_TO_CATAN

logger = logging.getLogger(__name__)

# Resource priority for discard decisions (keep high-value resources)
RESOURCE_PRIORITY = {
    "ORE": 5,
    "WHEAT": 4,
    "SHEEP": 3,
    "WOOD": 2,
    "BRICK": 1,
}


# ---------------------------------------------------------------------------
# Trade strategy: "blend" — value function before/after comparison
# ---------------------------------------------------------------------------

def _evaluate_trade_blend(offered_catan, wanted_catan, game, our_color,
                          creator_vps, vps_to_win, bb_value_fn):
    """Evaluate trade by comparing blend value before and after.

    Converts game to bitboard, evaluates, applies trade to resource
    counts, evaluates again. Accepts if our value improves.
    Also applies a match-point hard block.

    Returns float: positive = position improves, negative = worsens.
    """
    from robottler.bitboard.convert import game_to_bitboard
    from robottler.bitboard.state import PS_WOOD
    from robottler.bitboard.masks import RESOURCE_INDEX

    if not offered_catan and not wanted_catan:
        return -1.0

    # Hard block: never trade with someone at match point
    if vps_to_win > 0 and creator_vps >= vps_to_win - 1:
        return -100.0

    bb = game_to_bitboard(game)
    p_idx = bb.color_to_index[our_color]

    # Check we can afford it
    for resource, needed in Counter(wanted_catan).items():
        ri = PS_WOOD + RESOURCE_INDEX[resource]
        if bb.player_state[p_idx][ri] < needed:
            return -100.0

    value_before = bb_value_fn(bb, our_color)

    # Apply trade: modify resource counts in-place
    for r in wanted_catan:
        bb.player_state[p_idx][PS_WOOD + RESOURCE_INDEX[r]] -= 1
    for r in offered_catan:
        bb.player_state[p_idx][PS_WOOD + RESOURCE_INDEX[r]] += 1

    value_after = bb_value_fn(bb, our_color)

    return value_after - value_before


# ---------------------------------------------------------------------------
# Trade strategy: "heuristic" — hand-crafted scoring
# ---------------------------------------------------------------------------

# Build costs and strategic weights
_BUILD_COSTS = {
    "city":       {"WHEAT": 2, "ORE": 3},
    "settlement": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "dev_card":   {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
    "road":       {"WOOD": 1, "BRICK": 1},
}
_BUILD_WEIGHTS = {"city": 3.0, "settlement": 2.5, "dev_card": 1.5, "road": 1.0}


def _compute_production(game, color):
    """Per-resource production probability from settlements/cities."""
    catan_map = game.state.board.map
    buildings = game.state.buildings_by_color.get(color, {})

    robber_coord = game.state.board.robber_coordinate
    robbed_nodes = set()
    if robber_coord in catan_map.tiles:
        robbed_nodes = set(catan_map.tiles[robber_coord].nodes.values())

    production = {}
    for resource in RESOURCES:
        prod = 0.0
        for node_id in buildings.get(SETTLEMENT, []):
            if node_id not in robbed_nodes:
                prod += catan_map.node_production.get(node_id, {}).get(resource, 0.0)
        for node_id in buildings.get(CITY, []):
            if node_id not in robbed_nodes:
                prod += 2 * catan_map.node_production.get(node_id, {}).get(resource, 0.0)
        production[resource] = prod
    return production


def _resource_scarcity_value(resource, production):
    """Value of one unit inversely proportional to our production."""
    return 1.0 / (production.get(resource, 0.0) + 0.1)


def _build_progress_score(current_hand, post_trade_hand):
    """Score improvement in distance-to-next-build."""
    def distance(hand, cost):
        return sum(max(need - hand.get(r, 0), 0) for r, need in cost.items())

    score = 0.0
    for build_name, cost in _BUILD_COSTS.items():
        before = distance(current_hand, cost)
        after = distance(post_trade_hand, cost)
        improvement = before - after
        if improvement > 0:
            weight = _BUILD_WEIGHTS[build_name]
            completion_bonus = 1.5 if after == 0 else 1.0
            score += improvement * weight * completion_bonus
    return score * 0.5


def _harbor_discount(resource, game, color):
    """Multiplier reducing give-cost if we own a port."""
    port_resources = game.state.board.get_player_port_resources(color)
    if resource in port_resources:
        return 0.5
    if None in port_resources:
        return 0.75
    return 1.0


def _surplus_discount(resource, hand):
    """Multiplier reducing give-cost for surplus resources."""
    count = hand.get(resource, 0)
    if count >= 4:
        return 0.5
    elif count >= 3:
        return 0.7
    elif count >= 2:
        return 0.85
    return 1.0


def _opponent_threat_penalty(wanted_catan, creator_vps, vps_to_win):
    """Penalty for trading with an opponent close to winning."""
    if vps_to_win <= 0:
        return 0.0
    if creator_vps >= vps_to_win - 1:
        return 100.0
    proximity = creator_vps / vps_to_win
    if proximity < 0.5:
        return 0.0
    penalty_mult = max(0.0, 2.0 ** (4.0 * (proximity - 0.5)) - 0.5)
    return penalty_mult * len(wanted_catan) * 0.3


def _discard_risk_penalty(current_hand, offered_catan, wanted_catan):
    """Quadratic penalty if trade pushes hand over 7 cards."""
    current_total = sum(current_hand.values())
    post_total = current_total + len(offered_catan) - len(wanted_catan)
    if post_total <= 7:
        return 0.0
    excess = post_total - 7
    return 0.5 * excess * excess


def _evaluate_trade_heuristic(offered_catan, wanted_catan, game, our_color,
                              creator_vps, vps_to_win):
    """Evaluate trade using hand-crafted scoring factors."""
    if not offered_catan and not wanted_catan:
        return -1.0

    p_idx = game.state.color_to_index[our_color]
    current_hand = {}
    for r in RESOURCES:
        current_hand[r] = game.state.player_state.get(f"P{p_idx}_{r}_IN_HAND", 0)

    # Check we can afford it
    for resource, needed in Counter(wanted_catan).items():
        if current_hand.get(resource, 0) < needed:
            return -100.0

    post_hand = dict(current_hand)
    for r in wanted_catan:
        post_hand[r] -= 1
    for r in offered_catan:
        post_hand[r] = post_hand.get(r, 0) + 1

    production = _compute_production(game, our_color)

    gain_value = sum(_resource_scarcity_value(r, production) for r in offered_catan)
    give_value = 0.0
    for r in wanted_catan:
        base = _resource_scarcity_value(r, production)
        base *= _harbor_discount(r, game, our_color)
        base *= _surplus_discount(r, current_hand)
        give_value += base

    progress = _build_progress_score(current_hand, post_hand)
    opp_penalty = _opponent_threat_penalty(wanted_catan, creator_vps, vps_to_win)
    discard_pen = _discard_risk_penalty(current_hand, offered_catan, wanted_catan)

    return (gain_value - give_value) + progress - opp_penalty - discard_pen


# ---------------------------------------------------------------------------
# Bot interface
# ---------------------------------------------------------------------------

class BotInterface:
    """Wraps Catanatron bot players for use in the bridge.

    Args:
        bot_type: Player type for game decisions.
        trade_strategy: "blend" (default) uses the value function to compare
            before/after. "heuristic" uses hand-crafted scoring factors.
    """

    def __init__(self, bot_type: str = "value", search_depth: int = 2,
                 blend_weight: float = 1e8,
                 bc_model_path: str = "robottler/models/value_net_v2.pt",
                 trade_strategy: str = "blend"):
        self.bot_type = bot_type
        self.search_depth = search_depth
        self.blend_weight = blend_weight
        self.bc_model_path = bc_model_path
        self.trade_strategy = trade_strategy
        self._player_cache: Dict[Color, object] = {}
        self._bb_value_fn = None  # lazy-loaded for blend trade eval

    def _get_bb_value_fn(self):
        """Lazy-load the bitboard blend value function for trade evaluation."""
        if self._bb_value_fn is None:
            from robottler.search_player import make_bb_blended_value_fn
            self._bb_value_fn = make_bb_blended_value_fn(
                self.bc_model_path, blend_weight=self.blend_weight,
            )
            logger.info(f"Loaded blend value fn for trade eval "
                        f"(weight={self.blend_weight:.0e})")
        return self._bb_value_fn

    def _get_player(self, color: Color):
        """Get or create a bot player for the given color."""
        if color not in self._player_cache:
            if self.bot_type == "search":
                from robottler.search_player import BitboardSearchPlayer, make_bb_blended_value_fn
                bb_value_fn = make_bb_blended_value_fn(
                    self.bc_model_path, blend_weight=self.blend_weight,
                )
                player = BitboardSearchPlayer(
                    color, depth=self.search_depth, prunning=False,
                    bb_value_fn=bb_value_fn, dice_sample_size=5,
                )
                self._player_cache[color] = player
                # Reuse for trade eval too
                self._bb_value_fn = bb_value_fn
                logger.info(
                    f"Bitboard search bot: depth={self.search_depth}, "
                    f"blend_weight={self.blend_weight:.0e}, dice_sample=5"
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
        """Query the bot for the best action given the current game state."""
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
            return game.playable_actions[0]

    def decide_discard(
        self,
        my_resources: Dict[str, int],
        num_to_discard: int,
    ) -> List[str]:
        """Determine which resources to discard when hand > 7."""
        if num_to_discard <= 0:
            return []

        available = []
        for resource, count in my_resources.items():
            for _ in range(count):
                available.append(resource)

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
        game=None,
        our_color=None,
        creator_vps: int = 0,
        vps_to_win: int = 10,
    ) -> bool:
        """Decide whether to accept or reject a trade offer.

        Uses the configured trade_strategy when game state is available.
        Falls back to simple priority heuristic when game is None.
        """
        offered_catan = [COLONIST_RESOURCE_TO_CATAN[r]
                         for r in offered_resources
                         if r in COLONIST_RESOURCE_TO_CATAN]
        wanted_catan = [COLONIST_RESOURCE_TO_CATAN[r]
                        for r in wanted_resources
                        if r in COLONIST_RESOURCE_TO_CATAN]

        # Smart evaluation with full game state
        if game is not None and our_color is not None:
            try:
                if self.trade_strategy == "blend":
                    bb_vf = self._get_bb_value_fn()
                    score = _evaluate_trade_blend(
                        offered_catan, wanted_catan, game, our_color,
                        creator_vps, vps_to_win, bb_vf,
                    )
                else:
                    score = _evaluate_trade_heuristic(
                        offered_catan, wanted_catan, game, our_color,
                        creator_vps, vps_to_win,
                    )
                accept = score > 0.0
                logger.info(
                    f"Trade {trade_id} [{self.trade_strategy}]: score={score:.4f} → "
                    f"{'ACCEPT' if accept else 'REJECT'}  "
                    f"offered={offered_catan} wanted={wanted_catan}"
                )
                return accept
            except Exception as e:
                logger.error(f"Trade eval failed ({self.trade_strategy}): {e}",
                             exc_info=True)

        # Fallback: simple priority-based heuristic
        give_counts = Counter(wanted_catan)
        for resource in wanted_catan:
            if my_resources.get(resource, 0) < give_counts[resource]:
                logger.info(f"Rejecting trade {trade_id}: insufficient resources")
                return False

        offered_priority = sum(RESOURCE_PRIORITY.get(r, 0) for r in offered_catan if r)
        wanted_priority = sum(RESOURCE_PRIORITY.get(r, 0) for r in wanted_catan if r)

        if offered_priority >= wanted_priority:
            logger.info(f"Accepting trade {trade_id}: beneficial (fallback)")
            return True

        logger.info(f"Rejecting trade {trade_id}: unfavorable (fallback)")
        return False
