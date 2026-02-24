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

from bridge.config import COLONIST_RESOURCE_TO_CATAN, CATAN_RESOURCE_TO_COLONIST

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

# Opponent-aware trade weights (tunable via dataset comparison)
OPP_WEIGHT = 0.5       # How much we penalize opponent gain relative to ours
LOOKAHEAD_WEIGHT = 0.3  # Partial weight on opponent's best immediate build


def _opponent_new_build_types(bb, opp_idx):
    """Check what build types the opponent can afford in a given state.

    Returns a set of ActionType values for builds the opponent can make.
    Uses direct resource checks (no movegen) for speed.
    """
    from robottler.bitboard.state import PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE
    ps = bb.player_state[opp_idx]
    builds = set()
    # Road: WOOD + BRICK
    if ps[PS_WOOD] >= 1 and ps[PS_BRICK] >= 1:
        builds.add(ActionType.BUILD_ROAD)
    # Settlement: WOOD + BRICK + SHEEP + WHEAT
    if ps[PS_WOOD] >= 1 and ps[PS_BRICK] >= 1 and ps[PS_SHEEP] >= 1 and ps[PS_WHEAT] >= 1:
        builds.add(ActionType.BUILD_SETTLEMENT)
    # City: 2 WHEAT + 3 ORE
    if ps[PS_WHEAT] >= 2 and ps[PS_ORE] >= 3:
        builds.add(ActionType.BUILD_CITY)
    # Dev card: SHEEP + WHEAT + ORE
    if ps[PS_SHEEP] >= 1 and ps[PS_WHEAT] >= 1 and ps[PS_ORE] >= 1:
        builds.add(ActionType.BUY_DEVELOPMENT_CARD)
    return builds


def _evaluate_trade_blend(offered_catan, wanted_catan, game, our_color,
                          creator_vps, vps_to_win, bb_value_fn,
                          creator_color=None):
    """Evaluate trade by comparing blend value before and after for both sides.

    Converts game to bitboard, evaluates both our and opponent's position
    before/after trade. Penalizes trades that help the opponent more than us.
    Optionally does 1-ply lookahead to check opponent build potential.

    Returns float: positive = good trade, negative = bad trade.
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

    # Resolve opponent index (if creator_color provided and they can afford it)
    opp_idx = None
    if creator_color is not None and creator_color in bb.color_to_index:
        candidate_idx = bb.color_to_index[creator_color]
        # Verify opponent can afford what they're offering (protects against
        # imperfect state reconstruction / card-counting gaps)
        opp_can_afford = True
        for resource, needed in Counter(offered_catan).items():
            ri = PS_WOOD + RESOURCE_INDEX[resource]
            if bb.player_state[candidate_idx][ri] < needed:
                opp_can_afford = False
                break
        if opp_can_afford:
            opp_idx = candidate_idx

    # Snapshot opponent build capabilities before trade (for lookahead)
    builds_before = _opponent_new_build_types(bb, opp_idx) if opp_idx is not None else set()

    # Evaluate BOTH sides before trade
    value_us_before = bb_value_fn(bb, our_color)
    value_opp_before = bb_value_fn(bb, creator_color) if opp_idx is not None else 0.0

    # Apply trade to BOTH players' resources
    for r in wanted_catan:  # what we give = what they get
        ri = RESOURCE_INDEX[r]
        bb.player_state[p_idx][PS_WOOD + ri] -= 1
        if opp_idx is not None:
            bb.player_state[opp_idx][PS_WOOD + ri] += 1
    for r in offered_catan:  # what they give = what we get
        ri = RESOURCE_INDEX[r]
        bb.player_state[p_idx][PS_WOOD + ri] += 1
        if opp_idx is not None:
            bb.player_state[opp_idx][PS_WOOD + ri] -= 1

    # Evaluate BOTH sides after trade
    value_us_after = bb_value_fn(bb, our_color)
    value_opp_after = bb_value_fn(bb, creator_color) if opp_idx is not None else 0.0

    delta_us = value_us_after - value_us_before
    delta_opp = value_opp_after - value_opp_before

    # 1-ply lookahead: check if trade enables new build types for opponent
    lookahead_penalty = 0.0
    if opp_idx is not None and delta_us > 0:
        builds_after = _opponent_new_build_types(bb, opp_idx)
        new_builds = builds_after - builds_before
        if new_builds:
            # Penalty scales with build importance
            BUILD_THREAT = {
                ActionType.BUILD_CITY: 1.0,
                ActionType.BUILD_SETTLEMENT: 0.8,
                ActionType.BUY_DEVELOPMENT_CARD: 0.5,
                ActionType.BUILD_ROAD: 0.2,
            }
            threat = max(BUILD_THREAT.get(b, 0) for b in new_builds)
            # Scale penalty relative to trade benefit (scale-invariant)
            lookahead_penalty = LOOKAHEAD_WEIGHT * threat * abs(delta_us)

    score = delta_us - OPP_WEIGHT * delta_opp - lookahead_penalty

    logger.debug(
        f"Trade blend: delta_us={delta_us:.4f} delta_opp={delta_opp:.4f} "
        f"score={score:.4f}"
    )

    return score


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
        creator_color=None,
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
                        creator_color=creator_color,
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

    # ------------------------------------------------------------------
    # Trade proposal (outbound)
    # ------------------------------------------------------------------

    # Trade proposal escalation: each successive proposal per turn must clear
    # a higher delta threshold (fraction of position value per prior proposal).
    # 1st=free, 2nd=0.5%, 3rd=1.0%, then hard cap.
    _PROPOSAL_DELTA_STEP = 0.005  # 0.5% of |value_before| per prior proposal
    _MAX_PROPOSALS_PER_TURN = 3

    _PRUNE_TOP_N = 5  # Number of candidates to keep for depth-2 evaluation

    def propose_trade(
        self,
        game,
        our_color: Color,
        opponent_vps: Dict[int, int],
        vps_to_win: int = 10,
        proposals_this_turn: int = 0,
    ) -> Optional[Dict]:
        """Generate the best 1:1 or 2:1 trade proposal, or None.

        Two-phase evaluation:
        1. Fast prune: depth-0 bb_vf + opponent willingness → top N candidates
        2. Accurate eval: depth-2 bb_alphabeta on each survivor → pick best

        The delta threshold escalates with each prior proposal this turn:
        1st=free, 2nd=0.5% of position value, 3rd=1.0%, etc.

        Args:
            game: Reconstructed Catanatron game.
            our_color: Our Catanatron color.
            opponent_vps: {colonist_color_idx: vp_count} for all opponents.
            vps_to_win: VP target.
            proposals_this_turn: How many proposals we've already made this turn.

        Returns:
            {"offered": [colonist_resource_ids], "wanted": [colonist_resource_ids]}
            or None if no good trade found.
        """
        import time as _time
        from robottler.bitboard.convert import game_to_bitboard
        from robottler.bitboard.state import PS_WOOD
        from robottler.bitboard.masks import RESOURCE_INDEX

        # Don't propose if we're about to win (nobody will trade with us)
        our_idx = game.state.color_to_index[our_color]
        our_vps = game.state.player_state.get(
            f"P{our_idx}_ACTUAL_VICTORY_POINTS", 0)
        if our_vps >= vps_to_win - 1:
            return None

        bb_vf = self._get_bb_value_fn()
        bb = game_to_bitboard(game)
        p_idx = bb.color_to_index[our_color]

        # Our current hand
        hand = {}
        for res in RESOURCES:
            hand[res] = int(bb.player_state[p_idx][PS_WOOD + RESOURCE_INDEX[res]])

        # Gather opponent info (skip match-point opponents — they won't trade)
        opponents = []
        for opp_color in bb.color_to_index:
            if opp_color == our_color:
                continue
            opp_game_idx = game.state.color_to_index[opp_color]
            opp_vp = game.state.player_state.get(
                f"P{opp_game_idx}_ACTUAL_VICTORY_POINTS", 0)
            if opp_vp >= vps_to_win - 1:
                continue
            opp_bb_idx = bb.color_to_index[opp_color]
            opp_hand = {}
            for res in RESOURCES:
                opp_hand[res] = int(
                    bb.player_state[opp_bb_idx][PS_WOOD + RESOURCE_INDEX[res]]
                )
            opponents.append((opp_color, opp_bb_idx, opp_hand))

        if not opponents:
            return None

        value_before_us = bb_vf(bb, our_color)

        # Escalating threshold: 1st proposal=free, then 0.5% of position value per prior
        min_delta = proposals_this_turn * self._PROPOSAL_DELTA_STEP * abs(value_before_us)
        skipped_no_willing = 0

        # --- Phase 1: fast prune with depth-0 bb_vf + opponent willingness ---
        candidates = []  # (delta_us, give_res, give_count, want_res)

        for give_res in RESOURCES:
            give_ri = RESOURCE_INDEX[give_res]

            for give_count in (1, 2):
                if hand[give_res] < give_count:
                    continue

                for want_res in RESOURCES:
                    if want_res == give_res:
                        continue

                    want_ri = RESOURCE_INDEX[want_res]

                    # Quick depth-0 delta
                    bb.player_state[p_idx][PS_WOOD + give_ri] -= give_count
                    bb.player_state[p_idx][PS_WOOD + want_ri] += 1
                    delta_us = bb_vf(bb, our_color) - value_before_us
                    bb.player_state[p_idx][PS_WOOD + give_ri] += give_count
                    bb.player_state[p_idx][PS_WOOD + want_ri] -= 1

                    if delta_us <= 0:
                        continue

                    # Opponent willingness heuristic
                    any_willing = False
                    for _, _, opp_hand in opponents:
                        if opp_hand[want_res] < 1:
                            continue
                        if opp_hand[give_res] >= 3:
                            continue
                        any_willing = True
                        break

                    if not any_willing:
                        skipped_no_willing += 1
                        continue

                    candidates.append((delta_us, give_res, give_count, want_res))

        if not candidates:
            logger.info(
                f"No viable trade #{proposals_this_turn + 1} "
                f"(skipped {skipped_no_willing} no-willing-opponent)"
            )
            return None

        # Keep top N candidates by depth-0 score for depth-2 evaluation
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:self._PRUNE_TOP_N]

        # --- Phase 2: depth-2 search on each survivor ---
        player = self._get_player(our_color)
        search_depth = 2
        t0 = _time.monotonic()

        # Baseline: depth-2 value of current position (no trade)
        _, value_no_trade = player.bb_alphabeta(
            bb, search_depth, float("-inf"), float("inf"),
            _time.time() + 30, vps_to_win=vps_to_win,
        )

        best_value = value_no_trade + min_delta  # must beat no-trade + threshold
        best_proposal = None

        for d0_delta, give_res, give_count, want_res in candidates:
            give_ri = RESOURCE_INDEX[give_res]
            want_ri = RESOURCE_INDEX[want_res]

            # Apply trade
            bb.player_state[p_idx][PS_WOOD + give_ri] -= give_count
            bb.player_state[p_idx][PS_WOOD + want_ri] += 1

            _, value_after = player.bb_alphabeta(
                bb, search_depth, float("-inf"), float("inf"),
                _time.time() + 30, vps_to_win=vps_to_win,
            )

            # Revert
            bb.player_state[p_idx][PS_WOOD + give_ri] += give_count
            bb.player_state[p_idx][PS_WOOD + want_ri] -= 1

            if value_after > best_value:
                best_value = value_after
                best_proposal = {
                    "offered": [CATAN_RESOURCE_TO_COLONIST[give_res]] * give_count,
                    "wanted": [CATAN_RESOURCE_TO_COLONIST[want_res]],
                    "_offered_catan": [give_res] * give_count,
                    "_wanted_catan": [want_res],
                    "_delta_us": value_after - value_no_trade,
                }

        elapsed = _time.monotonic() - t0
        if best_proposal:
            logger.info(
                f"Trade proposal #{proposals_this_turn + 1}: "
                f"offer {best_proposal['_offered_catan']} "
                f"for {best_proposal['_wanted_catan']} "
                f"(d2_delta={best_proposal['_delta_us']:.2f}, "
                f"threshold={min_delta:.2f}, "
                f"{len(candidates)} candidates searched in {elapsed:.1f}s, "
                f"skipped {skipped_no_willing} no-willing)"
            )
        else:
            logger.info(
                f"No viable trade #{proposals_this_turn + 1} "
                f"(no candidate beat no-trade at depth 2, "
                f"{len(candidates)} searched in {elapsed:.1f}s, "
                f"skipped {skipped_no_willing} no-willing)"
            )
        return best_proposal

    def pick_trade_acceptee(
        self,
        game,
        our_color: Color,
        acceptees: List[int],
        offered_catan: List[str],
        wanted_catan: List[str],
        delta_us: float,
        opponent_vps: Dict[int, int],
        colonist_to_catan: Dict[int, Color],
        vps_to_win: int = 10,
    ) -> Optional[int]:
        """Pick the best acceptee from those who accepted our trade.

        Args:
            acceptees: List of colonist color indices who accepted.
            offered_catan: What we offered (Catanatron resource names).
            wanted_catan: What we wanted (Catanatron resource names).
            delta_us: Our value improvement from this trade.
            opponent_vps: {colonist_color_idx: vp_count}.
            colonist_to_catan: Color mapping.
            vps_to_win: VP target.

        Returns:
            Colonist color index of best acceptee, or None to cancel.
        """
        from robottler.bitboard.convert import game_to_bitboard
        from robottler.bitboard.state import PS_WOOD
        from robottler.bitboard.masks import RESOURCE_INDEX

        bb_vf = self._get_bb_value_fn()
        bb = game_to_bitboard(game)

        scored = []
        for col_idx in acceptees:
            opp_vp = opponent_vps.get(col_idx, 0)

            # Hard filter: never trade with someone at match point
            if opp_vp >= vps_to_win - 1:
                logger.info(f"Rejecting acceptee {col_idx}: at match point ({opp_vp} VP)")
                continue

            opp_color = colonist_to_catan.get(col_idx)
            if opp_color is None or opp_color not in bb.color_to_index:
                continue
            opp_idx = bb.color_to_index[opp_color]

            # Check opponent can afford what we want (= what they give us)
            can_afford = True
            for res, needed in Counter(wanted_catan).items():
                if bb.player_state[opp_idx][PS_WOOD + RESOURCE_INDEX[res]] < needed:
                    can_afford = False
                    break
            if not can_afford:
                continue

            # Evaluate opponent's gain from this trade
            value_opp_before = bb_vf(bb, opp_color)

            # Apply trade to opponent's resources (temporarily)
            for r in wanted_catan:  # they give us this (they lose)
                bb.player_state[opp_idx][PS_WOOD + RESOURCE_INDEX[r]] -= 1
            for r in offered_catan:  # they get this (they gain)
                bb.player_state[opp_idx][PS_WOOD + RESOURCE_INDEX[r]] += 1

            value_opp_after = bb_vf(bb, opp_color)

            # Revert
            for r in wanted_catan:
                bb.player_state[opp_idx][PS_WOOD + RESOURCE_INDEX[r]] += 1
            for r in offered_catan:
                bb.player_state[opp_idx][PS_WOOD + RESOURCE_INDEX[r]] -= 1

            delta_opp = value_opp_after - value_opp_before

            # VP-weighted: leaders' gains count more
            vp_ratio = opp_vp / vps_to_win if vps_to_win > 0 else 0
            adjusted_opp = delta_opp * (1.0 + vp_ratio)

            # Reject if they benefit way more than us
            if delta_us > 0 and adjusted_opp > 2.0 * delta_us:
                logger.info(
                    f"Rejecting acceptee {col_idx}: opp benefit too high "
                    f"(adj={adjusted_opp:.2f} vs us={delta_us:.2f})")
                continue

            scored.append((col_idx, adjusted_opp))
            logger.debug(
                f"Acceptee {col_idx}: delta_opp={delta_opp:.2f} "
                f"vp_ratio={vp_ratio:.2f} adjusted={adjusted_opp:.2f}")

        if not scored:
            logger.info("No acceptable trade partners, cancelling trade")
            return None

        # Pick the opponent who benefits least (relative to us)
        scored.sort(key=lambda x: x[1])
        winner = scored[0]
        logger.info(f"Picked acceptee {winner[0]} (adj_score={winner[1]:.2f})")
        return winner[0]
