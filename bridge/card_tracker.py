"""Opponent card counting via observable game events.

Tracks per-opponent resource hands using information from dice distributions,
trades, building costs, and dev card purchases.  Unknown cards (robber steals
by third parties, Year of Plenty) are handled via reconciliation with the
authoritative total from colonist.io state diffs.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bridge.config import COLONIST_RESOURCE_TO_CATAN

logger = logging.getLogger(__name__)

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

# Known building costs
BUILDING_COSTS: Dict[str, Dict[str, int]] = {
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "CITY": {"WHEAT": 2, "ORE": 3},
    "DEV_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}


@dataclass
class PlayerHand:
    """Tracked resource hand for one opponent."""
    known: Dict[str, int] = field(
        default_factory=lambda: {r: 0 for r in RESOURCES}
    )
    total: int = 0  # authoritative total from state diff

    @property
    def unknown(self) -> int:
        """Cards we can't attribute to a specific resource."""
        return max(0, self.total - sum(self.known.values()))


class OpponentCardTracker:
    """Track opponent resource hands from observable game events."""

    def __init__(self):
        self.hands: Dict[int, PlayerHand] = {}

    def reset(self):
        """Clear all tracking (new game)."""
        self.hands.clear()

    def init_player(self, col_idx: int):
        """Initialize tracking for an opponent."""
        self.hands[col_idx] = PlayerHand()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_resource_distribution(
        self, distributions: List[Dict], my_color_idx: int
    ):
        """Process type 28 dice distribution for opponents."""
        for dist in distributions:
            owner = dist.get("owner", -1)
            card = dist.get("card", -1)
            if owner == my_color_idx or owner not in self.hands:
                continue
            resource = COLONIST_RESOURCE_TO_CATAN.get(card)
            if resource:
                self.hands[owner].known[resource] += 1

    def on_trade(
        self,
        giving_player: int,
        receiving_player: int,
        giving_cards: List[int],
        receiving_cards: List[int],
        my_color_idx: int,
    ):
        """Process type 43 trade execution for opponents."""
        # Giver loses giving_cards, gains receiving_cards
        if giving_player != my_color_idx and giving_player in self.hands:
            hand = self.hands[giving_player]
            for card_id in giving_cards:
                resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                if resource:
                    hand.known[resource] = max(0, hand.known[resource] - 1)
            for card_id in receiving_cards:
                resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                if resource:
                    hand.known[resource] += 1

        # Receiver loses receiving_cards, gains giving_cards
        if receiving_player != my_color_idx and receiving_player in self.hands:
            hand = self.hands[receiving_player]
            for card_id in receiving_cards:
                resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                if resource:
                    hand.known[resource] = max(0, hand.known[resource] - 1)
            for card_id in giving_cards:
                resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                if resource:
                    hand.known[resource] += 1

    def on_build(self, col_idx: int, building_type: str):
        """Deduct known building costs when an opponent builds.

        building_type: "ROAD", "SETTLEMENT", "CITY", or "DEV_CARD"
        """
        if col_idx not in self.hands:
            return
        costs = BUILDING_COSTS.get(building_type)
        if not costs:
            return
        hand = self.hands[col_idx]
        for resource, amount in costs.items():
            hand.known[resource] = max(0, hand.known[resource] - amount)

    def on_buy_dev_card(self, col_idx: int):
        """Deduct dev card purchase cost (SHEEP + WHEAT + ORE)."""
        self.on_build(col_idx, "DEV_CARD")

    def on_rob_known(self, victim_col_idx: int, resource: str):
        """Deduct a specific resource when we rob an opponent."""
        if victim_col_idx not in self.hands:
            return
        self.hands[victim_col_idx].known[resource] = max(
            0, self.hands[victim_col_idx].known[resource] - 1
        )

    def add_known(self, col_idx: int, resource: str, count: int = 1):
        """Add known resources (e.g. from observed dice yields)."""
        if col_idx not in self.hands:
            return
        self.hands[col_idx].known[resource] += count

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, col_idx: int, authoritative_total: int):
        """Reconcile tracked hand with authoritative total from state diff.

        If sum(known) > total, scale down proportionally.  This handles
        unobserved events (robbery by others, free roads from road building
        card, etc.) without guessing.
        """
        if col_idx not in self.hands:
            self.init_player(col_idx)
        hand = self.hands[col_idx]
        hand.total = authoritative_total

        known_sum = sum(hand.known.values())
        if known_sum <= authoritative_total:
            return  # consistent — unknown bucket absorbs the gap

        if known_sum == 0:
            return  # nothing to scale

        # Overcount: scale down proportionally
        scale = authoritative_total / known_sum
        for resource in RESOURCES:
            hand.known[resource] = int(hand.known[resource] * scale)

        # Fix rounding — distribute remainder to largest counts
        remainder = authoritative_total - sum(hand.known.values())
        if remainder > 0:
            by_count = sorted(RESOURCES, key=lambda r: hand.known[r], reverse=True)
            for r in by_count:
                if remainder <= 0:
                    break
                hand.known[r] += 1
                remainder -= 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_resources_for_state(self, col_idx: int) -> Dict[str, int]:
        """Return resource dict for opponent_resources, distributing unknowns.

        Known resources are preserved exactly.  Unknown cards are distributed
        evenly across resource types (same as the old fallback, but now the
        unknown pool is much smaller).
        """
        if col_idx not in self.hands:
            return {r: 0 for r in RESOURCES}

        hand = self.hands[col_idx]
        result = dict(hand.known)
        unknown = hand.unknown

        if unknown > 0:
            per_type = unknown // 5
            remainder = unknown % 5
            for i, r in enumerate(RESOURCES):
                result[r] += per_type + (1 if i < remainder else 0)

        return result

    def log_state(self, col_idx: int, label: str = ""):
        """Log current tracking state for debugging."""
        if col_idx not in self.hands:
            return
        hand = self.hands[col_idx]
        known_str = ", ".join(f"{r[0]}:{hand.known[r]}" for r in RESOURCES)
        prefix = f"[{label}] " if label else ""
        logger.debug(
            f"{prefix}CardTracker P{col_idx}: known=[{known_str}] "
            f"total={hand.total} unknown={hand.unknown}"
        )
