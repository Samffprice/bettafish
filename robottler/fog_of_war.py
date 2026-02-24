"""Fog-of-war player wrapper for imperfect information benchmarks.

Wraps any player to hide opponent resource hands, optionally using card
counting from observable game events (dice distributions, builds, trades).

Usage:
    inner = BitboardSearchPlayer(Color.RED, ...)
    fog_player = FogOfWarPlayer(inner, use_counting=True)
    # fog_player.decide(game, actions) masks opponent hands before search
"""

from bridge.card_tracker import OpponentCardTracker, RESOURCES
from catanatron.apply_action import yield_resources
from catanatron.models.enums import ActionType
from catanatron.state_functions import player_num_resource_cards


class FogOfWarPlayer:
    """Wraps a player to simulate imperfect information.

    Hides opponent resource hands from the inner player.  With counting
    enabled, tracks observable events (dice yields, build costs, trades)
    to estimate opponent hands.  Without counting, distributes total
    card count evenly across resources (worst case).

    All estimates are reconciled with the authoritative total before
    each decision, so tracking errors don't accumulate.
    """

    def __init__(self, inner_player, use_counting=True):
        self.inner = inner_player
        self.color = inner_player.color
        self.use_counting = use_counting
        self.tracker = OpponentCardTracker() if use_counting else None
        self._last_record_idx = 0
        self._initialized = False
        self._setup_done = False
        # Track opponent road building (free roads don't cost resources)
        self._free_roads = {}  # {color: remaining_free_roads}

    # Expose inner player attributes the benchmark/game might need
    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, game, playable_actions):
        """Mask opponent hands, then delegate to inner player."""
        if not self._initialized:
            self._init_tracker(game)
            self._initialized = True

        self._process_new_records(game)
        masked_game = self._mask_game(game)
        return self.inner.decide(masked_game, playable_actions)

    def _init_tracker(self, game):
        if not self.tracker:
            return
        for color in game.state.colors:
            if color != self.color:
                pidx = game.state.color_to_index[color]
                self.tracker.init_player(pidx)

    def _process_new_records(self, game):
        """Process action records since last call to update card tracker."""
        if not self.tracker:
            return

        records = game.state.action_records
        state = game.state

        while self._last_record_idx < len(records):
            record = records[self._last_record_idx]
            action = record.action
            self._last_record_idx += 1

            # Track when setup ends (first ROLL = normal play begins)
            if action.action_type == ActionType.ROLL and not self._setup_done:
                self._setup_done = True

            color = action.color
            is_opponent = color != self.color

            # --- Dice distributions (observable for all players) ---
            if action.action_type == ActionType.ROLL:
                d1, d2 = record.result
                number = d1 + d2
                if number != 7:
                    # Compute yields from board state.  Using current board
                    # which may have builds placed after the roll — reconciliation
                    # corrects any minor over/undercount.
                    payout, _ = yield_resources(
                        state.board, state.resource_freqdeck, number
                    )
                    for pay_color, freqdeck in payout.items():
                        if pay_color == self.color:
                            continue
                        opp_idx = state.color_to_index[pay_color]
                        for ri, resource in enumerate(RESOURCES):
                            count = freqdeck[ri]
                            if count > 0:
                                self.tracker.add_known(opp_idx, resource, count)

            # --- Our rob steal (we see the stolen card) ---
            elif (action.action_type == ActionType.MOVE_ROBBER
                  and not is_opponent and record.result is not None):
                victim_color = action.value[1]
                if victim_color is not None:
                    victim_idx = state.color_to_index[victim_color]
                    stolen = record.result  # FastResource string
                    self.tracker.on_rob_known(victim_idx, stolen)

            # --- Opponent rob (we don't see which card, just -1 total) ---
            # Handled by reconciliation — no explicit tracking needed.

            # --- Skip non-opponent actions below ---
            if not is_opponent:
                continue

            pidx = state.color_to_index[color]

            # --- Builds (known costs, skip setup + free roads) ---
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                if self._setup_done:
                    self.tracker.on_build(pidx, "SETTLEMENT")

            elif action.action_type == ActionType.BUILD_CITY:
                self.tracker.on_build(pidx, "CITY")

            elif action.action_type == ActionType.BUILD_ROAD:
                free = self._free_roads.get(color, 0)
                if free > 0:
                    self._free_roads[color] = free - 1
                elif self._setup_done:
                    self.tracker.on_build(pidx, "ROAD")

            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                self.tracker.on_buy_dev_card(pidx)

            # --- Road building card (next 2 roads are free) ---
            elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
                self._free_roads[color] = 2

            # --- Monopoly (we lose our cards, we see the resource) ---
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                mono_resource = action.value
                # All opponents of the monopoly player lose that resource.
                # We know OUR loss exactly.  For other opponents, their
                # entire known count of that resource goes to the caller.
                for c in state.colors:
                    if c == color:
                        continue
                    c_idx = state.color_to_index[c]
                    if c == self.color:
                        continue  # our hand is always accurate
                    if c_idx in self.tracker.hands:
                        # They lose all of this resource
                        lost = self.tracker.hands[c_idx].known[mono_resource]
                        self.tracker.hands[c_idx].known[mono_resource] = 0
                        # Caller gains what we knew about
                        self.tracker.add_known(pidx, mono_resource, lost)

            # --- Discard (resource types visible in action record) ---
            elif action.action_type == ActionType.DISCARD:
                # action.value = list of discarded resource strings
                if isinstance(action.value, (list, tuple)):
                    for res in action.value:
                        if res in self.tracker.hands.get(pidx, _EMPTY).known:
                            self.tracker.hands[pidx].known[res] = max(
                                0, self.tracker.hands[pidx].known[res] - 1
                            )

            # --- Maritime trade (all public) ---
            elif action.action_type == ActionType.MARITIME_TRADE:
                trade_offer = action.value
                # Offered: non-None elements from [:-1]
                for res in trade_offer[:-1]:
                    if res is not None:
                        self.tracker.hands[pidx].known[res] = max(
                            0, self.tracker.hands[pidx].known.get(res, 0) - 1
                        )
                # Received: last element
                received = trade_offer[-1]
                if received is not None:
                    self.tracker.add_known(pidx, received)

            # --- Domestic trade (CONFIRM_TRADE has full details) ---
            elif action.action_type == ActionType.CONFIRM_TRADE:
                # value = (offering[5], asking[5], enemy_color)
                offering = action.value[:5]
                asking = action.value[5:10]
                enemy_color = action.value[10]
                enemy_idx = state.color_to_index[enemy_color]

                # Caller loses offering, gains asking
                for ri, resource in enumerate(RESOURCES):
                    if offering[ri] > 0:
                        self.tracker.hands[pidx].known[resource] = max(
                            0, self.tracker.hands[pidx].known[resource] - offering[ri]
                        )
                    if asking[ri] > 0:
                        self.tracker.add_known(pidx, resource, asking[ri])

                # Enemy loses asking, gains offering
                if enemy_color != self.color and enemy_idx in self.tracker.hands:
                    for ri, resource in enumerate(RESOURCES):
                        if asking[ri] > 0:
                            self.tracker.hands[enemy_idx].known[resource] = max(
                                0, self.tracker.hands[enemy_idx].known[resource] - asking[ri]
                            )
                        if offering[ri] > 0:
                            self.tracker.add_known(enemy_idx, resource, offering[ri])

    def _mask_game(self, game):
        """Create a game copy with opponent hands replaced by estimates."""
        masked = game.copy()

        for color in game.state.colors:
            if color == self.color:
                continue

            pidx = game.state.color_to_index[color]
            actual_total = player_num_resource_cards(game.state, color)

            if self.tracker:
                self.tracker.reconcile(pidx, actual_total)
                estimated = self.tracker.get_resources_for_state(pidx)
            else:
                # No counting — distribute total evenly
                estimated = {}
                for i, r in enumerate(RESOURCES):
                    estimated[r] = actual_total // 5 + (1 if i < actual_total % 5 else 0)

            for resource in RESOURCES:
                masked.state.player_state[
                    f"P{pidx}_{resource}_IN_HAND"
                ] = estimated.get(resource, 0)

        return masked


class _EmptyHand:
    known = {r: 0 for r in RESOURCES}

_EMPTY = _EmptyHand()
