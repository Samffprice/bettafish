import time
import random

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import expand_spectrum, list_prunned_actions
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
)


ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 20


class AlphaBetaPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    NOTE: More than 3 levels seems to take much longer, it would be
    interesting to see this with prunning.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false"
        self.value_fn_builder_name = (
            "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.use_value_function = None
        self.epsilon = epsilon
        self.dice_sample_size = None  # None = all 11 outcomes; set to e.g. 5 to sample

    def value_function(self, game, p0_color):
        raise NotImplementedError

    def get_actions(self, game):
        if self.prunning:
            return list_prunned_actions(game)
        return game.playable_actions

    def decide(self, game: Game, playable_actions):
        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        deadline = time.time() + MAX_SEARCH_TIME_SECS
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline
        )
        if result[0] is None:
            return playable_actions[0]
        return result[0]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline):
        """AlphaBeta MiniMax Algorithm.

        Returns:
            (action|None, value): Best action (None at leaf) and its value.
        """
        if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            return None, value_fn(game, self.color)

        maximizingPlayer = game.state.current_color() == self.color
        actions = self.get_actions(game)
        action_outcomes = expand_spectrum(game, actions, self.dice_sample_size)

        if maximizingPlayer:
            best_action = None
            best_value = float("-inf")
            for action, outcomes in action_outcomes.items():
                expected_value = 0
                for outcome, proba in outcomes:
                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline
                    )
                    expected_value += proba * result[1]

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break

            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for action, outcomes in action_outcomes.items():
                expected_value = 0
                for outcome, proba in outcomes:
                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline
                    )
                    expected_value += proba * result[1]

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

            return best_action, best_value


class SameTurnAlphaBetaPlayer(AlphaBetaPlayer):
    """
    Same like AlphaBeta but only within turn
    """

    def alphabeta(self, game, depth, alpha, beta, deadline):
        if (
            depth == 0
            or game.state.current_color() != self.color
            or game.winning_color() is not None
            or time.time() >= deadline
        ):
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            return None, value_fn(game, self.color)

        actions = self.get_actions(game)
        action_outcomes = expand_spectrum(game, actions, self.dice_sample_size)

        best_action = None
        best_value = float("-inf")
        for action, outcomes in action_outcomes.items():
            expected_value = 0
            for outcome, proba in outcomes:
                result = self.alphabeta(
                    outcome, depth - 1, alpha, beta, deadline
                )
                expected_value += proba * result[1]

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_action, best_value
