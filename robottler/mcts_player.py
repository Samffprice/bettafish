"""MCTS player for Catan using neural value function and RL policy priors.

MCTSPlayer uses Monte Carlo Tree Search with:
- Neural value function (or blend) for leaf evaluation
- RL policy priors for guiding exploration via PUCT
- Explicit chance nodes for dice rolls and stochastic actions
- Tree reuse between consecutive decisions within a turn

The tree has three node types:
- Decision nodes: player chooses an action, selected by UCB/PUCT
- Chance nodes: dice roll or stochastic outcome, sampled by probability
- Terminal nodes: game over, return win/loss
"""

import math
import time
from collections import defaultdict

import numpy as np

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.map import number_probability
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points


# ---------------------------------------------------------------------------
# Dice probability table (precomputed)
# ---------------------------------------------------------------------------

DICE_PROBS = {}
for _roll in range(2, 13):
    DICE_PROBS[_roll] = number_probability(_roll)

# Dice outcomes as (d1, d2) tuples matching Catanatron's format
DICE_OUTCOMES = {}
for _roll in range(2, 13):
    DICE_OUTCOMES[_roll] = (_roll // 2, math.ceil(_roll / 2))

# Stochastic action types that need chance-node handling
STOCHASTIC_ACTIONS = {ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER}


def _sigmoid(x):
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS tree.

    Supports lazy materialization: children are created as stubs (game=None)
    and only get their game state when first visited. This avoids creating
    game copies for children that are never explored.

    Attributes:
        game: Game state at this node, or None for unmaterialized stubs.
        parent: Parent MCTSNode (None for root).
        action: The action that led to this node from parent.
        children: List of (action, MCTSNode) for decision nodes,
                  or dict {outcome_key: MCTSNode} for chance nodes.
        visit_count: Number of times this node was visited.
        value_sum: Cumulative value from backpropagation.
        prior: Policy prior probability (for PUCT selection).
        is_expanded: Whether children have been created.
        is_chance: Whether this is a chance node (dice/stochastic).
        is_terminal: Whether the game is over at this node.
    """

    __slots__ = (
        'game', 'parent', 'action', 'children', 'visit_count',
        'value_sum', 'prior', 'is_expanded', 'is_chance',
        'is_terminal', '_winner',
    )

    def __init__(self, game, parent=None, action=None, prior=0.0):
        self.game = game  # None for stub nodes (lazy materialization)
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []  # List of (action, MCTSNode) for decision nodes
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.is_chance = False
        self.is_terminal = False
        self._winner = None

        if game is not None:
            self._init_from_game(game)

    def _init_from_game(self, game):
        """Initialize terminal/chance status from game state."""
        winner = game.winning_color()
        if winner is not None:
            self.is_terminal = True
            self._winner = winner
        elif game.state.num_turns >= 1000:
            self.is_terminal = True

        if not self.is_terminal:
            actions = game.playable_actions
            if len(actions) == 1 and actions[0].action_type in STOCHASTIC_ACTIONS:
                self.is_chance = True
                self.children = {}  # dict for chance nodes

    def materialize(self):
        """Create game state from parent's game + action (lazy expansion)."""
        if self.game is not None:
            return
        self.game = self.parent.game.copy()
        self.game.execute(self.action)
        self._init_from_game(self.game)

    @property
    def q_value(self):
        """Average value (exploitation term)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        """PUCT score for selecting among siblings.

        UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + visit_count)

        Unvisited nodes return inf to ensure all children are visited once
        before UCB guides further exploration. In Catan's high-stochasticity
        environment, this broad initial exploration gives more reliable Q
        estimates than FPU-based narrowing.
        """
        if self.visit_count == 0:
            return float('inf')
        exploration = c_puct * self.prior * math.sqrt(max(1, self.parent.visit_count)) / (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct):
        """Select the child with highest UCB score (for decision nodes)."""
        best = None
        best_score = float('-inf')
        for action, child in self.children:
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def most_visited_child(self):
        """Return the (action, child) with highest visit count."""
        best = None
        best_visits = -1
        for action, child in self.children:
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best = (action, child)
        return best


# ---------------------------------------------------------------------------
# MCTSPlayer
# ---------------------------------------------------------------------------

class MCTSPlayer(Player):
    """MCTS player using neural value function and optional RL policy priors.

    Args:
        color: Player color.
        value_fn: Callable(game, color) -> float. Should return values in [0, 1].
                  If using blend, wrap with sigmoid normalization.
        num_simulations: Number of MCTS simulations per decision (default 800).
        c_puct: Exploration constant for PUCT formula (default 1.4).
        time_limit: Optional time limit in seconds per decision.
        policy_fn: Optional callable(game, legal_actions) -> dict {action: prior}.
                   If None, uses uniform priors.
        dirichlet_alpha: If > 0, add Dirichlet noise to root priors (default 0.3).
        dirichlet_weight: Fraction of noise vs prior at root (default 0.25).
        reuse_tree: Whether to reuse subtree from previous decision (default True).
    """

    def __init__(self, color, value_fn, num_simulations=800, c_puct=1.4,
                 time_limit=None, policy_fn=None, dirichlet_alpha=0.3,
                 dirichlet_weight=0.25, reuse_tree=True, is_bot=True):
        super().__init__(color, is_bot)
        self.value_fn = value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.policy_fn = policy_fn
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.reuse_tree = reuse_tree

        # For tree reuse between decisions
        self._root = None

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            self._root = None
            return playable_actions[0]

        root = MCTSNode(game.copy())

        # Expand root with policy priors (if available)
        if not root.is_expanded and not root.is_terminal:
            self._expand(root, playable_actions, use_policy=True)

        # Add Dirichlet noise at root for exploration
        if self.dirichlet_alpha > 0 and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        # Run simulations
        deadline = time.time() + self.time_limit if self.time_limit else float('inf')
        for i in range(self.num_simulations):
            if time.time() >= deadline:
                break
            self._simulate(root)

        # Pick most-visited action
        best_action, best_child = root.most_visited_child()

        return best_action

    def _find_reuse_root(self, playable_actions):
        """Try to find a child of the stored root matching current state."""
        if self._root is None or not self._root.is_expanded:
            return None

        # For decision nodes, check if any child's playable_actions match
        if isinstance(self._root.children, list):
            for action, child in self._root.children:
                if child.is_expanded or child.visit_count > 0:
                    # Check if this child's game state matches current playable actions
                    try:
                        child_actions = child.game.playable_actions
                        if len(child_actions) == len(playable_actions):
                            # Good enough match — same number of actions
                            child.parent = None
                            return child
                    except Exception:
                        pass

        # For chance nodes
        elif isinstance(self._root.children, dict):
            for key, child in self._root.children.items():
                if child.is_expanded or child.visit_count > 0:
                    try:
                        child_actions = child.game.playable_actions
                        if len(child_actions) == len(playable_actions):
                            child.parent = None
                            return child
                    except Exception:
                        pass

        return None

    def _simulate(self, root):
        """Run one MCTS simulation: select -> evaluate -> expand -> backprop.

        Standard PUCT flow:
        1. SELECT: walk tree via UCB until reaching an unexpanded leaf
        2. MATERIALIZE: create game state for the leaf (lazy copy)
        3. EVALUATE: run value function on the leaf
        4. EXPAND: create stub children for future selection
        5. BACKPROPAGATE: update visit counts and values up to root
        """
        node = root

        # === SELECTION ===
        while node.is_expanded and not node.is_terminal:
            if node.is_chance:
                node = self._select_chance(node)
            else:
                _, node = node.best_child(self.c_puct)

        # === MATERIALIZE (lazy game copy) ===
        if node.game is None:
            node.materialize()

        # === EVALUATION ===
        if node.is_terminal:
            value = self._terminal_value(node)
        else:
            value = self.value_fn(node.game, self.color)
            # === EXPANSION (create stubs for future simulations) ===
            self._expand(node)

        # === BACKPROPAGATION ===
        self._backpropagate(node, value)

    def _select_chance(self, node):
        """At a chance node, sample an outcome weighted by probability."""
        actions = node.game.playable_actions
        if len(actions) != 1:
            # Shouldn't happen — chance nodes have exactly one stochastic action
            # Fall back to random selection
            action = actions[0]
        else:
            action = actions[0]

        if action.action_type == ActionType.ROLL:
            return self._sample_dice(node, action)
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return self._sample_dev_card(node, action)
        elif action.action_type == ActionType.MOVE_ROBBER:
            return self._sample_robber(node, action)
        else:
            # Unknown stochastic action — just execute it deterministically
            return self._execute_and_create_child(node, action, "det")

    def _sample_dice(self, node, action):
        """Sample a dice outcome weighted by probability, reuse if seen before."""
        # Sample roll weighted by standard dice probabilities
        rolls = list(range(2, 13))
        probs = [DICE_PROBS[r] for r in rolls]
        roll = rolls[np.random.choice(len(rolls), p=probs)]

        if roll in node.children:
            return node.children[roll]

        # Create new child for this roll outcome
        game_copy = node.game.copy()
        outcome = DICE_OUTCOMES[roll]
        roll_action = type(action)(action.color, ActionType.ROLL, outcome)
        game_copy.execute(roll_action, validate_action=False)

        child = MCTSNode(game_copy, parent=node, action=roll_action)
        node.children[roll] = child
        return child

    def _sample_dev_card(self, node, action):
        """Sample a dev card purchase outcome."""
        from catanatron.models.enums import DEVELOPMENT_CARDS
        from catanatron.state_functions import get_dev_cards_in_hand, get_enemy_colors

        # Build deck from current player's perspective
        current_deck = node.game.state.development_listdeck.copy()
        for c in get_enemy_colors(node.game.state.colors, action.color):
            for card in DEVELOPMENT_CARDS:
                n = get_dev_cards_in_hand(node.game.state, c, card)
                current_deck += [card] * n

        if len(current_deck) == 0:
            # No cards left — shouldn't happen but handle gracefully
            return self._execute_and_create_child(node, action, "no_cards")

        # Sample a card
        card = current_deck[np.random.randint(len(current_deck))]
        key = f"dev_{card}"

        if key in node.children:
            return node.children[key]

        game_copy = node.game.copy()
        buy_action = type(action)(action.color, ActionType.BUY_DEVELOPMENT_CARD, card)
        try:
            game_copy.execute(buy_action, validate_action=False)
        except Exception:
            game_copy = node.game.copy()
            game_copy.execute(action, validate_action=False)

        child = MCTSNode(game_copy, parent=node, action=buy_action)
        node.children[key] = child
        return child

    def _sample_robber(self, node, action):
        """Sample a robber steal outcome."""
        from catanatron.models.enums import RESOURCES
        from catanatron.state_functions import get_player_freqdeck

        coordinate, robbed_color = action.value
        if robbed_color is None:
            return self._execute_and_create_child(node, action, "rob_none")

        opponent_hand = get_player_freqdeck(node.game.state, robbed_color)
        total = sum(opponent_hand)
        if total == 0:
            return self._execute_and_create_child(node, action, "rob_empty")

        # Sample a resource weighted by opponent's hand
        probs = [c / total for c in opponent_hand]
        resource_idx = np.random.choice(len(RESOURCES), p=probs)
        resource = RESOURCES[resource_idx]
        key = f"rob_{resource}"

        if key in node.children:
            return node.children[key]

        game_copy = node.game.copy()
        rob_action = type(action)(action.color, ActionType.MOVE_ROBBER, (coordinate, robbed_color, resource))
        try:
            game_copy.execute(rob_action, validate_action=False)
        except Exception:
            # Fall back to original action
            game_copy = node.game.copy()
            game_copy.execute(action, validate_action=False)

        child = MCTSNode(game_copy, parent=node, action=rob_action)
        node.children[key] = child
        return child

    def _execute_and_create_child(self, node, action, key):
        """Execute action deterministically and create a child node."""
        if key in node.children:
            return node.children[key]

        game_copy = node.game.copy()
        game_copy.execute(action, validate_action=False)
        child = MCTSNode(game_copy, parent=node, action=action)
        node.children[key] = child
        return child

    def _expand(self, node, legal_actions=None, use_policy=None):
        """Expand a decision node: create stub children (lazy materialization).

        Children are created with game=None. Their game state is only
        materialized (copy + execute) when first visited during selection.
        This avoids paying the copy cost for children that are never explored.

        Policy priors are used only at the root (use_policy=True) to focus
        search on promising actions. Deeper nodes use uniform priors to avoid
        the expensive RL model forward pass at every expansion.
        """
        if node.is_expanded or node.is_terminal:
            return

        if node.is_chance:
            # Chance nodes are expanded lazily via sampling
            node.is_expanded = True
            return

        if legal_actions is None:
            legal_actions = node.game.playable_actions

        if len(legal_actions) == 0:
            node.is_terminal = True
            return

        # Get policy priors (only at root by default)
        if use_policy is None:
            use_policy = (node.parent is None)

        if use_policy and self.policy_fn is not None:
            priors = self.policy_fn(node.game, legal_actions)
        else:
            # Uniform priors
            uniform_p = 1.0 / len(legal_actions)
            priors = {a: uniform_p for a in legal_actions}

        # Create stub children (game=None, materialized on first visit)
        node.children = []
        for action in legal_actions:
            prior = priors.get(action, 0.0)
            child = MCTSNode(None, parent=node, action=action, prior=prior)
            node.children.append((action, child))

        node.is_expanded = True

    def _terminal_value(self, node):
        """Return value for terminal nodes from MCTS player's perspective."""
        if node._winner == self.color:
            return 1.0
        elif node._winner is not None:
            return 0.0
        else:
            # Draw (turns limit reached)
            return 0.5

    def _backpropagate(self, node, value):
        """Walk up the tree updating visit counts and values.

        Value is always from the MCTS player's perspective. At opponent nodes,
        we store the same perspective value — UCB naturally handles this because
        opponent nodes pick the child that MINIMIZES our value, which means the
        child with the lowest Q from our perspective.

        Actually, for PUCT to work correctly, each node should store value from
        the perspective of the player who just moved TO this node (i.e., the
        player who CHOSE the action leading here). This way, best_child()
        always maximizes.

        We flip the value when crossing player boundaries.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            # Determine if the node's parent chose this node (maximizing their value)
            # The value should be from the perspective of the player who selected this node
            if current.parent is not None and not current.parent.is_chance:
                # Who made the decision at parent?
                parent_color = current.parent.game.state.current_color()
                if parent_color == self.color:
                    current.value_sum += value
                else:
                    current.value_sum += (1.0 - value)
            else:
                # Root node or chance node parent — store from our perspective
                current.value_sum += value
            current = current.parent

    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root priors for exploration."""
        if not isinstance(node.children, list) or len(node.children) == 0:
            return

        n_children = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n_children)
        eps = self.dirichlet_weight

        for i, (action, child) in enumerate(node.children):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def __repr__(self):
        return (
            f"MCTSPlayer({self.color}, sims={self.num_simulations}, "
            f"c_puct={self.c_puct})"
        )


# ---------------------------------------------------------------------------
# Policy function factory (wraps RL model for MCTS priors)
# ---------------------------------------------------------------------------

def make_policy_fn(rl_model_path, bc_path, rl_model=None):
    """Create a policy function from RL model for MCTS priors.

    Returns callable(game, legal_actions) -> dict {action: prior_probability}.
    """
    import torch
    from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE
    from robottler.search_player import _load_feature_stats, _extract_obs

    if rl_model is None:
        from sb3_contrib import MaskablePPO
        rl_model = MaskablePPO.load(rl_model_path)

    strategic_names, means, stds = _load_feature_stats(bc_path)

    def policy_fn(game, legal_actions):
        # Extract observation (always from current player's perspective)
        color = game.state.current_color()
        obs = _extract_obs(game, color, strategic_names, means, stds)

        # Build action mask
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        action_to_idx = {}
        for action in legal_actions:
            try:
                idx = to_action_space(action)
                mask[idx] = True
                action_to_idx[id(action)] = idx
            except (ValueError, IndexError):
                pass

        # Get policy distribution
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = rl_model.policy.get_distribution(obs_t, action_masks=mask_t)
            probs = dist.distribution.probs.squeeze(0).numpy()

        # Map probabilities back to actions
        result = {}
        total_mapped = 0.0
        for action in legal_actions:
            aid = id(action)
            if aid in action_to_idx:
                p = float(probs[action_to_idx[aid]])
                result[action] = p
                total_mapped += p
            else:
                result[action] = 0.0

        # Normalize (in case some actions weren't mappable)
        if total_mapped > 0:
            for a in result:
                result[a] /= total_mapped
        else:
            # Fallback to uniform
            uniform_p = 1.0 / len(legal_actions)
            for a in legal_actions:
                result[a] = uniform_p

        return result

    return policy_fn


# ---------------------------------------------------------------------------
# Value function wrappers for MCTS (normalized to [0, 1])
# ---------------------------------------------------------------------------

def make_mcts_value_fn(bc_path):
    """Create a [0,1]-normalized value function using the BC neural net.

    For 1v1 (fast path): uses single neural forward pass. The raw output
    is already a win probability estimator (confirmed: 0.376 avg when winning
    vs 0.104 avg when losing). Rescaled to use more of [0, 1].

    For multiplayer: uses relative assessment my/(my+opp) for proper
    normalization across >2 players.
    """
    from robottler.value_model import load_value_model
    neural_fn = load_value_model(bc_path)

    def value_fn(game, p0_color):
        my_val = neural_fn(game, p0_color)
        opp_colors = [c for c in game.state.colors if c != p0_color]

        if len(opp_colors) == 1:
            # 1v1 fast path: single forward pass, rescale [0, 0.8] → [0, 1]
            return min(1.0, my_val * 1.5)
        else:
            # Multiplayer: ratio normalization (2 forward passes)
            opp_val = max(neural_fn(game, c) for c in opp_colors)
            total = my_val + opp_val
            if total < 1e-8:
                return 0.5
            return my_val / total

    return value_fn


def make_mcts_blend_value_fn(bc_path, blend_weight=1e8):
    """Create a [0,1]-normalized blend value function for MCTS.

    Uses sigmoid(diff / SCALE) of the blend function (heuristic + neural).
    Same SCALE=5e8 as the heuristic-only version, plus the neural signal
    adds an extra ~1e8 * neural(0.1) = 1e7 to the diff, providing
    strategic tiebreaking within the same VP tier.
    """
    from robottler.search_player import make_blended_value_fn
    blend_fn = make_blended_value_fn(bc_path, blend_weight=blend_weight)

    SCALE = 5e8

    def value_fn(game, p0_color):
        my_val = blend_fn(game, p0_color)
        opp_colors = [c for c in game.state.colors if c != p0_color]
        if opp_colors:
            opp_val = blend_fn(game, opp_colors[0])
            diff = my_val - opp_val
            return _sigmoid(diff / SCALE)
        return 0.5

    return value_fn


def make_mcts_fast_value_fn(bc_path):
    """Create a fast [0,1] value function using single neural pass + VP bonus.

    Uses the raw neural net output (already [0,1]) rescaled to have more
    spread, plus a VP-based component for strong positional signal.
    Single forward pass per evaluation, ~2x faster than ratio approach.
    """
    from robottler.value_model import load_value_model
    from catanatron.state_functions import get_actual_victory_points

    neural_fn = load_value_model(bc_path)

    def value_fn(game, p0_color):
        # Neural component: rescale [0, 0.8] → [0, 0.5]
        raw = neural_fn(game, p0_color)
        neural_part = min(0.5, raw * 0.625)

        # VP component: provides strong signal for VP differences
        my_vp = get_actual_victory_points(game.state, p0_color)
        opp_colors = [c for c in game.state.colors if c != p0_color]
        opp_vp = max(get_actual_victory_points(game.state, c) for c in opp_colors) if opp_colors else 0
        vp_diff = my_vp - opp_vp
        vp_part = _sigmoid(vp_diff * 0.5)  # Each VP ≈ 12% swing

        # Blend: VP dominates but neural provides tiebreaking
        return 0.6 * vp_part + 0.4 * neural_part

    return value_fn


def make_mcts_heuristic_value_fn():
    """Create a [0,1]-normalized heuristic-only value function for MCTS.

    Uses sigmoid(diff / SCALE) where SCALE=5e8 maps production-level
    differences (1e8) to sigmoid ≈ 0.55 and VP differences (3e14) to
    sigmoid ≈ 1.0. This gives good resolution for non-VP signals while
    strongly favoring VP advantages.
    """
    from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
    heuristic_fn = base_fn(DEFAULT_WEIGHTS)

    SCALE = 5e8

    def value_fn(game, p0_color):
        my_val = heuristic_fn(game, p0_color)
        opp_colors = [c for c in game.state.colors if c != p0_color]
        if opp_colors:
            opp_val = heuristic_fn(game, opp_colors[0])
            diff = my_val - opp_val
            return _sigmoid(diff / SCALE)
        return 0.5

    return value_fn
