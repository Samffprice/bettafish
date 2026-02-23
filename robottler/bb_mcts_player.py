"""AlphaZero-style MCTS player on the bitboard engine.

Uses CatanAlphaZeroNet (dual-head) for both value evaluation and policy priors.
Runs entirely on BitboardState for fast copy/apply/movegen.

Key differences from mcts_player.py:
- BitboardState instead of Game objects (5µs copy vs 35µs)
- Dual-head network provides both value and policy in one forward pass
- Value in [-1, +1] (tanh), no normalization needed
- Action mask via to_action_space for policy
"""

import math
import time

import numpy as np
import torch

from catanatron.models.enums import ActionType, Action
from catanatron.models.map import number_probability
from catanatron.models.player import Player

from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE, to_action_space,
)
from robottler.alphazero_net import (
    CatanAlphaZeroNet, load_checkpoint,
)
from robottler.bitboard.convert import game_to_bitboard, DEV_ID_TO_NAME
from robottler.bitboard.movegen import bb_generate_actions
from robottler.bitboard.actions import bb_apply_action
from robottler.bitboard.features import FeatureIndexer, bb_fill_feature_vector
from robottler.bitboard.state import (
    PS_RESOURCE_START, PS_RESOURCE_END,
    PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
)


# Dice probabilities
DICE_PROBS = {r: number_probability(r) for r in range(2, 13)}
DICE_OUTCOMES = {r: (r // 2, math.ceil(r / 2)) for r in range(2, 13)}

# Stochastic action types
STOCHASTIC_ACTIONS = {ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD}


class BBMCTSNode:
    """MCTS node backed by BitboardState.

    Supports lazy materialization: children start as stubs (bb_state=None)
    and only get their state when first visited.
    """

    __slots__ = (
        'bb_state', 'parent', 'action', 'children', 'visit_count',
        'value_sum', 'prior', 'is_expanded', 'is_chance',
        'is_terminal', '_winner_pidx', '_cached_actions',
    )

    def __init__(self, bb_state, parent=None, action=None, prior=0.0):
        self.bb_state = bb_state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.is_chance = False
        self.is_terminal = False
        self._winner_pidx = -1
        self._cached_actions = None

        if bb_state is not None:
            self._init_from_state(bb_state)

    def _init_from_state(self, bb_state):
        """Initialize terminal/chance status and cache actions."""
        w = bb_state.winning_player()
        if w >= 0:
            self.is_terminal = True
            self._winner_pidx = w
            return

        if bb_state.num_turns >= 1000:
            self.is_terminal = True
            return

        actions = bb_generate_actions(bb_state)
        self._cached_actions = actions
        if len(actions) == 1 and actions[0].action_type in STOCHASTIC_ACTIONS:
            self.is_chance = True
            self.children = {}  # dict for chance nodes

    def materialize(self):
        """Create bb_state from parent's state + action."""
        if self.bb_state is not None:
            return
        self.bb_state = self.parent.bb_state.copy()
        bb_apply_action(self.bb_state, self.action)
        self._init_from_state(self.bb_state)

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        """PUCT score. Unvisited nodes return inf for broad exploration."""
        if self.visit_count == 0:
            return float('inf')
        exploration = (c_puct * self.prior *
                       math.sqrt(max(1, self.parent.visit_count)) /
                       (1 + self.visit_count))
        return self.q_value + exploration

    def best_child(self, c_puct):
        """Select child with highest UCB score."""
        best = None
        best_score = float('-inf')
        for action, child in self.children:
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def most_visited_child(self):
        """Return (action, child) with highest visit count."""
        best = None
        best_visits = -1
        for action, child in self.children:
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best = (action, child)
        return best

    def visit_distribution(self):
        """Return visit count distribution over children as dict {action: count}."""
        if isinstance(self.children, list):
            return {action: child.visit_count for action, child in self.children}
        return {}


class BBMCTSPlayer(Player):
    """AlphaZero-style MCTS player on bitboard engine.

    Args:
        color: Player color
        az_net: CatanAlphaZeroNet instance (dual-head)
        feature_indexer: FeatureIndexer for fast feature extraction
        feature_means: numpy array of feature means for normalization
        feature_stds: numpy array of feature stds for normalization
        num_simulations: MCTS simulations per decision (default 800)
        c_puct: Exploration constant (default 1.4)
        time_limit: Optional time limit in seconds
        dirichlet_alpha: Noise alpha at root (default 0.3)
        dirichlet_weight: Noise weight at root (default 0.25)
        temperature: Action selection temperature (1.0=proportional, 0=greedy)
    """

    def __init__(self, color, az_net, feature_indexer, feature_means, feature_stds,
                 num_simulations=800, c_puct=1.4, time_limit=None,
                 dirichlet_alpha=0.3, dirichlet_weight=0.25,
                 temperature=0.0, is_bot=True, leaf_value_fn=None,
                 leaf_use_policy=True):
        super().__init__(color, is_bot)
        self.az_net = az_net
        self.fi = feature_indexer
        self.means = torch.tensor(feature_means, dtype=torch.float32)
        self.stds = torch.tensor(feature_stds, dtype=torch.float32)
        self.n_features = len(feature_means)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature = temperature
        self.leaf_value_fn = leaf_value_fn
        self.leaf_use_policy = leaf_use_policy

        # Pre-allocate buffers
        self._feat_buf = np.zeros(self.n_features, dtype=np.float32)
        self._feat_tensor = torch.zeros(1, self.n_features, dtype=torch.float32)
        self._mask_buf = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._mask_tensor = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        bb_state = game_to_bitboard(game)
        return self.bb_decide(bb_state, playable_actions)

    def bb_decide(self, bb_state, playable_actions):
        """Run MCTS on a BitboardState and return the best action."""
        root = BBMCTSNode(bb_state.copy())

        if not root.is_expanded and not root.is_terminal:
            self._expand(root, root._cached_actions)

        # Dirichlet noise at root
        if self.dirichlet_alpha > 0 and isinstance(root.children, list) and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        # Run simulations
        deadline = time.time() + self.time_limit if self.time_limit else float('inf')
        for _ in range(self.num_simulations):
            if time.time() >= deadline:
                break
            self._simulate(root)

        # Select action
        if self.temperature <= 0:
            # Greedy: most visited
            best_action, _ = root.most_visited_child()
        else:
            # Proportional to visit count^(1/temperature)
            best_action = self._sample_action(root)

        # Map back to original action list
        for a in playable_actions:
            if a.action_type == best_action.action_type and a.value == best_action.value:
                return a
        return playable_actions[0]  # fallback

    def _simulate(self, root):
        """One MCTS simulation: select → materialize → evaluate → expand → backprop."""
        node = root

        # SELECT
        while node.is_expanded and not node.is_terminal:
            if node.is_chance:
                node = self._select_chance(node)
            else:
                _, node = node.best_child(self.c_puct)

        # MATERIALIZE
        if node.bb_state is None:
            node.materialize()

        # After materialization, the node may turn out to be a chance node.
        # If so, mark expanded and continue selecting through it.
        if node.is_chance and not node.is_expanded:
            node.is_expanded = True
            # Sample through the chance node to get a decision leaf
            node = self._select_chance(node)
            if node.bb_state is None:
                node.materialize()

        # EVALUATE
        if node.is_terminal:
            value = self._terminal_value(node)
        else:
            actions = node._cached_actions
            value = self._evaluate_and_expand(node, actions)

        # BACKPROPAGATE
        self._backpropagate(node, value)

    def _evaluate_and_expand(self, node, actions):
        """Evaluate leaf with neural net and expand children with policy priors.

        Merges mask building + prior extraction into a single to_action_space pass.
        Returns value from the perspective of the MCTS player.

        When leaf_value_fn is set, uses external value function + uniform priors
        instead of the AZ network (for blend-bootstrap self-play).
        """
        bb = node.bb_state

        # --- External value + uniform priors (for blend-bootstrap self-play) ---
        if self.leaf_value_fn is not None and not self.leaf_use_policy:
            value = self.leaf_value_fn(bb, self.color)
            uniform = 1.0 / len(actions) if actions else 0.0
            node.children = []
            for action in actions:
                child = BBMCTSNode(None, parent=node, action=action, prior=uniform)
                node.children.append((action, child))
            node.is_expanded = True
            node._cached_actions = None
            return value

        # --- Standard AZ network path (leaf_value_fn overrides value below if set) ---

        # Extract features into pre-allocated buffer
        self._feat_buf[:] = 0.0
        bb_fill_feature_vector(bb, self.color, self._feat_buf, self.fi)

        # Normalize into pre-allocated tensor (no copy — write directly)
        ft = self._feat_tensor
        ft[0] = torch.from_numpy(self._feat_buf)
        ft[0].sub_(self.means).div_(self.stds)

        # Build action mask + collect indices in ONE pass
        self._mask_buf[:] = False
        action_indices = []  # (action, space_index) pairs
        for action in actions:
            try:
                idx = to_action_space(action)
                self._mask_buf[idx] = True
                action_indices.append((action, idx))
            except (ValueError, KeyError):
                action_indices.append((action, -1))

        # Copy mask to pre-allocated tensor
        mt = self._mask_tensor
        np.copyto(mt.numpy()[0], self._mask_buf)

        # Forward pass
        with torch.inference_mode():
            value, policy = self.az_net.predict(ft, mt)

        value = value.item()

        # Override value with external leaf function (keep AZ policy priors)
        if self.leaf_value_fn is not None:
            value = self.leaf_value_fn(bb, self.color)
        policy_np = policy[0].numpy()

        # Build children with priors from the SAME index pass
        total_p = 0.0
        raw_priors = []
        for action, idx in action_indices:
            if idx >= 0:
                p = float(policy_np[idx])
            else:
                p = 0.0
            raw_priors.append(p)
            total_p += p

        node.children = []
        if total_p > 1e-8:
            inv = 1.0 / total_p
            for i, (action, _idx) in enumerate(action_indices):
                prior = raw_priors[i] * inv
                child = BBMCTSNode(None, parent=node, action=action, prior=prior)
                node.children.append((action, child))
        else:
            uniform = 1.0 / len(actions) if actions else 0.0
            for action, _idx in action_indices:
                child = BBMCTSNode(None, parent=node, action=action, prior=uniform)
                node.children.append((action, child))

        node.is_expanded = True
        node._cached_actions = None  # free memory — no longer needed
        return value

    def _expand(self, node, actions):
        """Expand node. Chance nodes are expanded lazily via sampling."""
        if node.is_expanded or node.is_terminal:
            return
        if node.is_chance:
            node.is_expanded = True
            return
        self._evaluate_and_expand(node, actions)

    def _select_chance(self, node, action=None):
        """Sample a chance outcome."""
        if action is None:
            actions = node._cached_actions
            if not actions:
                return node
            action = actions[0]

        if action.action_type == ActionType.ROLL:
            return self._sample_dice(node, action)
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return self._sample_dev_card(node, action)
        else:
            return self._execute_child(node, action, "det")

    def _sample_dice(self, node, action):
        """Sample dice roll weighted by probability."""
        rolls = list(range(2, 13))
        probs = [DICE_PROBS[r] for r in rolls]
        roll = rolls[np.random.choice(len(rolls), p=probs)]

        if roll in node.children:
            return node.children[roll]

        copy = node.bb_state.copy()
        outcome = DICE_OUTCOMES[roll]
        roll_action = Action(action.color, ActionType.ROLL, outcome)
        bb_apply_action(copy, roll_action)

        child = BBMCTSNode(copy, parent=node, action=roll_action)
        node.children[roll] = child
        return child

    def _sample_dev_card(self, node, action):
        """Sample dev card purchase outcome."""
        bb = node.bb_state
        remaining = 25 - bb.dev_deck_idx
        if remaining <= 0:
            return self._execute_child(node, action, "no_cards")

        # Sample from remaining deck
        idx = np.random.randint(bb.dev_deck_idx, 25)
        card_name = DEV_ID_TO_NAME[int(bb.dev_deck[idx])]
        key = f"dev_{card_name}"

        if key in node.children:
            return node.children[key]

        copy = bb.copy()
        from catanatron.models.enums import DEVELOPMENT_CARDS
        card_map = {
            "KNIGHT": DEVELOPMENT_CARDS[0],
            "VICTORY_POINT": DEVELOPMENT_CARDS[1],
            "YEAR_OF_PLENTY": DEVELOPMENT_CARDS[2],
            "MONOPOLY": DEVELOPMENT_CARDS[3],
            "ROAD_BUILDING": DEVELOPMENT_CARDS[4],
        }
        card = card_map.get(card_name)
        buy_action = Action(action.color, ActionType.BUY_DEVELOPMENT_CARD, card)
        bb_apply_action(copy, buy_action)

        child = BBMCTSNode(copy, parent=node, action=buy_action)
        node.children[key] = child
        return child

    def _execute_child(self, node, action, key):
        """Deterministic execute and create child."""
        if key in node.children:
            return node.children[key]

        copy = node.bb_state.copy()
        bb_apply_action(copy, action)
        child = BBMCTSNode(copy, parent=node, action=action)
        node.children[key] = child
        return child

    def _terminal_value(self, node):
        """Value for terminal nodes from MCTS player's perspective."""
        if node._winner_pidx >= 0:
            winner_color = node.bb_state.colors[node._winner_pidx]
            if winner_color == self.color:
                return 1.0
            else:
                return -1.0
        return 0.0  # draw

    def _backpropagate(self, node, value):
        """Walk up tree updating visit counts and values.

        Value is from the MCTS player's perspective:
        - At our decision nodes: store value directly
        - At opponent decision nodes: store -value (opponent wants to minimize us)
        """
        current = node
        while current is not None:
            current.visit_count += 1
            if current.parent is not None and not current.parent.is_chance:
                parent_color = current.parent.bb_state.colors[
                    current.parent.bb_state.current_player_idx
                ]
                if parent_color == self.color:
                    current.value_sum += value
                else:
                    current.value_sum += (-value)
            else:
                current.value_sum += value
            current = current.parent

    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root priors."""
        if not isinstance(node.children, list) or len(node.children) == 0:
            return
        n = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_weight
        for i, (action, child) in enumerate(node.children):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _sample_action(self, root):
        """Sample action proportional to visit_count^(1/temp)."""
        if not isinstance(root.children, list) or len(root.children) == 0:
            return None
        visits = np.array([c.visit_count for _, c in root.children], dtype=np.float64)
        if self.temperature != 1.0:
            visits = visits ** (1.0 / self.temperature)
        total = visits.sum()
        if total == 0:
            idx = np.random.randint(len(root.children))
        else:
            probs = visits / total
            idx = np.random.choice(len(root.children), p=probs)
        return root.children[idx][0]

    def __repr__(self):
        return (
            f"BBMCTSPlayer({self.color}, sims={self.num_simulations}, "
            f"c_puct={self.c_puct})"
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_bb_mcts_player(color, az_checkpoint_path, catan_map=None,
                        num_simulations=800, c_puct=1.4, temperature=0.0,
                        **kwargs):
    """Create a BBMCTSPlayer from an AlphaZero checkpoint.

    Args:
        color: Player color
        az_checkpoint_path: Path to AlphaZero .pt checkpoint
        catan_map: CatanMap instance (needed for FeatureIndexer).
                   If None, must call player.fi.update_map(map) before decide().
        num_simulations: MCTS simulations per decision
        c_puct: Exploration constant
        temperature: Action selection temperature
        **kwargs: Additional args for BBMCTSPlayer

    Returns:
        BBMCTSPlayer instance
    """
    net, ckpt = load_checkpoint(az_checkpoint_path)
    feat_names = ckpt['feature_names']

    # Build feature_index_map: {name: int_index}
    feature_index_map = {name: i for i, name in enumerate(feat_names)}

    # Need a catan_map for FeatureIndexer; use a default one if not provided
    if catan_map is None:
        from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
        catan_map = build_map(BASE_MAP_TEMPLATE)

    fi = FeatureIndexer(feature_index_map, catan_map)

    return BBMCTSPlayer(
        color=color,
        az_net=net,
        feature_indexer=fi,
        feature_means=ckpt['feature_means'],
        feature_stds=ckpt['feature_stds'],
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
        **kwargs,
    )


def create_initial_az_checkpoint(value_ckpt_path, output_path):
    """Create an initial AlphaZero checkpoint by warm-starting from value_net_v2.

    This is the iteration-0 checkpoint used to bootstrap self-play.

    Args:
        value_ckpt_path: Path to value_net_v2.pt
        output_path: Where to save the AlphaZero checkpoint
    """
    from robottler.alphazero_net import (
        CatanAlphaZeroNet, warm_start_from_value_net, save_checkpoint,
    )

    net = CatanAlphaZeroNet(input_dim=176)
    feat_names, feat_means, feat_stds = warm_start_from_value_net(net, value_ckpt_path)

    save_checkpoint(
        net, optimizer=None, iteration=0,
        feature_names=feat_names,
        feature_means=feat_means,
        feature_stds=feat_stds,
        path=output_path,
        extra={'warm_started_from': value_ckpt_path},
    )
    print(f"Saved initial AZ checkpoint to {output_path}")
    return net
