"""Feature equivalence test: bb_create_sample dict vs bb_fill_feature_vector direct.

Verifies that the direct vector extraction produces identical values
to the dict-based approach across many random game states.
"""

import random
import time
import numpy as np
import torch

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import ActionType

from robottler.bitboard.convert import game_to_bitboard
from robottler.bitboard.features import (
    bb_create_sample, bb_fill_feature_vector, FeatureIndexer,
)
from robottler.bitboard.actions import bb_apply_action


def test_feature_equivalence(num_seeds=20, actions_per_game=10, verbose=True):
    """Compare dict-based vs direct vector extraction across random states.

    For each seed, plays actions_per_game random actions, then compares
    the feature vectors produced by both methods.
    """
    # Load checkpoint to get feature names
    bc_path = "robottler/models/value_net_v2.pt"
    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    n_features = len(feature_names)
    feature_index_map = {name: idx for idx, name in enumerate(feature_names)}

    if verbose:
        print(f"Checkpoint has {n_features} features")
        print(f"First 5: {feature_names[:5]}")
        print(f"Last 5: {feature_names[-5:]}")

    total_checks = 0
    total_mismatches = 0
    first_mismatch = None
    fi = None

    t0 = time.time()

    for seed in range(num_seeds):
        random.seed(seed)
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = Game(players, seed=seed)
        bb = game_to_bitboard(game)

        if fi is None:
            fi = FeatureIndexer(feature_index_map, bb.catan_map)
        else:
            fi.update_map(bb.catan_map)

        # Play some random actions
        actions_played = 0
        while actions_played < actions_per_game:
            if game.winning_color() is not None:
                break
            if not game.playable_actions:
                break
            action = random.choice(game.playable_actions)
            action_record = game.execute(action)
            try:
                bb_apply_action(bb, action, action_record)
            except Exception:
                bb = game_to_bitboard(game)
            actions_played += 1

        # Compare features for both colors
        for color in [Color.RED, Color.BLUE]:
            # Dict-based (reference)
            sample = bb_create_sample(bb, color)
            old_vec = np.array(
                [float(sample.get(f, 0.0)) for f in feature_names],
                dtype=np.float64,
            )

            # Direct vector (new)
            new_vec = np.zeros(n_features, dtype=np.float64)
            bb_fill_feature_vector(bb, color, new_vec, fi)

            total_checks += 1

            if not np.allclose(old_vec, new_vec, atol=1e-6):
                total_mismatches += 1
                diffs = np.where(~np.isclose(old_vec, new_vec, atol=1e-6))[0]
                if first_mismatch is None:
                    diff_details = []
                    for d in diffs[:10]:
                        diff_details.append(
                            f"  [{d}] {feature_names[d]}: old={old_vec[d]:.6f} new={new_vec[d]:.6f}"
                        )
                    first_mismatch = (
                        f"Seed {seed}, {color}, action {actions_played}: "
                        f"{len(diffs)} features differ:\n" + "\n".join(diff_details)
                    )
                if verbose:
                    print(f"  MISMATCH seed={seed} color={color}: {len(diffs)} diffs")

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{total_checks} checks across {num_seeds} seeds in {elapsed:.2f}s")
        print(f"Mismatches: {total_mismatches}")
        if first_mismatch:
            print(f"First mismatch:\n{first_mismatch}")

    assert total_mismatches == 0, (
        f"{total_mismatches}/{total_checks} mismatches. First:\n{first_mismatch}"
    )
    print("Feature equivalence: PASSED")


def test_neural_value_equivalence(num_seeds=5, verbose=True):
    """Verify that the optimized neural value function produces identical outputs."""
    from robottler.search_player import make_bb_neural_value_fn
    from robottler.bitboard.features import bb_create_sample
    from robottler.value_model import CatanValueNet

    bc_path = "robottler/models/value_net_v2.pt"

    # Old-style value function (dict-based)
    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    means_t = torch.tensor(checkpoint["feature_means"], dtype=torch.float32)
    stds_t = torch.tensor(checkpoint["feature_stds"], dtype=torch.float32)
    old_model = CatanValueNet(input_dim=len(feature_names))
    old_model.load_state_dict(checkpoint["model_state_dict"])
    old_model.eval()

    def old_value_fn(bb_state, p0_color):
        sample = bb_create_sample(bb_state, p0_color)
        vec = [float(sample.get(f, 0.0)) for f in feature_names]
        x = torch.tensor(vec, dtype=torch.float32)
        x = (x - means_t) / stds_t
        with torch.no_grad():
            logit = old_model(x.unsqueeze(0))
            return torch.sigmoid(logit).item()

    # New-style value function (direct vector)
    new_value_fn = make_bb_neural_value_fn(bc_path)

    total_checks = 0
    max_diff = 0.0

    for seed in range(num_seeds):
        random.seed(seed)
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = Game(players, seed=seed)
        bb = game_to_bitboard(game)

        # Play some actions
        for _ in range(15):
            if game.winning_color() is not None or not game.playable_actions:
                break
            action = random.choice(game.playable_actions)
            action_record = game.execute(action)
            try:
                bb_apply_action(bb, action, action_record)
            except Exception:
                bb = game_to_bitboard(game)

        for color in [Color.RED, Color.BLUE]:
            old_val = old_value_fn(bb, color)
            new_val = new_value_fn(bb, color)
            diff = abs(old_val - new_val)
            max_diff = max(max_diff, diff)
            total_checks += 1

            if diff > 1e-5:
                print(f"  DIFF seed={seed} color={color}: old={old_val:.8f} new={new_val:.8f} diff={diff:.2e}")

    if verbose:
        print(f"Neural value equivalence: {total_checks} checks, max_diff={max_diff:.2e}")

    # Allow small float32 rounding differences
    assert max_diff < 1e-4, f"Max diff {max_diff:.2e} exceeds threshold"
    print("Neural value equivalence: PASSED")


def test_heuristic_equivalence(num_seeds=5, verbose=True):
    """Verify that numpy-based heuristic matches dict-based heuristic."""
    bc_path = "robottler/models/value_net_v2.pt"
    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_index_map = {name: idx for idx, name in enumerate(checkpoint["feature_names"])}

    from robottler.bitboard.features import FeatureIndexer

    fi = None
    total_checks = 0
    max_diff = 0.0

    # Import old heuristic internals for reference
    from robottler.bitboard.features import (
        _iter_players, _production_features, _reachability_features,
    )
    from robottler.bitboard.state import (
        PS_VP, PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
        PS_LONGEST_ROAD, PS_PLAYED_KNIGHT,
        PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
        PS_RESOURCE_START, PS_RESOURCE_END,
    )
    from robottler.bitboard.masks import (
        NUM_TILES, TILE_NODES, NODE_BIT, popcount64, bitscan,
    )
    from robottler.bitboard.features import compute_production_np, compute_reachability_np
    from catanatron.models.enums import RESOURCES
    from catanatron.players.value import DEFAULT_WEIGHTS

    for seed in range(num_seeds):
        random.seed(seed)
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = Game(players, seed=seed)
        bb = game_to_bitboard(game)

        if fi is None:
            fi = FeatureIndexer(feature_index_map, bb.catan_map)
        else:
            fi.update_map(bb.catan_map)

        for _ in range(15):
            if game.winning_color() is not None or not game.playable_actions:
                break
            action = random.choice(game.playable_actions)
            action_record = game.execute(action)
            try:
                bb_apply_action(bb, action, action_record)
            except Exception:
                bb = game_to_bitboard(game)

        for color in [Color.RED, Color.BLUE]:
            p0_idx = bb.color_to_index[color]

            # Old-style: dict-based production
            feats = {}
            _production_features(bb, p0_idx, feats, consider_robber=True, prefix="EFFECTIVE_")
            old_prod_0 = np.array([feats[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES])
            old_prod_1 = np.array([feats[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RESOURCES])

            # New-style: numpy production
            p1_idx = (p0_idx + 1) % bb.num_players
            new_prod_0 = compute_production_np(bb, p0_idx, fi.node_prod, consider_robber=True)
            new_prod_1 = compute_production_np(bb, p1_idx, fi.node_prod, consider_robber=True)

            prod_diff = max(np.max(np.abs(old_prod_0 - new_prod_0)),
                          np.max(np.abs(old_prod_1 - new_prod_1)))
            max_diff = max(max_diff, prod_diff)

            # Old-style: dict-based reachability
            reach_feats = {}
            _reachability_features(bb, p0_idx, reach_feats, max_level=1)
            old_reach_0 = np.array([reach_feats.get(f"P0_0_ROAD_REACHABLE_{r}", 0.0) for r in RESOURCES])
            old_reach_1 = np.array([reach_feats.get(f"P0_1_ROAD_REACHABLE_{r}", 0.0) for r in RESOURCES])

            # New-style: numpy reachability
            reach = compute_reachability_np(bb, p0_idx, fi.node_prod, max_level=1)
            new_reach_0 = reach[0]
            new_reach_1 = reach[1]

            reach_diff = max(np.max(np.abs(old_reach_0 - new_reach_0)),
                           np.max(np.abs(old_reach_1 - new_reach_1)))
            max_diff = max(max_diff, reach_diff)

            total_checks += 1

    if verbose:
        print(f"Heuristic equivalence: {total_checks} checks, max_diff={max_diff:.2e}")

    assert max_diff < 1e-10, f"Max diff {max_diff:.2e} exceeds threshold"
    print("Heuristic equivalence: PASSED")


if __name__ == "__main__":
    print("Testing feature equivalence (20 seeds)...")
    test_feature_equivalence(num_seeds=20, actions_per_game=15)
    print()

    print("Testing heuristic equivalence (5 seeds)...")
    test_heuristic_equivalence(num_seeds=5)
    print()

    print("Testing neural value equivalence (5 seeds)...")
    test_neural_value_equivalence(num_seeds=5)
    print()

    print("All direct vector tests passed!")
