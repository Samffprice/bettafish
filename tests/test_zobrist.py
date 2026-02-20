"""Tests for Zobrist hashing consistency.

Verifies that incremental hash updates match full recomputation
across random games with all action types.
"""

import random
import numpy as np
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from robottler.zobrist import (
    ZobristTracker, EDGE_LIST, EDGE_TO_INDEX, NUM_EDGES,
    Z_SETTLEMENT, Z_CITY, Z_ROAD, Z_ROBBER,
)


def play_random_game_with_tracker(seed):
    """Play one random game, verifying hash consistency after every action.

    Returns (num_actions, num_mismatches).
    """
    random.seed(seed)
    np.random.seed(seed)

    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=seed)

    tracker = ZobristTracker()
    tracker.init(game)

    num_actions = 0
    num_mismatches = 0

    while game.winning_color() is None and game.state.num_turns < 500:
        if not game.playable_actions:
            break

        action = random.choice(game.playable_actions)
        game.execute(action)
        tracker.update(game, action)

        # Verify: recompute from scratch matches incremental
        full_hash = tracker.compute_full(game)
        if tracker.zobrist_hash != full_hash:
            num_mismatches += 1

        num_actions += 1

    return num_actions, num_mismatches


def test_edge_index_mapping():
    """EDGE_LIST should have 72 unique edges, and EDGE_TO_INDEX should work both ways."""
    assert NUM_EDGES == 72, f"Expected 72 edges, got {NUM_EDGES}"
    assert len(EDGE_LIST) == 72

    for idx, (a, b) in enumerate(EDGE_LIST):
        assert a < b, f"Edge {idx} not sorted: ({a}, {b})"
        assert EDGE_TO_INDEX[(a, b)] == idx
        assert EDGE_TO_INDEX[(b, a)] == idx


def test_zobrist_tables_unique():
    """All Zobrist keys in a table should be unique (collision check)."""
    settlement_keys = set(Z_SETTLEMENT.flat)
    assert len(settlement_keys) == Z_SETTLEMENT.size, "Duplicate keys in Z_SETTLEMENT"

    road_keys = set(Z_ROAD.flat)
    assert len(road_keys) == Z_ROAD.size, "Duplicate keys in Z_ROAD"


def test_zobrist_consistency_single_game():
    """Play one game and verify hash consistency."""
    num_actions, num_mismatches = play_random_game_with_tracker(42)
    assert num_actions > 0, "Game produced no actions"
    assert num_mismatches == 0, f"Hash mismatch in {num_mismatches}/{num_actions} actions"


def test_zobrist_consistency_100_games():
    """Play 100 random games, verify hash consistency after every action."""
    total_actions = 0
    total_mismatches = 0

    for seed in range(100):
        num_actions, num_mismatches = play_random_game_with_tracker(seed)
        total_actions += num_actions
        total_mismatches += num_mismatches

    assert total_mismatches == 0, (
        f"Hash mismatches in {total_mismatches}/{total_actions} actions across 100 games"
    )
    print(f"Verified {total_actions} actions across 100 games with 0 mismatches")


def test_zobrist_different_states_different_hashes():
    """Two different game states should (almost certainly) have different hashes."""
    random.seed(123)
    np.random.seed(123)

    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game1 = Game(players, seed=100)
    game2 = Game(players, seed=200)

    tracker1 = ZobristTracker()
    tracker1.init(game1)
    tracker2 = ZobristTracker()
    tracker2.init(game2)

    # Different maps should give different hashes (different robber position at least)
    # This isn't guaranteed but is overwhelmingly likely with 64-bit hashes
    # Actually they might match if board is same. Let's play a few moves first.
    for _ in range(10):
        if game1.playable_actions:
            a = random.choice(game1.playable_actions)
            game1.execute(a)
            tracker1.update(game1, a)
        if game2.playable_actions:
            a = random.choice(game2.playable_actions)
            game2.execute(a)
            tracker2.update(game2, a)

    assert tracker1.zobrist_hash != tracker2.zobrist_hash, (
        "Two different games have the same hash (extremely unlikely)"
    )


def test_zobrist_copy():
    """Tracker copy should maintain independent hash."""
    random.seed(42)
    np.random.seed(42)

    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=42)

    tracker = ZobristTracker()
    tracker.init(game)

    # Play a few moves
    for _ in range(5):
        if game.playable_actions:
            action = random.choice(game.playable_actions)
            game.execute(action)
            tracker.update(game, action)

    # Copy and verify
    copy = tracker.copy()
    assert copy.zobrist_hash == tracker.zobrist_hash

    # Play more moves on original - copy should diverge
    for _ in range(5):
        if game.playable_actions:
            action = random.choice(game.playable_actions)
            game.execute(action)
            tracker.update(game, action)

    assert copy.zobrist_hash != tracker.zobrist_hash


if __name__ == "__main__":
    print("Testing edge index mapping...")
    test_edge_index_mapping()
    print("  PASSED")

    print("Testing Zobrist table uniqueness...")
    test_zobrist_tables_unique()
    print("  PASSED")

    print("Testing single game consistency...")
    test_zobrist_consistency_single_game()
    print("  PASSED")

    print("Testing different states give different hashes...")
    test_zobrist_different_states_different_hashes()
    print("  PASSED")

    print("Testing tracker copy...")
    test_zobrist_copy()
    print("  PASSED")

    print("Testing 100 games consistency (this may take a moment)...")
    test_zobrist_consistency_100_games()
    print("  PASSED")

    print("\nAll Zobrist tests passed!")
