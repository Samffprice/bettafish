"""Dual-engine validation tests for the bitboard Catan engine.

Plays random games using both Catanatron (reference) and the bitboard engine
in parallel, verifying after each action that:
1. game_to_bitboard(game) matches the incrementally updated BitboardState
2. bb_generate_actions(bb) matches game.playable_actions (as sets)
3. State conversion matches (all fields)
"""

import random
import time
import traceback
from collections import Counter

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.enums import ActionType

from robottler.bitboard.convert import game_to_bitboard, compare_states
from robottler.bitboard.actions import bb_apply_action
from robottler.bitboard.movegen import bb_generate_actions


def _action_set(actions):
    """Convert action list to set for comparison, normalizing edge order."""
    result = set()
    for a in actions:
        if a.action_type == ActionType.BUILD_ROAD and a.value is not None:
            # Normalize edge to sorted tuple
            result.add((a.color, a.action_type, tuple(sorted(a.value))))
        else:
            result.add((a.color, a.action_type, a.value))
    return result


def play_dual_engine_game(seed, verbose=False):
    """Play one game with both engines in parallel.

    Returns (num_actions, state_diffs, action_diffs) where:
    - state_diffs is count of state comparison failures
    - action_diffs is count of action set mismatches
    """
    random.seed(seed)
    players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
    game = Game(players, seed=seed)

    # Initialize bitboard from game
    bb = game_to_bitboard(game)

    num_actions = 0
    state_diffs = 0
    action_diffs = 0
    first_state_diff = None
    first_action_diff = None

    while game.winning_color() is None and game.state.num_turns < 500:
        if not game.playable_actions:
            break

        # Compare legal actions
        try:
            bb_actions = bb_generate_actions(bb)
            ref_set = _action_set(game.playable_actions)
            bb_set = _action_set(bb_actions)

            if ref_set != bb_set:
                action_diffs += 1
                if first_action_diff is None:
                    missing = ref_set - bb_set
                    extra = bb_set - ref_set
                    first_action_diff = (
                        f"Action {num_actions}: "
                        f"missing={list(missing)[:5]}, extra={list(extra)[:5]}"
                    )
                    if verbose:
                        print(f"  ACTION DIFF at action {num_actions}:")
                        print(f"    missing from bb: {list(missing)[:5]}")
                        print(f"    extra in bb: {list(extra)[:5]}")
        except Exception as e:
            action_diffs += 1
            if first_action_diff is None:
                first_action_diff = f"Action {num_actions}: movegen error: {e}"
            if verbose:
                print(f"  MOVEGEN ERROR at action {num_actions}: {e}")

        # Pick a random action and apply to both
        action = random.choice(game.playable_actions)
        action_record = game.execute(action)

        try:
            bb_apply_action(bb, action, action_record)
        except Exception as e:
            state_diffs += 1
            if first_state_diff is None:
                first_state_diff = f"Action {num_actions}: apply error: {e}"
            if verbose:
                print(f"  APPLY ERROR at action {num_actions}: {e}")
                traceback.print_exc()
            # Re-sync from game state
            bb = game_to_bitboard(game)
            num_actions += 1
            continue

        # Compare states by re-converting from game
        ref_bb = game_to_bitboard(game)
        diffs = compare_states(game, bb)
        if diffs:
            state_diffs += 1
            if first_state_diff is None:
                first_state_diff = f"Action {num_actions} ({action.action_type}): {diffs[:3]}"
            if verbose:
                print(f"  STATE DIFF at action {num_actions} ({action.action_type}):")
                for d in diffs[:5]:
                    print(f"    {d}")
                # Debug road state
                print(f"    bb road_holder={bb.road_holder} len={bb.road_holder_length}")
                print(f"    bb road_lengths={list(bb.road_lengths)}")
                print(f"    ref road_color={game.state.board.road_color} len={game.state.board.road_length}")
                print(f"    ref road_lengths={dict(game.state.board.road_lengths)}")
            # Re-sync
            bb = game_to_bitboard(game)

        num_actions += 1

    return num_actions, state_diffs, action_diffs, first_state_diff, first_action_diff


def test_single_game():
    """Play one game with dual-engine validation."""
    num_actions, sd, ad, fsd, fad = play_dual_engine_game(42, verbose=True)
    print(f"Game 42: {num_actions} actions, {sd} state diffs, {ad} action diffs")
    if fsd:
        print(f"  First state diff: {fsd}")
    if fad:
        print(f"  First action diff: {fad}")
    assert sd == 0, f"State diffs: {sd}. First: {fsd}"
    assert ad == 0, f"Action diffs: {ad}. First: {fad}"


def test_20_games():
    """Play 20 games with dual-engine validation."""
    total_actions = 0
    total_sd = 0
    total_ad = 0
    first_sd = None
    first_ad = None

    t0 = time.time()
    for seed in range(20):
        na, sd, ad, fsd, fad = play_dual_engine_game(seed)
        total_actions += na
        total_sd += sd
        total_ad += ad
        if sd > 0 and first_sd is None:
            first_sd = f"Seed {seed}: {fsd}"
        if ad > 0 and first_ad is None:
            first_ad = f"Seed {seed}: {fad}"

    elapsed = time.time() - t0
    print(f"20 games: {total_actions} actions, {total_sd} state diffs, "
          f"{total_ad} action diffs in {elapsed:.2f}s")

    if first_sd:
        print(f"  First state diff: {first_sd}")
    if first_ad:
        print(f"  First action diff: {first_ad}")

    assert total_sd == 0, f"Total state diffs: {total_sd}"
    assert total_ad == 0, f"Total action diffs: {total_ad}"


if __name__ == "__main__":
    print("Testing single game...")
    test_single_game()
    print("  PASSED\n")

    print("Testing 20 games...")
    test_20_games()
    print("  PASSED\n")

    print("All dual-engine tests passed!")
