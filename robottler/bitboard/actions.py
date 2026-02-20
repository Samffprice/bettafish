"""Action application for BitboardState.

bb_apply_action(state, action) mutates a BitboardState in-place,
mirroring Catanatron's apply_action.py logic.
"""

import random
import numpy as np
from catanatron.models.enums import (
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT,
    SETTLEMENT, CITY,
    ActionType, Action, ActionRecord,
)
from catanatron.models.decks import (
    ROAD_COST_FREQDECK, SETTLEMENT_COST_FREQDECK, CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    freqdeck_from_listdeck,
)
from robottler.bitboard.masks import (
    NUM_NODES, NUM_EDGES, NUM_TILES,
    NODE_BIT, TILE_NODES, EDGE_TO_INDEX, EDGE_ENDPOINTS, INCIDENT_EDGES,
    RESOURCE_INDEX, popcount64, bitscan, HAS_CYTHON,
)
from robottler.bitboard.state import (
    PS_VP, PS_ROADS_AVAIL, PS_SETTLE_AVAIL, PS_CITY_AVAIL,
    PS_HAS_ROAD, PS_HAS_ARMY, PS_HAS_ROLLED, PS_HAS_PLAYED_DEV,
    PS_ACTUAL_VP, PS_LONGEST_ROAD,
    PS_KNIGHT_START, PS_MONO_START, PS_YOP_START, PS_RB_START,
    PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
    PS_KNIGHT_HAND, PS_PLAYED_KNIGHT,
    PS_YOP_HAND, PS_PLAYED_YOP,
    PS_MONO_HAND, PS_PLAYED_MONO,
    PS_RB_HAND, PS_PLAYED_RB,
    PS_VP_HAND, PS_PLAYED_VP,
    PS_RESOURCE_START, PS_RESOURCE_END,
    DEV_HAND_INDICES, DEV_PLAYED_INDICES, DEV_START_INDICES,
    PROMPT_BUILD_INITIAL_SETTLEMENT, PROMPT_BUILD_INITIAL_ROAD,
    PROMPT_PLAY_TURN, PROMPT_DISCARD, PROMPT_MOVE_ROBBER,
    PROMPT_DECIDE_TRADE, PROMPT_DECIDE_ACCEPTEES,
)
from robottler.bitboard.convert import DEV_CARD_ID, DEV_ID_TO_NAME

# Resource index array for iteration
_RESOURCE_NAMES = [WOOD, BRICK, SHEEP, WHEAT, ORE]
_RESOURCE_PS = [PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE]

# Dev card hand → start-of-turn mapping
_DEV_HAND_TO_START = {
    PS_KNIGHT_HAND: PS_KNIGHT_START,
    PS_MONO_HAND: PS_MONO_START,
    PS_YOP_HAND: PS_YOP_START,
    PS_RB_HAND: PS_RB_START,
}


def bb_apply_action(state, action, action_record=None):
    """Apply an action to a BitboardState. Mirrors Catanatron's apply_action.

    Args:
        state: BitboardState to mutate
        action: Catanatron Action namedtuple
        action_record: Optional ActionRecord for replay (dice rolls, steals, dev card draws)

    Returns:
        ActionRecord
    """
    atype = action.action_type
    color = action.color
    pidx = state.color_to_index[color]

    if atype == ActionType.BUILD_SETTLEMENT:
        return _apply_build_settlement(state, action, pidx)
    elif atype == ActionType.BUILD_ROAD:
        return _apply_build_road(state, action, pidx)
    elif atype == ActionType.BUILD_CITY:
        return _apply_build_city(state, action, pidx)
    elif atype == ActionType.BUY_DEVELOPMENT_CARD:
        return _apply_buy_dev_card(state, action, pidx, action_record)
    elif atype == ActionType.ROLL:
        return _apply_roll(state, action, pidx, action_record)
    elif atype == ActionType.DISCARD:
        return _apply_discard(state, action, pidx, action_record)
    elif atype == ActionType.MOVE_ROBBER:
        return _apply_move_robber(state, action, pidx, action_record)
    elif atype == ActionType.PLAY_KNIGHT_CARD:
        return _apply_play_knight(state, action, pidx)
    elif atype == ActionType.PLAY_YEAR_OF_PLENTY:
        return _apply_play_yop(state, action, pidx)
    elif atype == ActionType.PLAY_MONOPOLY:
        return _apply_play_monopoly(state, action, pidx)
    elif atype == ActionType.PLAY_ROAD_BUILDING:
        return _apply_play_road_building(state, action, pidx)
    elif atype == ActionType.MARITIME_TRADE:
        return _apply_maritime_trade(state, action, pidx)
    elif atype == ActionType.END_TURN:
        return _apply_end_turn(state, action, pidx)
    elif atype == ActionType.OFFER_TRADE:
        return _apply_offer_trade(state, action, pidx)
    elif atype == ActionType.ACCEPT_TRADE:
        return _apply_accept_trade(state, action, pidx)
    elif atype == ActionType.REJECT_TRADE:
        return _apply_reject_trade(state, action, pidx)
    elif atype == ActionType.CONFIRM_TRADE:
        return _apply_confirm_trade(state, action, pidx)
    elif atype == ActionType.CANCEL_TRADE:
        return _apply_cancel_trade(state, action, pidx)
    else:
        raise ValueError(f"Unknown action type: {atype}")


# --- Handlers ---

def _apply_build_settlement(state, action, pidx):
    node_id = action.value
    ps = state.player_state[pidx]

    if state.is_initial_build_phase:
        state.build_settlement_initial(pidx, node_id)
        ps[PS_SETTLE_AVAIL] -= 1
        ps[PS_VP] += 1
        ps[PS_ACTUAL_VP] += 1

        # Yield resources if second settlement
        settlements_count = popcount64(state.settlement_bb[pidx])
        if settlements_count == 2:
            for tid in range(NUM_TILES):
                if TILE_NODES[tid] & NODE_BIT[node_id]:
                    res = state.tile_resource[tid]
                    if res >= 0:
                        ps[PS_RESOURCE_START + res] += 1
                        state.bank[res] -= 1

        state.current_prompt = PROMPT_BUILD_INITIAL_ROAD
    else:
        prev_holder, new_holder = state.build_settlement_normal(pidx, node_id)
        ps[PS_SETTLE_AVAIL] -= 1
        ps[PS_VP] += 1
        ps[PS_ACTUAL_VP] += 1

        # Pay cost
        ps[PS_WOOD] -= 1
        ps[PS_BRICK] -= 1
        ps[PS_SHEEP] -= 1
        ps[PS_WHEAT] -= 1

        # Replenish bank
        state.bank[0] += 1  # WOOD
        state.bank[1] += 1  # BRICK
        state.bank[2] += 1  # SHEEP
        state.bank[3] += 1  # WHEAT

        _maintain_longest_road(state, prev_holder, new_holder)

    return ActionRecord(action=action, result=None)


def _apply_build_road(state, action, pidx):
    edge = action.value
    eidx = EDGE_TO_INDEX[tuple(sorted(edge))]
    ps = state.player_state[pidx]

    if state.is_initial_build_phase:
        prev_holder, new_holder = state.build_road(pidx, eidx)
        ps[PS_ROADS_AVAIL] -= 1
        # No cost in initial phase

        # Determine if we advance turn
        # Count total settlements across all players
        total_settlements = 0
        for p in range(state.num_players):
            total_settlements += popcount64(state.settlement_bb[p])
        num_players = state.num_players
        going_forward = total_settlements < num_players
        at_the_end = total_settlements == num_players
        done = total_settlements == 2 * num_players

        if going_forward:
            _advance_turn(state)
            state.current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT
        elif at_the_end:
            state.current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT
        elif done:
            state.is_initial_build_phase = False
            state.current_prompt = PROMPT_PLAY_TURN
        else:
            _advance_turn(state, -1)
            state.current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT

    elif state.is_road_building and state.free_roads > 0:
        prev_holder, new_holder = state.build_road(pidx, eidx)
        ps[PS_ROADS_AVAIL] -= 1
        # Free road, no cost
        _maintain_longest_road(state, prev_holder, new_holder)

        state.free_roads -= 1
        if state.free_roads == 0 or not _has_buildable_road_edges(state, pidx):
            state.is_road_building = False
            state.free_roads = 0
    else:
        prev_holder, new_holder = state.build_road(pidx, eidx)
        ps[PS_ROADS_AVAIL] -= 1

        # Pay cost
        ps[PS_WOOD] -= 1
        ps[PS_BRICK] -= 1
        state.bank[0] += 1  # WOOD
        state.bank[1] += 1  # BRICK
        _maintain_longest_road(state, prev_holder, new_holder)

    return ActionRecord(action=action, result=None)


def _apply_build_city(state, action, pidx):
    node_id = action.value
    ps = state.player_state[pidx]

    state.build_city(pidx, node_id)
    ps[PS_SETTLE_AVAIL] += 1
    ps[PS_CITY_AVAIL] -= 1
    ps[PS_VP] += 1
    ps[PS_ACTUAL_VP] += 1

    # Pay cost
    ps[PS_WHEAT] -= 2
    ps[PS_ORE] -= 3
    state.bank[3] += 2  # WHEAT
    state.bank[4] += 3  # ORE

    return ActionRecord(action=action, result=None)


def _apply_buy_dev_card(state, action, pidx, action_record=None):
    ps = state.player_state[pidx]

    if action_record is not None:
        card = action_record.result
    elif action.value is not None:
        # execute_spectrum passes the card in action.value
        card = action.value
    else:
        # Draw from deck
        remaining = 25 - state.dev_deck_idx
        if remaining <= 0:
            raise ValueError("No more dev cards")
        card_id = state.dev_deck[state.dev_deck_idx]
        card = DEV_ID_TO_NAME[int(card_id)]

    # Always increment deck index when a card is bought
    state.dev_deck_idx += 1

    # Pay cost
    ps[PS_SHEEP] -= 1
    ps[PS_WHEAT] -= 1
    ps[PS_ORE] -= 1
    state.bank[2] += 1  # SHEEP
    state.bank[3] += 1  # WHEAT
    state.bank[4] += 1  # ORE

    # Add card to hand
    dev_idx = _DEV_NAME_TO_HAND_IDX[card]
    ps[dev_idx] += 1

    if card == VICTORY_POINT:
        ps[PS_ACTUAL_VP] += 1

    action = Action(action.color, action.action_type, card)
    return ActionRecord(action=action, result=card)


def _apply_roll(state, action, pidx, action_record=None):
    ps = state.player_state[pidx]
    ps[PS_HAS_ROLLED] = 1

    if action_record is not None:
        dices = action_record.result
    elif action.value is not None:
        dices = action.value
    else:
        dices = (random.randint(1, 6), random.randint(1, 6))

    number = dices[0] + dices[1]
    action = Action(action.color, action.action_type, dices)

    if number == 7:
        # Check for discarders
        discarder_idx = -1
        for p in range(state.num_players):
            total = int(np.sum(state.player_state[p, PS_RESOURCE_START:PS_RESOURCE_END]))
            if total > state.discard_limit:
                discarder_idx = p
                break

        if discarder_idx >= 0:
            state.current_player_idx = discarder_idx
            state.is_discarding = True
            state.current_prompt = PROMPT_DISCARD
        else:
            state.is_moving_knight = True
            state.current_prompt = PROMPT_MOVE_ROBBER
    else:
        _yield_resources(state, number)
        # current_prompt stays PLAY_TURN

    return ActionRecord(action=action, result=dices)


def _apply_discard(state, action, pidx, action_record=None):
    ps = state.player_state[pidx]

    # Calculate total hand and discard half
    total = int(np.sum(ps[PS_RESOURCE_START:PS_RESOURCE_END]))
    num_to_discard = total // 2

    if action_record is not None:
        discarded = action_record.result
    else:
        # Build hand array and randomly sample
        hand = []
        for ri, rname in enumerate(_RESOURCE_NAMES):
            count = int(ps[PS_RESOURCE_START + ri])
            hand.extend([rname] * count)
        discarded = random.sample(hand, k=num_to_discard)

    # Apply discard
    to_discard = freqdeck_from_listdeck(discarded)
    for ri in range(5):
        ps[PS_RESOURCE_START + ri] -= to_discard[ri]
        state.bank[ri] += to_discard[ri]

    action = Action(action.color, action.action_type, discarded)

    # Find next discarder
    found_next = False
    for p in range(state.current_player_idx + 1, state.num_players):
        total = int(np.sum(state.player_state[p, PS_RESOURCE_START:PS_RESOURCE_END]))
        if total > 7:
            state.current_player_idx = p
            found_next = True
            break
            # current_prompt stays DISCARD

    if not found_next:
        state.current_player_idx = state.current_turn_idx
        state.is_discarding = False
        state.is_moving_knight = True
        state.current_prompt = PROMPT_MOVE_ROBBER

    return ActionRecord(action=action, result=discarded)


def _apply_move_robber(state, action, pidx, action_record=None):
    coord, robbed_color = action.value
    robbed_resource = None

    # Move robber
    if coord in state.tile_coord_to_id:
        state.robber_tile = np.int8(state.tile_coord_to_id[coord])

    if robbed_color is not None:
        robbed_pidx = state.color_to_index[robbed_color]
        if action_record is not None:
            robbed_resource = action_record.result
        else:
            # Random steal
            hand = []
            for ri, rname in enumerate(_RESOURCE_NAMES):
                count = int(state.player_state[robbed_pidx, PS_RESOURCE_START + ri])
                hand.extend([rname] * count)
            if hand:
                robbed_resource = random.choice(hand)

        if robbed_resource is not None:
            ri = RESOURCE_INDEX[robbed_resource]
            state.player_state[robbed_pidx, PS_RESOURCE_START + ri] -= 1
            state.player_state[pidx, PS_RESOURCE_START + ri] += 1

    # Note: Catanatron does NOT reset is_moving_knight in apply_move_robber.
    # It only sets current_prompt = PLAY_TURN. We mirror that behavior.
    state.current_prompt = PROMPT_PLAY_TURN

    return ActionRecord(action=action, result=robbed_resource)


def _apply_play_knight(state, action, pidx):
    ps = state.player_state[pidx]

    # Record previous army holder
    prev_army_holder = state.army_holder
    prev_army_size = state.army_holder_size

    # Play card
    ps[PS_KNIGHT_HAND] -= 1
    ps[PS_HAS_PLAYED_DEV] = 1
    ps[PS_PLAYED_KNIGHT] += 1

    # Check largest army
    knight_count = int(ps[PS_PLAYED_KNIGHT])
    _maintain_largest_army(state, pidx, prev_army_holder, prev_army_size, knight_count)

    # Note: Catanatron does NOT set is_moving_knight for knight card play.
    # is_moving_knight is only set during 7-roll/discard sequences.
    state.current_prompt = PROMPT_MOVE_ROBBER

    return ActionRecord(action=action, result=None)


def _apply_play_yop(state, action, pidx):
    ps = state.player_state[pidx]
    cards_selected = freqdeck_from_listdeck(action.value)

    for ri in range(5):
        ps[PS_RESOURCE_START + ri] += cards_selected[ri]
        state.bank[ri] -= cards_selected[ri]

    ps[PS_YOP_HAND] -= 1
    ps[PS_HAS_PLAYED_DEV] = 1
    ps[PS_PLAYED_YOP] += 1

    return ActionRecord(action=action, result=None)


def _apply_play_monopoly(state, action, pidx):
    ps = state.player_state[pidx]
    mono_resource = action.value
    ri = RESOURCE_INDEX[mono_resource]

    total_stolen = 0
    for p in range(state.num_players):
        if p == pidx:
            continue
        amount = int(state.player_state[p, PS_RESOURCE_START + ri])
        state.player_state[p, PS_RESOURCE_START + ri] = 0
        total_stolen += amount

    ps[PS_RESOURCE_START + ri] += total_stolen

    ps[PS_MONO_HAND] -= 1
    ps[PS_HAS_PLAYED_DEV] = 1
    ps[PS_PLAYED_MONO] += 1

    return ActionRecord(action=action, result=None)


def _apply_play_road_building(state, action, pidx):
    ps = state.player_state[pidx]
    ps[PS_RB_HAND] -= 1
    ps[PS_HAS_PLAYED_DEV] = 1
    ps[PS_PLAYED_RB] += 1

    state.is_road_building = True
    state.free_roads = 2

    return ActionRecord(action=action, result=None)


def _apply_maritime_trade(state, action, pidx):
    ps = state.player_state[pidx]
    trade_offer = action.value

    # Offering (first 4 elements, some may be None)
    offering = freqdeck_from_listdeck(r for r in trade_offer[:-1] if r is not None)
    # Asking (last element)
    asking = freqdeck_from_listdeck(trade_offer[-1:])

    for ri in range(5):
        ps[PS_RESOURCE_START + ri] -= offering[ri]
        state.bank[ri] += offering[ri]
        ps[PS_RESOURCE_START + ri] += asking[ri]
        state.bank[ri] -= asking[ri]

    return ActionRecord(action=action, result=None)


def _apply_end_turn(state, action, pidx):
    ps = state.player_state[pidx]

    # Clean turn
    ps[PS_HAS_PLAYED_DEV] = 0
    ps[PS_HAS_ROLLED] = 0

    # Update owned-at-start flags
    ps[PS_KNIGHT_START] = int(ps[PS_KNIGHT_HAND] > 0)
    ps[PS_MONO_START] = int(ps[PS_MONO_HAND] > 0)
    ps[PS_YOP_START] = int(ps[PS_YOP_HAND] > 0)
    ps[PS_RB_START] = int(ps[PS_RB_HAND] > 0)

    _advance_turn(state)
    state.current_prompt = PROMPT_PLAY_TURN
    return ActionRecord(action=action, result=None)


def _apply_offer_trade(state, action, pidx):
    state.is_resolving_trade = True
    # Find first non-offering player
    for i in range(state.num_players):
        if state.colors[i] != action.color:
            state.current_player_idx = i
            break
    state.current_prompt = PROMPT_DECIDE_TRADE
    return ActionRecord(action=action, result=None)


def _apply_accept_trade(state, action, pidx):
    # Find next non-offering player after current
    found = False
    offering_color = state.colors[state.current_turn_idx]
    for i in range(state.current_player_idx + 1, state.num_players):
        if state.colors[i] != offering_color:
            state.current_player_idx = i
            found = True
            break
            # current_prompt stays DECIDE_TRADE
    if not found:
        # At least 1 acceptee (this player), go to DECIDE_ACCEPTEES
        state.current_player_idx = state.current_turn_idx
        state.current_prompt = PROMPT_DECIDE_ACCEPTEES
    return ActionRecord(action=action, result=None)


def _apply_reject_trade(state, action, pidx):
    offering_color = state.colors[state.current_turn_idx]
    found = False
    for i in range(state.current_player_idx + 1, state.num_players):
        if state.colors[i] != offering_color:
            state.current_player_idx = i
            found = True
            break
            # current_prompt stays DECIDE_TRADE
    if not found:
        # No acceptees (simplified — search rejects all trades)
        _reset_trading_state(state)
        state.current_player_idx = state.current_turn_idx
        state.current_prompt = PROMPT_PLAY_TURN
    return ActionRecord(action=action, result=None)


def _apply_confirm_trade(state, action, pidx):
    offering = action.value[:5]
    asking = action.value[5:10]
    enemy_color = action.value[10]
    enemy_pidx = state.color_to_index[enemy_color]

    for ri in range(5):
        state.player_state[pidx, PS_RESOURCE_START + ri] -= offering[ri]
        state.player_state[pidx, PS_RESOURCE_START + ri] += asking[ri]
        state.player_state[enemy_pidx, PS_RESOURCE_START + ri] -= asking[ri]
        state.player_state[enemy_pidx, PS_RESOURCE_START + ri] += offering[ri]

    _reset_trading_state(state)
    state.current_player_idx = state.current_turn_idx
    state.current_prompt = PROMPT_PLAY_TURN
    return ActionRecord(action=action, result=None)


def _apply_cancel_trade(state, action, pidx):
    _reset_trading_state(state)
    state.current_player_idx = state.current_turn_idx
    state.current_prompt = PROMPT_PLAY_TURN
    return ActionRecord(action=action, result=None)


# --- Helpers ---

_DEV_NAME_TO_HAND_IDX = {
    KNIGHT: PS_KNIGHT_HAND,
    YEAR_OF_PLENTY: PS_YOP_HAND,
    MONOPOLY: PS_MONO_HAND,
    ROAD_BUILDING: PS_RB_HAND,
    VICTORY_POINT: PS_VP_HAND,
}


def _advance_turn(state, direction=1):
    next_idx = (state.current_player_idx + direction) % state.num_players
    state.current_player_idx = next_idx
    state.current_turn_idx = next_idx
    state.num_turns += 1


def _yield_resources(state, number):
    """Compute and distribute resource yields for a dice roll."""
    if HAS_CYTHON:
        from robottler.bitboard._fast import yield_resources_c
        yield_resources_c(
            number, state.num_players, int(state.robber_tile),
            state.tile_number, state.tile_resource, TILE_NODES,
            state.settlement_bb, state.city_bb,
            state.player_state, state.bank, PS_RESOURCE_START,
        )
        return

    _yield_resources_py(state, number)


def _yield_resources_py(state, number):
    """Python fallback for resource distribution."""
    # First pass: compute intended payouts and check for depletion
    intended = np.zeros((state.num_players, 5), dtype=np.int16)

    for tid in range(NUM_TILES):
        if state.tile_number[tid] != number:
            continue
        if tid == state.robber_tile:
            continue
        res = int(state.tile_resource[tid])
        if res < 0:
            continue

        tile_mask = TILE_NODES[tid]
        for p in range(state.num_players):
            s_count = popcount64(state.settlement_bb[p] & tile_mask)
            c_count = popcount64(state.city_bb[p] & tile_mask)
            intended[p, res] += s_count + 2 * c_count

    # Check for depletion: if total demand for a resource exceeds bank supply
    for ri in range(5):
        total_demand = int(np.sum(intended[:, ri]))
        if total_demand > state.bank[ri]:
            intended[:, ri] = 0  # nobody gets this resource

    # Apply payouts
    for p in range(state.num_players):
        for ri in range(5):
            amount = int(intended[p, ri])
            if amount > 0:
                state.player_state[p, PS_RESOURCE_START + ri] += amount
                state.bank[ri] -= amount


def _maintain_longest_road(state, prev_holder, new_holder):
    """Update VP for longest road changes."""
    # Update road lengths in player_state
    for p in range(state.num_players):
        state.player_state[p, PS_LONGEST_ROAD] = state.road_lengths[p]

    if new_holder == prev_holder:
        return  # no change
    if new_holder == -1:
        return  # shouldn't happen (road can't be unbuilt)

    # New holder gains 2 VP
    state.player_state[new_holder, PS_HAS_ROAD] = 1
    state.player_state[new_holder, PS_VP] += 2
    state.player_state[new_holder, PS_ACTUAL_VP] += 2

    # Previous holder loses 2 VP
    if prev_holder >= 0:
        state.player_state[prev_holder, PS_HAS_ROAD] = 0
        state.player_state[prev_holder, PS_VP] -= 2
        state.player_state[prev_holder, PS_ACTUAL_VP] -= 2


def _maintain_largest_army(state, pidx, prev_army_holder, prev_army_size, knight_count):
    """Update largest army after knight play."""
    if knight_count < 3:
        return

    if prev_army_holder == -1:
        # First to reach 3 knights
        state.player_state[pidx, PS_HAS_ARMY] = 1
        state.player_state[pidx, PS_VP] += 2
        state.player_state[pidx, PS_ACTUAL_VP] += 2
        state.army_holder = pidx
        state.army_holder_size = knight_count
    elif prev_army_size < knight_count and prev_army_holder != pidx:
        # Overtake
        state.player_state[pidx, PS_HAS_ARMY] = 1
        state.player_state[pidx, PS_VP] += 2
        state.player_state[pidx, PS_ACTUAL_VP] += 2

        state.player_state[prev_army_holder, PS_HAS_ARMY] = 0
        state.player_state[prev_army_holder, PS_VP] -= 2
        state.player_state[prev_army_holder, PS_ACTUAL_VP] -= 2

        state.army_holder = pidx
        state.army_holder_size = knight_count
    elif prev_army_holder == pidx:
        state.army_holder_size = knight_count


def _has_buildable_road_edges(state, pidx):
    """Check if player has any buildable road edges (for road building dev card).

    Matches Catanatron's road_building_possibilities(state, color, False) which
    checks ROADS_AVAILABLE and buildable edges.
    """
    ps = state.player_state[pidx]
    if ps[PS_ROADS_AVAIL] <= 0:
        return False

    # Iterate all nodes in player's components
    for node in bitscan(state.reachable_bb[pidx]):
        for eidx in INCIDENT_EDGES[node]:
            # Check if edge is unowned
            owned = False
            for p in range(state.num_players):
                if state.has_edge(p, eidx):
                    owned = True
                    break
            if not owned:
                return True
    return False


def _reset_trading_state(state):
    state.is_resolving_trade = False
