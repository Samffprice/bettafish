"""Legal move generation for BitboardState.

bb_generate_actions(state, vps_to_win=10) returns the same list of actions
as Catanatron's generate_playable_actions(state).
"""

from catanatron.models.enums import (
    RESOURCES, WOOD, BRICK, SHEEP, WHEAT, ORE,
    Action, ActionType,
)
from catanatron.models.decks import (
    freqdeck_from_listdeck, freqdeck_contains, freqdeck_can_draw, freqdeck_count,
)
from robottler.bitboard.masks import (
    NUM_NODES, NUM_EDGES, NUM_TILES,
    NODE_BIT, EDGE_TO_INDEX, EDGE_LIST, EDGE_ENDPOINTS, INCIDENT_EDGES,
    RESOURCE_INDEX, PORT_TYPE_INDEX, popcount64, bitscan,
)
from robottler.bitboard.state import (
    PS_VP, PS_ROADS_AVAIL, PS_SETTLE_AVAIL, PS_CITY_AVAIL,
    PS_HAS_ROAD, PS_HAS_ARMY, PS_HAS_ROLLED, PS_HAS_PLAYED_DEV,
    PS_ACTUAL_VP, PS_LONGEST_ROAD,
    PS_KNIGHT_START, PS_MONO_START, PS_YOP_START, PS_RB_START,
    PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
    PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
    PS_RESOURCE_START, PS_RESOURCE_END,
    PROMPT_BUILD_INITIAL_SETTLEMENT, PROMPT_BUILD_INITIAL_ROAD,
    PROMPT_PLAY_TURN, PROMPT_DISCARD, PROMPT_MOVE_ROBBER,
    PROMPT_DECIDE_TRADE, PROMPT_DECIDE_ACCEPTEES,
)


def bb_generate_actions(state):
    """Generate all legal actions for the current player.

    Mirrors Catanatron's generate_playable_actions(state).
    Dispatches on state.current_prompt (matching Catanatron's ActionPrompt).
    """
    pidx = state.current_player_idx
    color = state.colors[pidx]
    ps = state.player_state[pidx]
    prompt = state.current_prompt

    if prompt == PROMPT_BUILD_INITIAL_SETTLEMENT:
        return _settlement_possibilities_initial(state, color)

    if prompt == PROMPT_BUILD_INITIAL_ROAD:
        return _initial_road_possibilities(state, pidx, color)

    if prompt == PROMPT_DISCARD:
        return [Action(color, ActionType.DISCARD, None)]

    if prompt == PROMPT_MOVE_ROBBER:
        return _robber_possibilities(state, pidx, color)

    if prompt == PROMPT_DECIDE_TRADE or prompt == PROMPT_DECIDE_ACCEPTEES:
        return _trade_response_possibilities(state, pidx, color)

    # PLAY_TURN prompt
    if state.is_road_building:
        return _road_building_free_possibilities(state, pidx, color)

    actions = []

    # Dev card plays (before and after rolling)
    if _can_play_dev(ps, PS_KNIGHT_HAND, PS_KNIGHT_START):
        actions.append(Action(color, ActionType.PLAY_KNIGHT_CARD, None))

    if _can_play_dev(ps, PS_YOP_HAND, PS_YOP_START):
        bank_freqdeck = [int(state.bank[i]) for i in range(5)]
        actions.extend(_year_of_plenty_possibilities(color, bank_freqdeck))

    if _can_play_dev(ps, PS_MONO_HAND, PS_MONO_START):
        actions.extend(_monopoly_possibilities(color))

    if _can_play_dev(ps, PS_RB_HAND, PS_RB_START):
        if _road_building_free_possibilities(state, pidx, color):
            actions.append(Action(color, ActionType.PLAY_ROAD_BUILDING, None))

    if not ps[PS_HAS_ROLLED]:
        actions.append(Action(color, ActionType.ROLL, None))
    else:
        actions.append(Action(color, ActionType.END_TURN, None))

        # Building actions
        actions.extend(_road_possibilities(state, pidx, color))
        actions.extend(_settlement_possibilities(state, pidx, color))
        actions.extend(_city_possibilities(state, pidx, color))

        # Buy dev card
        can_afford_dev = (ps[PS_SHEEP] >= 1 and ps[PS_WHEAT] >= 1 and ps[PS_ORE] >= 1)
        has_dev_cards = (25 - state.dev_deck_idx) > 0
        if can_afford_dev and has_dev_cards:
            actions.append(Action(color, ActionType.BUY_DEVELOPMENT_CARD, None))

        # Maritime trade
        actions.extend(_maritime_trade_possibilities(state, pidx, color))

    return actions


# --- Settlement ---

def _settlement_possibilities_initial(state, color):
    """All buildable nodes during initial placement."""
    actions = []
    for node in bitscan(state.board_buildable):
        actions.append(Action(color, ActionType.BUILD_SETTLEMENT, node))
    return sorted(actions, key=lambda a: a.value)


def _settlement_possibilities(state, pidx, color):
    """Settlement possibilities during normal play (must afford + connected)."""
    ps = state.player_state[pidx]
    if ps[PS_SETTLE_AVAIL] <= 0:
        return []
    if ps[PS_WOOD] < 1 or ps[PS_BRICK] < 1 or ps[PS_SHEEP] < 1 or ps[PS_WHEAT] < 1:
        return []

    # Buildable = board_buildable AND reachable by player's road network
    buildable = state.board_buildable & state.reachable_bb[pidx]
    actions = []
    for node in bitscan(buildable):
        actions.append(Action(color, ActionType.BUILD_SETTLEMENT, node))
    return sorted(actions, key=lambda a: a.value)


def _initial_road_possibilities(state, pidx, color):
    """Road possibilities during initial placement (connected to last settlement)."""
    # Find last settlement (highest-bit node in settlement_bb)
    last_node = -1
    for node in bitscan(state.settlement_bb[pidx]):
        last_node = node  # last one enumerated is the one we want? Not necessarily.

    # Actually, we need the most recently placed settlement.
    # In the bitboard we don't track order. Use the one that has fewer roads around it.
    # Better approach: find settlement nodes without roads.
    settle_nodes = list(bitscan(state.settlement_bb[pidx]))
    for node in settle_nodes:
        has_road = False
        for eidx in INCIDENT_EDGES[node]:
            if state.has_edge(pidx, eidx):
                has_road = True
                break
        if not has_road:
            last_node = node
            break

    if last_node < 0:
        return []

    # Find buildable edges incident to last settlement
    actions = []
    for eidx in INCIDENT_EDGES[last_node]:
        # Check edge is not already built by anyone
        owned = False
        for p in range(state.num_players):
            if state.has_edge(p, eidx):
                owned = True
                break
        if not owned:
            # Check edge endpoints are land nodes (they should be from INCIDENT_EDGES)
            edge = EDGE_LIST[eidx]
            actions.append(Action(color, ActionType.BUILD_ROAD, edge))

    return actions


def _road_possibilities(state, pidx, color):
    """Road possibilities during normal play (must afford + connected)."""
    ps = state.player_state[pidx]
    if ps[PS_ROADS_AVAIL] <= 0:
        return []
    if ps[PS_WOOD] < 1 or ps[PS_BRICK] < 1:
        return []

    return _buildable_edge_actions(state, pidx, color)


def _road_building_free_possibilities(state, pidx, color):
    """Road possibilities for Road Building dev card (free, must be connected)."""
    ps = state.player_state[pidx]
    if ps[PS_ROADS_AVAIL] <= 0:
        return []
    return _buildable_edge_actions(state, pidx, color)


def _buildable_edge_actions(state, pidx, color):
    """All unowned edges connected to player's road network."""
    actions = []
    seen = set()

    for node in bitscan(state.reachable_bb[pidx]):
        for eidx in INCIDENT_EDGES[node]:
            if eidx in seen:
                continue
            seen.add(eidx)

            # Check edge is unowned
            owned = False
            for p in range(state.num_players):
                if state.has_edge(p, eidx):
                    owned = True
                    break
            if not owned:
                edge = EDGE_LIST[eidx]
                actions.append(Action(color, ActionType.BUILD_ROAD, edge))

    return actions


# --- City ---

def _city_possibilities(state, pidx, color):
    """City upgrade possibilities."""
    ps = state.player_state[pidx]
    if ps[PS_CITY_AVAIL] <= 0:
        return []
    if ps[PS_WHEAT] < 2 or ps[PS_ORE] < 3:
        return []

    actions = []
    for node in bitscan(state.settlement_bb[pidx]):
        actions.append(Action(color, ActionType.BUILD_CITY, node))
    return actions


# --- Robber ---

def _robber_possibilities(state, pidx, color):
    """Generate robber placement + steal target actions."""
    actions = []
    friendly = state.friendly_robber

    for tid in range(NUM_TILES):
        if tid == state.robber_tile:
            continue  # must move robber

        coord = state.tile_id_to_coord[tid]
        tile_mask = state.catan_map.land_tiles[coord].nodes.values()

        # Friendly robber: skip tiles where ALL enemy buildings belong to
        # players with < 3 visible VP
        if friendly:
            enemy_owners = set()
            for node_id in tile_mask:
                if node_id >= NUM_NODES:
                    continue
                owner, _ = state.building_owner(node_id)
                if owner is not None and owner != pidx:
                    enemy_owners.add(owner)
            if enemy_owners and all(
                int(state.player_state[opp, PS_VP]) < 3
                for opp in enemy_owners
            ):
                continue

        # Find stealable players at this tile
        stealable = set()
        for node_id in tile_mask:
            if node_id >= NUM_NODES:
                continue
            owner, _ = state.building_owner(node_id)
            if owner is not None and owner != pidx:
                # Check they have resources
                total = int(state.player_state[owner, PS_RESOURCE_START:PS_RESOURCE_END].sum())
                if total >= 1:
                    stealable.add(state.colors[owner])

        if not stealable:
            actions.append(Action(color, ActionType.MOVE_ROBBER, (coord, None)))
        else:
            for enemy_color in stealable:
                actions.append(Action(color, ActionType.MOVE_ROBBER, (coord, enemy_color)))

    return actions


# --- Dev Cards ---

def _can_play_dev(ps, hand_idx, start_idx):
    """Check if player can play a dev card (has it, owned at start, not played one this turn)."""
    return (not ps[PS_HAS_PLAYED_DEV] and ps[hand_idx] >= 1 and ps[start_idx])


def _monopoly_possibilities(color):
    return [Action(color, ActionType.PLAY_MONOPOLY, card) for card in RESOURCES]


def _year_of_plenty_possibilities(color, bank_freqdeck):
    """Generate YoP actions, matching Catanatron's logic."""
    options = set()
    for i, first_card in enumerate(RESOURCES):
        for j in range(i, len(RESOURCES)):
            second_card = RESOURCES[j]
            to_draw = freqdeck_from_listdeck([first_card, second_card])
            if freqdeck_contains(bank_freqdeck, to_draw):
                options.add((first_card, second_card))
            else:
                if freqdeck_can_draw(bank_freqdeck, 1, first_card):
                    options.add((first_card,))
                if freqdeck_can_draw(bank_freqdeck, 1, second_card):
                    options.add((second_card,))

    return [Action(color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(cards)) for cards in options]


# --- Maritime Trade ---

def _maritime_trade_possibilities(state, pidx, color):
    """Generate maritime trade actions."""
    ps = state.player_state[pidx]

    # Determine trade rates
    rates = [4, 4, 4, 4, 4]  # default 4:1
    if state.has_port(pidx, 5):  # 3:1 port
        rates = [3, 3, 3, 3, 3]
    for ri in range(5):
        if state.has_port(pidx, ri):
            rates[ri] = 2

    trade_offers = set()
    for ri, resource in enumerate(RESOURCES):
        amount = int(ps[PS_RESOURCE_START + ri])
        if amount >= rates[ri]:
            resource_out = [resource] * rates[ri] + [None] * (4 - rates[ri])
            for ji, j_resource in enumerate(RESOURCES):
                if ri != ji and state.bank[ji] > 0:
                    trade_offer = tuple(resource_out + [j_resource])
                    trade_offers.add(trade_offer)

    return [Action(color, ActionType.MARITIME_TRADE, t) for t in trade_offers]


# --- Trade response ---

def _trade_response_possibilities(state, pidx, color):
    """Simplified trade response (search doesn't do domestic trades)."""
    # This shouldn't be called during search, but handle gracefully
    return [Action(color, ActionType.REJECT_TRADE, None)]


# --- Helpers ---

def _count_roads(state, pidx):
    """Count total roads for a player."""
    count = 0
    for eidx in range(NUM_EDGES):
        if state.has_edge(pidx, eidx):
            count += 1
    return count
