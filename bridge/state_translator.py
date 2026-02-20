"""Shadow state manager for the AICatan bridge.

Maintains a lightweight shadow state from colonist.io messages and reconstructs
a Catanatron-compatible Game object at each decision point.

Key design decisions:
- Uses Game(players=[], initialize=False) to bypass player reshuffling
- Buildings are replayed chronologically to maintain connected_components
- Pending roads are tracked optimistically for road_building card
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import STATIC_GRAPH, Board
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    ActionPrompt,
    SETTLEMENT,
    CITY,
    ROAD,
)
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, Player
from catanatron.state import State, PLAYER_INITIAL_STATE
from catanatron.models.decks import starting_resource_bank, starting_devcard_bank

from bridge.coordinate_map import CoordinateMapper, build_coordinate_mapper
from bridge.config import (
    COLONIST_RESOURCE_TO_CATAN,
    COLONIST_ACTION_STATE_SETUP_SETTLEMENT,
    COLONIST_ACTION_STATE_SETUP_ROAD,
    COLONIST_ACTION_STATE_MOVE_ROBBER,
    COLONIST_ACTION_STATE_ROB_VICTIM,
    COLONIST_ACTION_STATE_DISCARD,
    COLONIST_TURN_STATE_ROLL,
    COLONIST_TURN_STATE_PLAY,
)

logger = logging.getLogger(__name__)


# Catanatron Color values - colonist.io uses integer indices
CATANATRON_COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


class TurnPhase(Enum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    START_TURN = 2      # need to roll
    PLAYER_TURN = 3     # rolled, can build
    OPPONENT_TURN = 4


@dataclass
class BuildEvent:
    """Records a single build event in chronological order."""
    building_type: str    # SETTLEMENT, CITY, ROAD
    color_index: int      # colonist.io color index
    col_coord: Any        # colonist vertex (x,y,z) or edge (x,y,z)
    is_setup: bool        # True if this was an initial build


@dataclass
class ShadowState:
    """Lightweight state tracked from colonist.io messages."""
    # Player info
    my_color_index: int = -1             # colonist.io color index (1-based)
    my_catan_color: Optional[Color] = None
    colonist_to_catan_color: Dict = field(default_factory=dict)  # col_idx -> Color
    catanatron_to_colonist_color: Dict = field(default_factory=dict)  # Color -> col_idx
    player_order: List[int] = field(default_factory=list)  # colonist color indices in seating order

    # Board state
    mapper: Optional[CoordinateMapper] = None
    robber_coord: Optional[Tuple] = None  # Catanatron cube coordinate

    # Resource hand (our resources only - opponents are hidden)
    my_resources: Dict = field(default_factory=lambda: {r: 0 for r in RESOURCES})

    # Dev cards in our hand
    my_dev_cards: Dict = field(default_factory=lambda: {c: 0 for c in DEVELOPMENT_CARDS})
    played_dev_cards: Dict = field(default_factory=lambda: {c: 0 for c in DEVELOPMENT_CARDS})
    # Count of hidden (id=10) dev cards in hand — colonist.io hides card types
    unknown_dev_card_count: int = 0
    # Cards in hand with known types (accumulated from developmentCardsBoughtThisTurn)
    known_dev_card_types: list = field(default_factory=list)
    # How many entries in developmentCardsBoughtThisTurn we've already processed
    _bought_this_turn_processed: int = 0

    # Building history for chronological reconstruction
    build_history: List[BuildEvent] = field(default_factory=list)

    # Pending roads (optimistic tracking for road building card)
    pending_roads: List[Any] = field(default_factory=list)  # list of catanatron edge tuples

    # Index maps (populated from type 4 initial state)
    vertex_index_to_coord: Dict = field(default_factory=dict)  # int index -> (x,y,z)
    edge_index_to_coord: Dict = field(default_factory=dict)    # int index -> (x,y,z)

    # Per-player played dev cards — {col_idx: [card_id, ...]} for ALL players (for largest army)
    all_players_dev_cards_used: Dict[int, List[int]] = field(default_factory=dict)
    # Dev cards bought THIS turn — can't play same-turn purchases
    dev_cards_bought_this_turn: int = 0
    dev_card_types_bought_this_turn: List[str] = field(default_factory=list)
    # Whether we already played a dev card this turn
    has_used_dev_card_this_turn: bool = False

    # Road building card state
    is_road_building: bool = False
    free_roads_available: int = 0

    # Opponent resources — {col_idx: {"WOOD": n, ...}} for all non-us players
    opponent_resources: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # Turn state
    current_turn_state: int = -1     # colonist turnState: 1=roll, 2=play, 3=game_over
    current_action_state: int = -1   # colonist actionState: 0=main, 1=setup_settlement, etc.
    current_turn_player_color: int = -1
    is_setup_phase: bool = True      # True until first turnState appears in a diff
    turn_phase: TurnPhase = TurnPhase.SETUP_SETTLEMENT

    # Game setup tracking
    setup_settlements_placed: int = 0


class StateTranslator:
    """Translates colonist.io events into Catanatron game state.

    Maintains a ShadowState and provides reconstruct_game() to create
    a Catanatron Game object for bot decision-making.
    """

    def __init__(self):
        self.state = ShadowState()

    def set_my_color(self, color_index: int) -> None:
        """Record our player color from colonist.io."""
        self.state.my_color_index = color_index
        logger.info(f"My color index: {color_index}")

    def set_player_order(self, play_order: List[int]) -> None:
        """Set all players from the type 4 playOrder array.

        Discovers all players upfront with correct seating indices.
        Our color must be set first via set_my_color().
        """
        self.state.player_order = list(play_order)
        for i, col_idx in enumerate(play_order):
            self._discover_player(col_idx, seating_index=i)
        logger.info(
            f"Player order set: {play_order}, "
            f"mapping: {self.state.colonist_to_catan_color}"
        )

    def setup_board(self, tiles: List[Dict], vertices: List[Dict], edges: List[Dict]) -> None:
        """Initialize coordinate mapping from colonist.io board setup message."""
        try:
            self.state.mapper = build_coordinate_mapper(tiles, vertices, edges)
            # Find desert tile for initial robber placement
            catan_map = self.state.mapper.catan_map
            for coord, tile in catan_map.land_tiles.items():
                if tile.resource is None:  # desert
                    self.state.robber_coord = coord
                    break
            logger.info(f"Board setup complete. Robber at {self.state.robber_coord}")
        except Exception as e:
            logger.error(f"Failed to setup board: {e}", exc_info=True)

    def update_turn_state(
        self,
        current_turn_state: Optional[int] = None,
        current_action_state: Optional[int] = None,
        current_turn_player_color: Optional[int] = None,
    ) -> None:
        """Update turn/action state from colonist.io diff (partial update).

        Only updates fields that are not None, preserving previously set values.
        """
        if current_turn_state is not None:
            self.state.current_turn_state = current_turn_state
            # turnState 0 = setup phase; >= 1 means game has started (1=ROLL, 2=PLAY, 3=GAME_OVER)
            if current_turn_state >= 1:
                self.state.is_setup_phase = False
        if current_action_state is not None:
            self.state.current_action_state = current_action_state
        if current_turn_player_color is not None:
            self.state.current_turn_player_color = current_turn_player_color

    def is_our_turn(self) -> bool:
        """Return True if it's currently our turn."""
        return self.state.current_turn_player_color == self.state.my_color_index

    def update_resources_from_distribution(self, distributions: List[Dict]) -> None:
        """Update our resource hand from a resource distribution event."""
        for dist in distributions:
            owner = dist.get("owner", -1)
            card = dist.get("card", -1)
            if owner == self.state.my_color_index and card in COLONIST_RESOURCE_TO_CATAN:
                resource = COLONIST_RESOURCE_TO_CATAN[card]
                self.state.my_resources[resource] = self.state.my_resources.get(resource, 0) + 1

    def update_resources_from_trade(
        self,
        giving_player: int,
        receiving_player: int,
        giving_cards: List[int],
        receiving_cards: List[int],
    ) -> None:
        """Update our resources when a trade is executed."""
        my_color = self.state.my_color_index
        if giving_player == my_color:
            for card in giving_cards:
                if card in COLONIST_RESOURCE_TO_CATAN:
                    resource = COLONIST_RESOURCE_TO_CATAN[card]
                    self.state.my_resources[resource] = max(
                        0, self.state.my_resources.get(resource, 0) - 1
                    )
            for card in receiving_cards:
                if card in COLONIST_RESOURCE_TO_CATAN:
                    resource = COLONIST_RESOURCE_TO_CATAN[card]
                    self.state.my_resources[resource] = self.state.my_resources.get(resource, 0) + 1
        elif receiving_player == my_color:
            for card in receiving_cards:
                if card in COLONIST_RESOURCE_TO_CATAN:
                    resource = COLONIST_RESOURCE_TO_CATAN[card]
                    self.state.my_resources[resource] = max(
                        0, self.state.my_resources.get(resource, 0) - 1
                    )
            for card in giving_cards:
                if card in COLONIST_RESOURCE_TO_CATAN:
                    resource = COLONIST_RESOURCE_TO_CATAN[card]
                    self.state.my_resources[resource] = self.state.my_resources.get(resource, 0) + 1

    def update_vertex(
        self,
        x: int, y: int, z: int,
        owner: int,
        building_type: int,
        harbor_type: int = 0,
    ) -> None:
        """Record a vertex (settlement/city) update from colonist.io."""
        col_coord = (x, y, z)

        # Skip unowned entries (initial board state, not actual builds)
        if owner < 0:
            return

        # Discover player if new
        self._discover_player(owner)

        # Deduplicate: check if we already have a build at this coord
        btype = CITY if building_type == 2 else SETTLEMENT
        for existing in self.state.build_history:
            if existing.col_coord == col_coord and existing.color_index == owner:
                if existing.building_type == btype:
                    # Exact duplicate (same coord, same owner, same type) — skip
                    return
                # Different type at same coord (settlement→city) is a valid upgrade — allow

        # Record build event
        is_setup = self.state.is_setup_phase
        event = BuildEvent(
            building_type=btype,
            color_index=owner,
            col_coord=col_coord,
            is_setup=is_setup,
        )
        self.state.build_history.append(event)

        if owner == self.state.my_color_index and building_type == 1:
            self.state.setup_settlements_placed += 1
            # Deduct resources for settlement (not during setup)
            if not is_setup:
                for r in ["WOOD", "BRICK", "SHEEP", "WHEAT"]:
                    self.state.my_resources[r] = max(0, self.state.my_resources.get(r, 0) - 1)
        elif owner == self.state.my_color_index and building_type == 2:
            # City: costs 2 wheat, 3 ore
            if not is_setup:
                self.state.my_resources["WHEAT"] = max(0, self.state.my_resources.get("WHEAT", 0) - 2)
                self.state.my_resources["ORE"] = max(0, self.state.my_resources.get("ORE", 0) - 3)

    def update_edge(self, x: int, y: int, z: int, owner: int) -> None:
        """Record an edge (road) update from colonist.io."""
        # Skip unowned entries (initial board state, not actual builds)
        if owner < 0:
            return

        col_coord = (x, y, z)

        # Deduplicate: check if we already have a road at this coord
        for existing in self.state.build_history:
            if existing.col_coord == col_coord and existing.building_type == ROAD and existing.color_index == owner:
                # Exact duplicate road — skip
                return

        # Discover player if new
        self._discover_player(owner)

        # Clear any pending road that matches this edge (optimistic tracking fulfilled)
        if self.state.mapper is not None:
            catan_edge = self.state.mapper.colonist_edge_to_catan.get(col_coord)
            if catan_edge and catan_edge in self.state.pending_roads:
                self.state.pending_roads.remove(catan_edge)

        is_setup = self.state.is_setup_phase
        event = BuildEvent(
            building_type=ROAD,
            color_index=owner,
            col_coord=col_coord,
            is_setup=is_setup,
        )
        self.state.build_history.append(event)

        if owner == self.state.my_color_index and not is_setup:
            if self.state.is_road_building and self.state.free_roads_available > 0:
                # Free road from road building card — no resource cost
                self.state.free_roads_available -= 1
                if self.state.free_roads_available <= 0:
                    self.state.is_road_building = False
                    self.state.free_roads_available = 0
            else:
                # Normal road: costs 1 wood, 1 brick
                self.state.my_resources["WOOD"] = max(0, self.state.my_resources.get("WOOD", 0) - 1)
                self.state.my_resources["BRICK"] = max(0, self.state.my_resources.get("BRICK", 0) - 1)

    def update_robber(self, new_tile_x: int, new_tile_y: int) -> None:
        """Update robber location from colonist.io (x, y) coordinates."""
        if self.state.mapper is None:
            return
        xy = (new_tile_x, new_tile_y)
        catan_tile = self.state.mapper.colonist_tile_to_catan.get(xy)
        if catan_tile is not None:
            for cube_coord, tile in self.state.mapper.catan_map.land_tiles.items():
                if tile.id == catan_tile.id:
                    self.state.robber_coord = cube_coord
                    logger.info(f"Robber moved to cube {cube_coord}")
                    break

    def update_robber_by_tile_index(self, tile_index: int) -> None:
        """Update robber location from a tile index (used in type 91 diffs)."""
        if self.state.mapper is None:
            return
        xy = self.state.mapper.colonist_tile_index_to_xy.get(tile_index)
        if xy is not None:
            self.update_robber(xy[0], xy[1])
        else:
            logger.warning(f"Unknown tile index for robber: {tile_index}")

    def add_dev_card(self, card_type: str) -> None:
        """Record that we bought/received a dev card."""
        if card_type in self.state.my_dev_cards:
            self.state.my_dev_cards[card_type] = self.state.my_dev_cards[card_type] + 1

    def play_dev_card(self, card_type: str) -> None:
        """Record that we played a dev card."""
        if card_type in self.state.my_dev_cards:
            self.state.my_dev_cards[card_type] = max(
                0, self.state.my_dev_cards[card_type] - 1
            )
            self.state.played_dev_cards[card_type] = (
                self.state.played_dev_cards.get(card_type, 0) + 1
            )

    def add_pending_road(self, catan_edge: Tuple) -> None:
        """Optimistically record a road being placed (for road_building card)."""
        self.state.pending_roads.append(catan_edge)

    def deduct_buy_dev_card(self) -> None:
        """Deduct resources for buying a dev card."""
        self.state.my_resources["ORE"] = max(0, self.state.my_resources.get("ORE", 0) - 1)
        self.state.my_resources["WHEAT"] = max(0, self.state.my_resources.get("WHEAT", 0) - 1)
        self.state.my_resources["SHEEP"] = max(0, self.state.my_resources.get("SHEEP", 0) - 1)

    def validate_resources(self, expected: Dict) -> None:
        """Log warning if our resource count doesn't match expected."""
        for resource, count in expected.items():
            actual = self.state.my_resources.get(resource, 0)
            if actual != count:
                logger.warning(
                    f"Resource desync for {resource}: expected {count}, have {actual}"
                )

    def _discover_player(self, color_index: int, seating_index: Optional[int] = None) -> None:
        """Assign a Catanatron Color to a colonist.io color index if not already done.

        Players are discovered in the order they are first seen. The player order
        determines which Catanatron Color they get (RED, BLUE, ORANGE, WHITE).
        """
        if color_index in self.state.colonist_to_catan_color:
            return
        if color_index < 0:
            return

        if seating_index is not None:
            idx = seating_index
        else:
            idx = len(self.state.player_order)

        if idx >= len(CATANATRON_COLORS):
            logger.warning(f"Too many players discovered (>4): color_index={color_index}")
            return

        catan_color = CATANATRON_COLORS[idx]
        self.state.colonist_to_catan_color[color_index] = catan_color
        self.state.catanatron_to_colonist_color[catan_color] = color_index

        if color_index not in self.state.player_order:
            self.state.player_order.append(color_index)

        if color_index == self.state.my_color_index:
            self.state.my_catan_color = catan_color

        logger.debug(f"Discovered player: colonist={color_index} -> catanatron={catan_color}")

    def reconstruct_game(self) -> Optional[Game]:
        """Reconstruct a Catanatron Game object from shadow state.

        Uses Game(players=[], initialize=False) to bypass State.__init__()'s
        random.sample() player randomization. All State fields are set manually
        following the pattern from Game.copy() and State.copy().

        Buildings are replayed chronologically via board.build_settlement() and
        board.build_road() to correctly maintain connected_components, board_buildable_ids,
        and road_lengths.

        Returns:
            Reconstructed Game object, or None if mapper is not set.
        """
        if self.state.mapper is None:
            logger.warning("Cannot reconstruct game: mapper not initialized")
            return None

        mapper = self.state.mapper
        catan_map = mapper.catan_map

        # Determine colors in seating order
        discovered_colors = []
        for col_idx in self.state.player_order:
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color:
                discovered_colors.append(catan_color)

        # If we haven't discovered all players yet, add our color first
        if not discovered_colors and self.state.my_catan_color:
            discovered_colors = [self.state.my_catan_color]
        elif self.state.my_catan_color and self.state.my_catan_color not in discovered_colors:
            discovered_colors.insert(0, self.state.my_catan_color)

        # Ensure we have at least 4 colors for a complete game
        for c in CATANATRON_COLORS:
            if c not in discovered_colors:
                discovered_colors.append(c)
        colors = tuple(discovered_colors)

        # Create Player objects
        players = [Player(color) for color in colors]

        # Create Game without initialization (bypass random.sample)
        game = Game(players=[], initialize=False)
        game.seed = 0
        game.id = "bridge-reconstructed"
        game.vps_to_win = 15 if len(colors) == 2 else 10

        # Create State without initialization (bypass random.sample)
        state = State([], None, initialize=False)
        state.players = players
        state.colors = colors
        state.color_to_index = {color: i for i, color in enumerate(colors)}
        state.discard_limit = 7
        state.friendly_robber = len(colors) == 2  # 1v1 always has friendly robber

        # Initialize player_state
        state.player_state = {}
        for index in range(len(colors)):
            for key, value in PLAYER_INITIAL_STATE.items():
                state.player_state[f"P{index}_{key}"] = value

        # Initialize resource and dev card banks
        state.resource_freqdeck = starting_resource_bank()
        state.development_listdeck = starting_devcard_bank()

        # Initialize buildings_by_color
        state.buildings_by_color = {color: defaultdict(list) for color in colors}

        state.action_records = []
        state.num_turns = 0

        # Set turn state
        my_color = self.state.my_catan_color or Color.RED
        my_index = state.color_to_index.get(my_color, 0)

        # Set prompts based on current phase
        action_s = self.state.current_action_state

        if self.state.is_setup_phase:
            state.is_initial_build_phase = True
            state.is_discarding = False
            state.is_moving_knight = False
            state.is_road_building = False
            state.free_roads_available = 0
            if action_s == COLONIST_ACTION_STATE_SETUP_SETTLEMENT:
                state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
            elif action_s == COLONIST_ACTION_STATE_SETUP_ROAD:
                state.current_prompt = ActionPrompt.BUILD_INITIAL_ROAD
            else:
                state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        else:
            state.is_initial_build_phase = False
            state.is_discarding = (action_s == COLONIST_ACTION_STATE_DISCARD)
            state.is_moving_knight = (action_s == COLONIST_ACTION_STATE_MOVE_ROBBER)
            state.is_road_building = self.state.is_road_building
            state.free_roads_available = self.state.free_roads_available

            # Set current_prompt based on colonist.io action state
            if action_s == COLONIST_ACTION_STATE_MOVE_ROBBER:
                state.current_prompt = ActionPrompt.MOVE_ROBBER
            elif action_s == COLONIST_ACTION_STATE_DISCARD:
                state.current_prompt = ActionPrompt.DISCARD
            else:
                state.current_prompt = ActionPrompt.PLAY_TURN

            # Mark dice as rolled if colonist.io says we're in the PLAY phase
            turn_s = self.state.current_turn_state
            if turn_s == COLONIST_TURN_STATE_PLAY:
                state.player_state[f"P{my_index}_HAS_ROLLED"] = True

        state.is_resolving_trade = False
        state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        state.acceptees = tuple(False for _ in colors)
        state.current_player_index = my_index
        state.current_turn_index = my_index

        # Create Board with CatanMap (don't initialize - we'll set it up manually)
        board = Board(catan_map=catan_map, initialize=True)

        # Set robber location
        if self.state.robber_coord is not None:
            board.robber_coordinate = self.state.robber_coord

        # Replay builds chronologically
        for event in self.state.build_history:
            col_idx = event.color_index
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color is None:
                logger.warning(f"Unknown color_index {col_idx} in build_history")
                continue

            if event.building_type == SETTLEMENT:
                node_id = mapper.colonist_vertex_to_catan.get(event.col_coord)
                if node_id is None:
                    logger.warning(f"Unknown vertex coord {event.col_coord} for settlement")
                    continue

                # Skip if already recorded as a settlement or city at this node
                bbc = state.buildings_by_color[catan_color]
                if node_id in bbc[SETTLEMENT] or node_id in bbc[CITY]:
                    continue

                try:
                    board.build_settlement(catan_color, node_id, initial_build_phase=event.is_setup)
                except ValueError as e:
                    # Force-set board state so all buildings are registered regardless
                    # of replay order (on reconnect, order may not match original game)
                    logger.debug(f"Settlement build failed during replay ({e}), force-setting")
                    board.buildings[node_id] = (catan_color, SETTLEMENT)
                    board.board_buildable_ids.discard(node_id)
                    for n in STATIC_GRAPH.neighbors(node_id):
                        board.board_buildable_ids.discard(n)
                    board.connected_components[catan_color].append({node_id})
                    board.buildable_edges_cache = {}

                bbc[SETTLEMENT].append(node_id)
                idx = state.color_to_index[catan_color]
                state.player_state[f"P{idx}_VICTORY_POINTS"] += 1
                state.player_state[f"P{idx}_ACTUAL_VICTORY_POINTS"] += 1
                state.player_state[f"P{idx}_SETTLEMENTS_AVAILABLE"] = max(
                    0, state.player_state[f"P{idx}_SETTLEMENTS_AVAILABLE"] - 1
                )

            elif event.building_type == CITY:
                node_id = mapper.colonist_vertex_to_catan.get(event.col_coord)
                if node_id is None:
                    logger.warning(f"Unknown vertex coord {event.col_coord} for city")
                    continue

                # Always update buildings_by_color even if board.build_city fails,
                # otherwise generate_playable_actions sees a stale settlement and
                # the bot tries to upgrade a city that's already a city.
                bbc = state.buildings_by_color[catan_color]
                idx = state.color_to_index[catan_color]
                already_city = node_id in bbc[CITY]

                if already_city:
                    # Already recorded as a city — skip entirely (duplicate event)
                    continue

                try:
                    board.build_city(catan_color, node_id)
                except ValueError as e:
                    # Force-set the board building to CITY so board state stays consistent
                    logger.warning(f"City build failed during replay ({e}), force-setting board state")
                    board.buildings[node_id] = (catan_color, CITY)

                if node_id in bbc[SETTLEMENT]:
                    bbc[SETTLEMENT].remove(node_id)
                bbc[CITY].append(node_id)
                # City = +1 VP (settlement already counted, city adds +1 more)
                state.player_state[f"P{idx}_VICTORY_POINTS"] += 1
                state.player_state[f"P{idx}_ACTUAL_VICTORY_POINTS"] += 1
                state.player_state[f"P{idx}_CITIES_AVAILABLE"] = max(
                    0, state.player_state[f"P{idx}_CITIES_AVAILABLE"] - 1
                )
                state.player_state[f"P{idx}_SETTLEMENTS_AVAILABLE"] += 1  # settlement returned

            elif event.building_type == ROAD:
                catan_edge = mapper.colonist_edge_to_catan.get(event.col_coord)
                if catan_edge is None:
                    logger.debug(f"Unknown edge coord {event.col_coord} for road")
                    continue

                # Skip if already recorded
                bbc = state.buildings_by_color[catan_color]
                if catan_edge in bbc[ROAD]:
                    continue

                try:
                    board.build_road(catan_color, catan_edge)
                except ValueError as e:
                    # Force-set road on board (replay order may not match original)
                    logger.debug(f"Road build failed during replay ({e}), force-setting")
                    board.roads[catan_edge] = catan_color
                    board.roads[(catan_edge[1], catan_edge[0])] = catan_color
                    # Add nodes to connected component
                    a, b = catan_edge
                    merged = False
                    for comp in board.connected_components[catan_color]:
                        if a in comp or b in comp:
                            comp.add(a)
                            comp.add(b)
                            merged = True
                            break
                    if not merged:
                        board.connected_components[catan_color].append({a, b})
                    board.buildable_edges_cache = {}

                bbc[ROAD].append(catan_edge)
                idx = state.color_to_index[catan_color]
                state.player_state[f"P{idx}_ROADS_AVAILABLE"] = max(
                    0, state.player_state[f"P{idx}_ROADS_AVAILABLE"] - 1
                )

        # Add pending roads (optimistic tracking for road_building card)
        for catan_edge in self.state.pending_roads:
            try:
                board.build_road(my_color, catan_edge)
            except ValueError:
                pass  # May already be built or invalid

        # Compute longest road ONCE after all roads are replayed
        for col_idx in self.state.player_order:
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color is None:
                continue
            p_idx = state.color_to_index[catan_color]
            state.player_state[f"P{p_idx}_LONGEST_ROAD_LENGTH"] = board.road_lengths.get(catan_color, 0)

        if board.road_color is not None and board.road_length >= 5:
            p_idx = state.color_to_index[board.road_color]
            state.player_state[f"P{p_idx}_HAS_ROAD"] = True
            state.player_state[f"P{p_idx}_VICTORY_POINTS"] += 2
            state.player_state[f"P{p_idx}_ACTUAL_VICTORY_POINTS"] += 2

        # Set our resource hand
        my_key = f"P{my_index}"
        for resource in RESOURCES:
            count = self.state.my_resources.get(resource, 0)
            state.player_state[f"{my_key}_{resource}_IN_HAND"] = count

        # Set opponent resource hands (for robber steal checks)
        for col_idx, opp_resources in self.state.opponent_resources.items():
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color is None:
                continue
            p_idx = state.color_to_index[catan_color]
            for resource in RESOURCES:
                count = opp_resources.get(resource, 0)
                state.player_state[f"P{p_idx}_{resource}_IN_HAND"] = count

        # Set our dev cards
        # colonist.io reveals actual card types in developmentCards.cards for
        # our own player (e.g. id=11 for KNIGHT, 12 for VP). These go into
        # my_dev_cards with real counts. Any remaining hidden cards (id=10)
        # are in unknown_dev_card_count and treated as Knights.
        for dev_card in DEVELOPMENT_CARDS:
            count = self.state.my_dev_cards.get(dev_card, 0)
            played = self.state.played_dev_cards.get(dev_card, 0)
            if dev_card == "KNIGHT":
                count += self.state.unknown_dev_card_count
            state.player_state[f"{my_key}_{dev_card}_IN_HAND"] = count
            state.player_state[f"{my_key}_PLAYED_{dev_card}"] = played

        # Set _OWNED_AT_START for playable dev cards (not VICTORY_POINT).
        # Cards bought THIS turn can't be played until next turn.
        # Count how many of each type were bought this turn.
        bought_type_counts = {}
        for t in self.state.dev_card_types_bought_this_turn:
            bought_type_counts[t] = bought_type_counts.get(t, 0) + 1
        # Hidden cards bought this turn (type unknown)
        hidden_bought = max(0, self.state.dev_cards_bought_this_turn - len(self.state.dev_card_types_bought_this_turn))
        playable_unknown = max(0, self.state.unknown_dev_card_count - hidden_bought)

        PLAYABLE_DEV_CARDS = ["KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING"]
        for dev_card in PLAYABLE_DEV_CARDS:
            count = self.state.my_dev_cards.get(dev_card, 0)
            count -= bought_type_counts.get(dev_card, 0)  # exclude same-turn purchases
            if dev_card == "KNIGHT":
                count += playable_unknown
            state.player_state[f"{my_key}_{dev_card}_OWNED_AT_START"] = (count > 0)

        # Set HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN
        state.player_state[f"{my_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = (
            self.state.has_used_dev_card_this_turn
        )

        # Set PLAYED_KNIGHT for all players (from all_players_dev_cards_used)
        for col_idx in self.state.player_order:
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color is None:
                continue
            p_idx = state.color_to_index[catan_color]
            used = self.state.all_players_dev_cards_used.get(col_idx, [])
            knight_count = sum(1 for c in used if c == 11)
            state.player_state[f"P{p_idx}_PLAYED_KNIGHT"] = knight_count

        # Determine largest army holder
        best_color, best_count = None, 0
        for col_idx in self.state.player_order:
            catan_color = self.state.colonist_to_catan_color.get(col_idx)
            if catan_color is None:
                continue
            p_idx = state.color_to_index[catan_color]
            kc = state.player_state[f"P{p_idx}_PLAYED_KNIGHT"]
            if kc >= 3 and kc > best_count:
                best_count = kc
                best_color = catan_color

        if best_color is not None:
            p_idx = state.color_to_index[best_color]
            state.player_state[f"P{p_idx}_HAS_ARMY"] = True
            state.player_state[f"P{p_idx}_VICTORY_POINTS"] += 2
            state.player_state[f"P{p_idx}_ACTUAL_VICTORY_POINTS"] += 2

        state.board = board
        game.state = state

        # Dev card diagnostics
        total_dev = sum(self.state.my_dev_cards.values()) + self.state.unknown_dev_card_count
        if total_dev > 0 or any(self.state.played_dev_cards.values()):
            hand_str = {k: v for k, v in self.state.my_dev_cards.items() if v > 0}
            played_str = {k: v for k, v in self.state.played_dev_cards.items() if v > 0}
            has_played = state.player_state.get(f"{my_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", False)
            logger.info(
                f"reconstruct dev cards: hand={hand_str}, "
                f"hidden={self.state.unknown_dev_card_count}, "
                f"played={played_str}, "
                f"used_this_turn={has_played}, "
                f"prompt={state.current_prompt}"
            )

        try:
            game.playable_actions = generate_playable_actions(state)
        except Exception as e:
            logger.error(f"Failed to generate playable actions: {e}", exc_info=True)
            game.playable_actions = []

        return game
