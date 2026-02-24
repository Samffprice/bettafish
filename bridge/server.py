"""WebSocket bridge server.

Receives colonist.io game messages from the Tampermonkey userscript,
routes them through the state translator, queries the bot, and sends
translated actions back.

Includes:
- Async message consumer/producer pattern
- Decision timeout (5s) with END_TURN fallback
- 30-second watchdog that auto-sends END_TURN if stalled
- Proper resource cleanup in finally block
"""
import argparse
import asyncio
import inspect
import json
import logging
import random
import time
from typing import Dict, List, Optional

import websockets

from bridge.protocol import (
    AvailableActionsMsg,
    DiscardPromptMsg,
    EndTurnMsg,
    GameInitMsg,
    GameOverMsg,
    GameStateDiffMsg,
    HeartbeatMsg,
    ResourceDistributionMsg,
    RobVictimPromptMsg,
    TradeExecutionMsg,
    UnknownMsg,
    parse_message,
)
from bridge.state_translator import StateTranslator, TurnPhase
from bridge.action_translator import ActionTranslator
from bridge.bot_interface import BotInterface
from bridge.game_logger import GameLogger
from bridge.config import (
    WS_HOST,
    WS_PORT,
    COLONIST_RESOURCE_TO_CATAN,
    COLONIST_TILE_TO_RESOURCE,
    COLONIST_VALUE_TO_DEVCARD,
    DECISION_TIMEOUT_SECONDS,
    WATCHDOG_TIMEOUT_SECONDS,
    WS_SEND_TIMEOUT_SECONDS,
    WS_RECV_TIMEOUT_SECONDS,
    COLONIST_ACTION_STATE_MAIN,
    COLONIST_ACTION_STATE_SETUP_SETTLEMENT,
    COLONIST_ACTION_STATE_SETUP_ROAD,
    COLONIST_ACTION_STATE_MOVE_ROBBER,
    COLONIST_ACTION_STATE_ROB_VICTIM,
    COLONIST_ACTION_STATE_DISCARD,
    COLONIST_TURN_STATE_ROLL,
    COLONIST_TURN_STATE_PLAY,
    COLONIST_TURN_STATE_GAME_OVER,
    TRADE_RESPONSE_PENDING,
    TRADE_RESPONSE_ACCEPT,
    MSG_TYPE_AVAILABLE_SETTLEMENTS,
    MSG_TYPE_AVAILABLE_ROADS,
    MSG_TYPE_AVAILABLE_CITIES,
    ANTICHEAT_ENABLED,
    ANTICHEAT_MIN_THINK_MS,
    ANTICHEAT_MAX_THINK_MS,
    ANTICHEAT_ACTION_INTERVAL_MS,
    ACTION_BUILD_ROAD_DEV,
)
from bridge.card_tracker import OpponentCardTracker
from catanatron.models.enums import ActionType, DEVELOPMENT_CARDS, SETTLEMENT, ROAD

logger = logging.getLogger(__name__)


class BridgeServer:
    """WebSocket bridge server connecting colonist.io to Catanatron."""

    def __init__(self, bot_type: str = "value", anticheat: bool = None,
                 search_depth: int = 2, blend_weight: float = 1e8,
                 bc_model_path: str = "robottler/models/value_net_v2.pt",
                 trade_strategy: str = "blend",
                 propose_trades: bool = False):
        self.translator = StateTranslator()
        self.action_translator: Optional[ActionTranslator] = None
        self.bot = BotInterface(
            bot_type=bot_type, search_depth=search_depth,
            blend_weight=blend_weight, bc_model_path=bc_model_path,
            trade_strategy=trade_strategy,
        )
        self.game_logger = GameLogger()
        self.card_tracker = OpponentCardTracker()
        self.queue: asyncio.Queue = asyncio.Queue()

        # Pending multi-step action state
        self._pending_robber_steal_color = None  # Catanatron Color to steal from
        self._pending_monopoly_resource = None
        self._pending_yop_resources = None
        self._road_building_roads_sent = 0
        self._attempted_dev_play_this_turn = False  # Prevent retry if dev card play fails

        # Timing / watchdog
        self._last_action_sent_time: float = time.monotonic()
        self._last_our_turn_msg_time: float = 0.0

        # Background tasks
        self._consumer_task: Optional[asyncio.Task] = None
        self._producer_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

        # Index maps from type 4 initial state
        self._vertex_index_to_coord: Dict[int, tuple] = {}
        self._edge_index_to_coord: Dict[int, tuple] = {}
        self._vertex_coord_to_idx: Dict[tuple, int] = {}
        self._edge_coord_to_idx: Dict[tuple, int] = {}

        # Available action positions (updated by types 30/31/32)
        self._available_settlements: List[int] = []
        self._available_roads: List[int] = []
        self._available_cities: List[int] = []

        # Tile data for action descriptions
        self._tile_by_index: Dict[int, Dict] = {}
        self._tile_by_coord: Dict[tuple, Dict] = {}

        # Trade state tracking
        self._active_trade_offers: Dict[str, Dict] = {}

        # Trade proposal (outbound) — only active when --propose-trades is set
        self._propose_trades = propose_trades
        self._pending_our_trade: bool = False
        self._pending_our_trade_id: Optional[str] = None  # trade_id for pending proposal
        self._trade_proposal_time: float = 0.0
        self._trades_proposed_this_turn: int = 0
        self._first_accept_time: float = 0.0
        self._pending_trade_proposal: Optional[Dict] = None  # cached proposal for acceptee eval

        # Active websocket connection (reject duplicates)
        self._active_ws = None

        # Double-action prevention
        self._dice_rolled_this_turn = False

        # Anti-cheat delay settings
        self._anticheat = anticheat if anticheat is not None else ANTICHEAT_ENABLED
        self._ac_min_think = ANTICHEAT_MIN_THINK_MS / 1000.0
        self._ac_max_think = ANTICHEAT_MAX_THINK_MS / 1000.0
        self._ac_action_interval = ANTICHEAT_ACTION_INTERVAL_MS / 1000.0
        if self._anticheat:
            logger.info(
                f"Anticheat enabled: think={self._ac_min_think:.1f}-{self._ac_max_think:.1f}s, "
                f"click interval={self._ac_action_interval:.3f}s"
            )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def handle_connection(self, websocket):
        """Handle a WebSocket connection with proper cleanup in finally block."""
        # Reject duplicate connections
        if self._active_ws is not None:
            logger.warning("Rejecting duplicate WebSocket connection (already have an active one)")
            await websocket.close(1008, "Duplicate connection rejected")
            return

        self._active_ws = websocket
        logger.info("Userscript connected")
        self.game_logger.start_game()
        try:
            self._consumer_task = asyncio.create_task(
                self._consumer(websocket), name="consumer"
            )
            self._producer_task = asyncio.create_task(
                self._producer(websocket), name="producer"
            )
            self._watchdog_task = asyncio.create_task(
                self._watchdog(), name="watchdog"
            )
            done, pending = await asyncio.wait(
                [self._consumer_task, self._producer_task, self._watchdog_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    logger.error(f"Task {task.get_name()} failed: {exc}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in handle_connection: {e}", exc_info=True)
        finally:
            for task in [self._consumer_task, self._producer_task, self._watchdog_task]:
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            self._active_ws = None
            # Reset ephemeral turn state on disconnect
            self._clear_pending_trade()
            self._trades_proposed_this_turn = 0
            self._dice_rolled_this_turn = False
            self._attempted_dev_play_this_turn = False
            self._active_trade_offers.clear()
            self.game_logger.end_game()
            logger.info("Connection closed, all tasks cleaned up")

    # ------------------------------------------------------------------
    # Consumer / producer
    # ------------------------------------------------------------------

    async def _consumer(self, websocket):
        """Consume incoming messages with error handling."""
        iterator = websocket.__aiter__()
        while True:
            try:
                if hasattr(iterator, "__anext__"):
                    raw_message = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=WS_RECV_TIMEOUT_SECONDS,
                    )
                else:
                    try:
                        raw_message = next(iterator)
                    except StopIteration:
                        break
            except asyncio.TimeoutError:
                logger.warning(
                    f"WebSocket receive timed out after {WS_RECV_TIMEOUT_SECONDS}s"
                )
                break
            except StopAsyncIteration:
                break

            try:
                msg = parse_message(raw_message)
                self.game_logger.log_incoming(type(msg).__name__, str(raw_message))

                if isinstance(msg, UnknownMsg):
                    self.game_logger.log_security_event(
                        "unknown_message",
                        f"Unrecognized message structure: {str(raw_message)[:200]}",
                    )

                await self._handle_message(msg)
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                self.game_logger.log_security_event(
                    "handler_exception",
                    f"Exception in message handler: {e}",
                )

    async def _producer(self, websocket):
        """Send outgoing messages with timeout to prevent hangs."""
        while True:
            message = await self.queue.get()
            if self._anticheat:
                await asyncio.sleep(random.uniform(0.05, self._ac_action_interval))
            try:
                await asyncio.wait_for(
                    websocket.send(message),
                    timeout=WS_SEND_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(f"WebSocket send timed out after {WS_SEND_TIMEOUT_SECONDS}s")
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket closed during send")
                break

    async def _watchdog(self):
        """Watchdog: auto-sends END_TURN if no action sent within watchdog timeout."""
        while True:
            await asyncio.sleep(5)

            # Check for pending trade — finalize if accepted, timeout if not
            if self._pending_our_trade:
                elapsed = time.monotonic() - self._trade_proposal_time
                trade_id = self._pending_our_trade_id
                if elapsed > 5.0:
                    if trade_id:
                        # Try to execute if anyone accepted (safety net)
                        logger.info("Trade proposal watchdog: checking for acceptees")
                        await self._execute_pending_trade(trade_id)
                    else:
                        logger.info("Trade proposal timed out (watchdog), resuming turn")
                        self._clear_pending_trade()
                        if self.translator.is_our_turn():
                            await self._safe_execute(self._handle_our_turn)

            if self._last_our_turn_msg_time <= 0:
                continue
            elapsed = time.monotonic() - self._last_our_turn_msg_time
            since_last_action = time.monotonic() - self._last_action_sent_time
            if elapsed > WATCHDOG_TIMEOUT_SECONDS and since_last_action > WATCHDOG_TIMEOUT_SECONDS:
                logger.warning(
                    f"Watchdog triggered: {elapsed:.1f}s since last turn msg, "
                    f"{since_last_action:.1f}s since last action. Auto-sending END_TURN."
                )
                self._send({"action": 5})
                self._last_our_turn_msg_time = 0.0

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _handle_message(self, msg) -> None:
        """Route a parsed message to the appropriate handler."""
        if isinstance(msg, GameInitMsg):
            await self._handle_game_init(msg)

        elif isinstance(msg, GameStateDiffMsg):
            await self._handle_game_state_diff(msg)

        elif isinstance(msg, ResourceDistributionMsg):
            self.translator.update_resources_from_distribution(msg.distributions)
            self.card_tracker.on_resource_distribution(
                msg.distributions, self.translator.state.my_color_index,
            )

        elif isinstance(msg, TradeExecutionMsg):
            self.translator.update_resources_from_trade(
                msg.giving_player,
                msg.receiving_player,
                msg.giving_cards,
                msg.receiving_cards,
            )
            self.card_tracker.on_trade(
                msg.giving_player, msg.receiving_player,
                msg.giving_cards, msg.receiving_cards,
                self.translator.state.my_color_index,
            )

        elif isinstance(msg, DiscardPromptMsg):
            # Sync resources from the discard prompt's validCardsToSelect (ground truth)
            if msg.valid_cards:
                new_resources = {r: 0 for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]}
                for card_id in msg.valid_cards:
                    resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                    if resource:
                        new_resources[resource] += 1
                self.translator.state.my_resources = new_resources
            await self._safe_execute(lambda: self._handle_discard(msg.amount))

        elif isinstance(msg, RobVictimPromptMsg):
            await self._safe_execute(
                lambda: self._handle_rob_selection(msg.players_to_select)
            )

        elif isinstance(msg, AvailableActionsMsg):
            self._handle_available_actions(msg)

        elif isinstance(msg, GameOverMsg):
            logger.info("Game over received")

        elif isinstance(msg, (HeartbeatMsg, EndTurnMsg)):
            pass  # Ignore heartbeats and end-turn confirmations

    # ------------------------------------------------------------------
    # Game init handler
    # ------------------------------------------------------------------

    async def _handle_game_init(self, msg: GameInitMsg) -> None:
        """Handle type 4: game initialization with board, color, and player order.

        Also handles re-initialization on mid-game reconnect — clears stale
        index maps and ephemeral state so the bot starts fresh.
        """
        # Clear index maps and tile data from previous init (reconnect safety)
        self._vertex_index_to_coord.clear()
        self._edge_index_to_coord.clear()
        self._vertex_coord_to_idx.clear()
        self._edge_coord_to_idx.clear()
        self._tile_by_index.clear()
        self._tile_by_coord.clear()
        self._active_trade_offers.clear()
        # Reset ephemeral turn state
        self._clear_pending_trade()
        self._trades_proposed_this_turn = 0
        self._dice_rolled_this_turn = False
        self._attempted_dev_play_this_turn = False

        # Set our color and player order
        self.translator.set_my_color(msg.player_color)
        self.translator.set_player_order(msg.play_order)

        # Initialize card tracker for opponents
        self.card_tracker.reset()
        for col_idx in msg.play_order:
            if col_idx != msg.player_color:
                self.card_tracker.init_player(col_idx)

        game_state = msg.game_state
        map_state = game_state.get("mapState", {})

        # Build index → coord maps from initial state
        self._build_index_maps_from_init(map_state)

        # Store tile data for action descriptions
        self._store_tile_data(map_state.get("tileHexStates", {}))

        # Convert tileHexStates → list of tile dicts for coordinate_map.py
        tiles = self._convert_tile_hex_states(map_state.get("tileHexStates", {}))
        vertices = self._convert_tile_corner_states(map_state.get("tileCornerStates", {}))
        edges = self._convert_tile_edge_states(map_state.get("tileEdgeStates", {}))

        # Populate vertex harborType from portEdgeStates (ports are on edges, not vertices)
        port_edge_states = map_state.get("portEdgeStates", {})
        self._apply_port_edge_harbors(port_edge_states, vertices)

        # Setup the board coordinate mapping
        self.translator.setup_board(tiles, vertices, edges)

        # Override port_nodes on the CatanMap using authoritative portEdgeStates
        self._override_port_nodes(port_edge_states)

        # Store index maps in state translator for later use
        self.translator.state.vertex_index_to_coord = dict(self._vertex_index_to_coord)
        self.translator.state.edge_index_to_coord = dict(self._edge_index_to_coord)

        # Parse existing buildings/roads/state for mid-game reconnect
        self._parse_existing_state(game_state)

        self._refresh_action_translator()
        logger.info(
            f"Game initialized: color={msg.player_color}, "
            f"order={msg.play_order}, "
            f"vertices={len(self._vertex_index_to_coord)}, "
            f"edges={len(self._edge_index_to_coord)}"
        )

        # If reconnecting mid-game and it's our turn, trigger the bot
        if self.translator.is_our_turn():
            logger.info("Reconnected during our turn — triggering bot")
            self._last_our_turn_msg_time = time.monotonic()
            await self._safe_execute(self._handle_our_turn)

    def _build_index_maps_from_init(self, map_state: Dict) -> None:
        """Build vertex/edge index→coord and coord→index maps from type 4 mapState."""
        # Vertex index → (x, y, z)
        corner_states = map_state.get("tileCornerStates", {})
        for idx_str, data in corner_states.items():
            idx = int(idx_str)
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            z = int(data.get("z", 0))
            coord = (x, y, z)
            self._vertex_index_to_coord[idx] = coord
            self._vertex_coord_to_idx[coord] = idx

        # Edge index → (x, y, z)
        # Edges may not all be present in the initial state (only populated as roads are built)
        # But tileEdgeStates in the initial gameState should have all edge positions
        edge_states = map_state.get("tileEdgeStates", {})
        for idx_str, data in edge_states.items():
            idx = int(idx_str)
            if isinstance(data, dict) and "x" in data:
                x = int(data.get("x", 0))
                y = int(data.get("y", 0))
                z = int(data.get("z", 0))
                coord = (x, y, z)
                self._edge_index_to_coord[idx] = coord
                self._edge_coord_to_idx[coord] = idx

    def _convert_tile_hex_states(self, tile_hex_states: Dict) -> List[Dict]:
        """Convert tileHexStates dict to list of tile dicts for coordinate_map.py.

        Input: {"0": {"x": 0, "y": -2, "type": 2, "diceNumber": 11}, ...}
        Output: [{"hexFace": {"x": 0, "y": -2}, "tileType": 2, "diceNumber": 11}, ...]
        """
        tiles = []
        for idx_str in sorted(tile_hex_states.keys(), key=int):
            data = tile_hex_states[idx_str]
            tiles.append({
                "hexFace": {"x": data.get("x", 0), "y": data.get("y", 0)},
                "tileType": data.get("type", 0),
                "diceNumber": data.get("diceNumber", 0),
            })
        return tiles

    def _convert_tile_corner_states(self, corner_states: Dict) -> List[Dict]:
        """Convert tileCornerStates dict to list of vertex dicts for coordinate_map.py.

        Input: {"0": {"x": 0, "y": -2, "z": 0}, ...}
        Output: [{"hexCorner": {"x": 0, "y": -2, "z": 0}, "harborType": 0}, ...]
        """
        vertices = []
        for idx_str in sorted(corner_states.keys(), key=int):
            data = corner_states[idx_str]
            vertices.append({
                "hexCorner": {
                    "x": data.get("x", 0),
                    "y": data.get("y", 0),
                    "z": data.get("z", 0),
                },
                "harborType": data.get("harborType", 0),
            })
        return vertices

    def _convert_tile_edge_states(self, edge_states: Dict) -> List[Dict]:
        """Convert tileEdgeStates dict to list of edge dicts for coordinate_map.py.

        Input: {"0": {"x": 0, "y": -2, "z": 0}, ...}
        Output: [{"hexEdge": {"x": 0, "y": -2, "z": 0}}, ...]
        """
        edges = []
        for idx_str in sorted(edge_states.keys(), key=int):
            data = edge_states[idx_str]
            if isinstance(data, dict) and "x" in data:
                edges.append({
                    "hexEdge": {
                        "x": data.get("x", 0),
                        "y": data.get("y", 0),
                        "z": data.get("z", 0),
                    },
                })
        return edges

    def _apply_port_edge_harbors(
        self, port_edge_states: Dict, vertices: List[Dict]
    ) -> None:
        """Set harborType on vertices from portEdgeStates.

        colonist.io stores port info on edges, not vertices. Each port edge
        connects two vertices. We compute those vertices and set harborType
        so _determine_port_resources can detect them.

        Edge-to-vertex formulas (from colonist.io board.py):
          edge(x,y,0): connects vertex(x,y,0) and vertex(x,y-1,1)
          edge(x,y,1): connects vertex(x,y-1,1) and vertex(x-1,y+1,0)
          edge(x,y,2): connects vertex(x-1,y+1,0) and vertex(x,y,1)
        """
        if not port_edge_states:
            return

        # Build vertex coord → vertex dict index for fast lookup
        coord_to_vertex = {}
        for v in vertices:
            corner = v.get("hexCorner", {})
            coord = (corner.get("x", 0), corner.get("y", 0), corner.get("z", 0))
            coord_to_vertex[coord] = v

        for idx_str, data in port_edge_states.items():
            if not isinstance(data, dict):
                continue
            ex = data.get("x", 0)
            ey = data.get("y", 0)
            ez = data.get("z", 0)
            port_type = data.get("type", 0)
            if port_type <= 0:
                continue

            # Compute the two vertices this edge connects
            if ez == 0:
                v1_coord = (ex, ey, 0)
                v2_coord = (ex, ey - 1, 1)
            elif ez == 1:
                v1_coord = (ex, ey - 1, 1)
                v2_coord = (ex - 1, ey + 1, 0)
            else:  # ez == 2
                v1_coord = (ex - 1, ey + 1, 0)
                v2_coord = (ex, ey, 1)

            # Set harborType on both vertices
            for vc in (v1_coord, v2_coord):
                vtx = coord_to_vertex.get(vc)
                if vtx is not None:
                    vtx["harborType"] = port_type
                else:
                    logger.debug(
                        f"Port edge ({ex},{ey},{ez}) vertex {vc} not in vertex list"
                    )

        port_count = sum(
            1 for v in vertices if v.get("harborType", 0) > 0
        )
        logger.info(f"Applied port harbors: {port_count} vertices marked from {len(port_edge_states)} port edges")

    def _override_port_nodes(self, port_edge_states: Dict) -> None:
        """Override CatanMap.port_nodes using authoritative portEdgeStates.

        Instead of relying on template-based port detection (which can miss ports),
        directly build port_nodes from colonist.io's port edge data + our vertex mapping.

        Each port edge has two vertices. We convert those to Catanatron node IDs
        and map them to the port's resource type.
        """
        mapper = self.translator.state.mapper
        if mapper is None or not port_edge_states:
            return

        catan_map = mapper.catan_map
        harbor_to_resource = {
            1: None,      # 3:1 generic
            2: "WOOD",
            3: "BRICK",
            4: "SHEEP",
            5: "WHEAT",
            6: "ORE",
        }

        from collections import defaultdict
        new_port_nodes = defaultdict(set)

        for idx_str, data in port_edge_states.items():
            if not isinstance(data, dict):
                continue
            ex = data.get("x", 0)
            ey = data.get("y", 0)
            ez = data.get("z", 0)
            port_type = data.get("type", 0)
            if port_type <= 0:
                continue

            resource = harbor_to_resource.get(port_type)

            # Compute the two vertices this edge connects
            if ez == 0:
                v1 = (ex, ey, 0)
                v2 = (ex, ey - 1, 1)
            elif ez == 1:
                v1 = (ex, ey - 1, 1)
                v2 = (ex - 1, ey + 1, 0)
            else:  # ez == 2
                v1 = (ex - 1, ey + 1, 0)
                v2 = (ex, ey, 1)

            # Convert colonist vertex coords to Catanatron node IDs
            for vc in (v1, v2):
                node_id = mapper.colonist_vertex_to_catan.get(vc)
                if node_id is not None:
                    new_port_nodes[resource].add(node_id)

        # Override the CatanMap's port_nodes
        catan_map.port_nodes = dict(new_port_nodes)
        total = sum(len(nodes) for nodes in new_port_nodes.values())
        logger.info(
            f"Port nodes override: {total} nodes across {len(new_port_nodes)} port types "
            f"({', '.join(str(k) + ':' + str(len(v)) for k, v in new_port_nodes.items())})"
        )

    # ------------------------------------------------------------------
    # Existing state parser (mid-game reconnect)
    # ------------------------------------------------------------------

    def _parse_existing_state(self, game_state: Dict) -> None:
        """Parse existing buildings, roads, resources, and turn state from type 4 gameState.

        On a mid-game reconnect, gameState contains the full current state of the
        game, not just the initial board layout. This method extracts that state
        so the bot can resume correctly.
        """
        map_state = game_state.get("mapState", {})
        buildings_found = 0
        roads_found = 0

        # 1. Existing buildings from tileCornerStates
        corner_states = map_state.get("tileCornerStates", {})
        for idx_str, data in corner_states.items():
            if not isinstance(data, dict):
                continue
            owner = data.get("owner", -1)
            if owner < 0:
                continue
            building_type = data.get("buildingType", 1)
            idx = int(idx_str)
            coord = self._vertex_index_to_coord.get(idx)
            if coord is None:
                continue
            if building_type == 2:
                # City: synthesize a settlement first so board.build_settlement()
                # runs during replay, setting up connected_components and
                # board_buildable_ids. Then the city upgrade works correctly.
                self.translator.update_vertex(
                    coord[0], coord[1], coord[2],
                    owner, 1,  # settlement first
                )
            self.translator.update_vertex(
                coord[0], coord[1], coord[2],
                owner, building_type,
            )
            buildings_found += 1

        # 2. Existing roads from tileEdgeStates
        edge_states = map_state.get("tileEdgeStates", {})
        for idx_str, data in edge_states.items():
            if not isinstance(data, dict):
                continue
            owner = data.get("owner", -1)
            if owner < 0:
                continue
            idx = int(idx_str)
            coord = self._edge_index_to_coord.get(idx)
            if coord is None:
                continue
            self.translator.update_edge(
                coord[0], coord[1], coord[2], owner,
            )
            roads_found += 1

        # 3. Robber position
        robber_state = game_state.get("mechanicRobberState")
        if isinstance(robber_state, dict):
            tile_index = robber_state.get("locationTileIndex")
            if tile_index is not None:
                self.translator.update_robber_by_tile_index(tile_index)

        # 4. Our resources from playerStates
        player_states = game_state.get("playerStates")
        if isinstance(player_states, dict):
            self._sync_resources_from_player_states(player_states)

        # 4b. Dev card state (hand, played, bought-this-turn, largest army)
        dev_state = game_state.get("mechanicDevelopmentCardsState")
        if isinstance(dev_state, dict):
            logger.info(
                f"Init dev card state: player_keys={list(dev_state.keys())}, "
                f"my_col={self.translator.state.my_color_index}"
            )
            self._process_dev_card_diff(dev_state)
        else:
            logger.info(f"Init: no mechanicDevelopmentCardsState found")

        # 5. Turn/action state from currentState
        # Always parse actionState + currentTurnPlayerColor (needed for setup too).
        # Only pass turnState through if present — update_turn_state sets
        # is_setup_phase=False when turnState is not None.
        current_state = game_state.get("currentState")
        if isinstance(current_state, dict):
            turn_state = current_state.get("turnState")
            action_state = current_state.get("actionState")
            current_player = current_state.get("currentTurnPlayerColor")

            self.translator.update_turn_state(
                current_turn_state=turn_state,
                current_action_state=action_state,
                current_turn_player_color=current_player,
            )

        if buildings_found > 0 or roads_found > 0:
            logger.info(
                f"Parsed existing state: {buildings_found} buildings, "
                f"{roads_found} roads, resources={self.translator.state.my_resources}"
            )

    # ------------------------------------------------------------------
    # Game state diff handler
    # ------------------------------------------------------------------

    async def _handle_game_state_diff(self, msg: GameStateDiffMsg) -> None:
        """Handle type 91: process diff sub-sections."""
        diff = msg.diff
        logger.debug(f"Type 91 diff keys: {list(diff.keys())}")

        # 1. Process building updates (before turn state, so state is current when we decide)
        map_state = diff.get("mapState")
        if map_state:
            self._process_building_diffs(map_state)

        # 2. Process robber movement
        robber_state = diff.get("mechanicRobberState")
        if robber_state:
            tile_index = robber_state.get("locationTileIndex")
            if tile_index is not None:
                self.translator.update_robber_by_tile_index(tile_index)

        # 3. Sync our resource hand from authoritative playerStates
        player_states = diff.get("playerStates")
        if player_states:
            self._sync_resources_from_player_states(player_states)

        # 3b. Process dev card state (hand, played, bought-this-turn, largest army)
        dev_state = diff.get("mechanicDevelopmentCardsState")
        if dev_state:
            logger.info(
                f"Dev card diff received: player_keys={list(dev_state.keys())}, "
                f"my_col={self.translator.state.my_color_index}"
            )
            self._process_dev_card_diff(dev_state)

        # 4. Process trade state
        trade_state = diff.get("tradeState")
        if trade_state:
            await self._process_trade_diff(trade_state)

        # 5. Process turn/action state changes (last, so it triggers our turn with up-to-date state)
        current_state = diff.get("currentState")
        if current_state:
            turn_state = current_state.get("turnState")
            action_state = current_state.get("actionState")
            current_player = current_state.get("currentTurnPlayerColor")

            self.translator.update_turn_state(
                current_turn_state=turn_state,
                current_action_state=action_state,
                current_turn_player_color=current_player,
            )

            self._refresh_action_translator()

            if self.translator.is_our_turn():
                self._last_our_turn_msg_time = time.monotonic()
                await self._safe_execute(self._handle_our_turn)
            else:
                # Not our turn — reset flags for our next turn
                self._dice_rolled_this_turn = False
                self._attempted_dev_play_this_turn = False
                self._trades_proposed_this_turn = 0
                self._clear_pending_trade()
                # Reset road building flags on turn change
                self.translator.state.is_road_building = False
                self.translator.state.free_roads_available = 0

    def _process_building_diffs(self, map_state: Dict) -> None:
        """Process building placement diffs from mapState.

        Type 91 diffs only include CHANGED fields. For a city upgrade, owner
        doesn't change so it may be absent — we look it up from build_history.
        """
        # Settlement/city updates
        corners = map_state.get("tileCornerStates")
        if corners:
            for idx_str, data in corners.items():
                if not isinstance(data, dict):
                    continue
                idx = int(idx_str)
                owner = data.get("owner", -1)
                building_type = data.get("buildingType", -1)

                # Skip entries with neither owner nor buildingType
                if owner < 0 and building_type < 0:
                    continue

                coord = self._vertex_index_to_coord.get(idx)
                if coord is None:
                    logger.warning(f"Unknown vertex index {idx} in diff")
                    continue

                # If owner is missing from diff, look up from existing build history
                if owner < 0:
                    for event in reversed(self.translator.state.build_history):
                        if event.col_coord == coord:
                            owner = event.color_index
                            break
                    if owner < 0:
                        logger.warning(f"Vertex diff at idx={idx} has no owner and none in history")
                        continue

                # If buildingType is missing, default to settlement (new placement)
                if building_type < 0:
                    building_type = 1

                self.translator.update_vertex(
                    coord[0], coord[1], coord[2],
                    owner, building_type,
                )

                # Track opponent building costs
                my_col = self.translator.state.my_color_index
                if owner != my_col and not self.translator.state.is_setup_phase:
                    if building_type == 1:
                        self.card_tracker.on_build(owner, "SETTLEMENT")
                    elif building_type == 2:
                        self.card_tracker.on_build(owner, "CITY")

        # Road updates
        edges = map_state.get("tileEdgeStates")
        if edges:
            for idx_str, data in edges.items():
                if not isinstance(data, dict):
                    continue
                idx = int(idx_str)
                owner = data.get("owner", -1)

                # If owner is missing, look up from existing build history
                if owner < 0 and "owner" not in data:
                    coord = self._edge_index_to_coord.get(idx)
                    if coord is not None:
                        for event in reversed(self.translator.state.build_history):
                            if event.col_coord == coord and event.building_type == ROAD:
                                owner = event.color_index
                                break

                if owner < 0:
                    continue

                coord = self._edge_index_to_coord.get(idx)
                if coord is None:
                    logger.warning(f"Unknown edge index {idx} in diff")
                    continue

                self.translator.update_edge(
                    coord[0], coord[1], coord[2], owner,
                )

                # Track opponent road costs
                my_col = self.translator.state.my_color_index
                if owner != my_col and not self.translator.state.is_setup_phase:
                    self.card_tracker.on_build(owner, "ROAD")

    def _sync_resources_from_player_states(self, player_states: Dict) -> None:
        """Sync resource hands from authoritative playerStates in type 91 diff.

        colonist.io sends the full card list under
        playerStates.<color>.resourceCards.cards as an array of resource IDs.
        For our player, card IDs are real resource types (1-5).
        For opponents, unknown cards appear as id=0; we distribute them evenly
        across the 5 resource types to preserve total count for robber checks.
        """
        my_col = self.translator.state.my_color_index
        resource_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

        for col_str, p_data in player_states.items():
            try:
                col_idx = int(col_str)
            except (ValueError, TypeError):
                continue
            if not isinstance(p_data, dict):
                continue

            rc = p_data.get("resourceCards")
            if not isinstance(rc, dict):
                continue
            cards = rc.get("cards")
            if not isinstance(cards, list):
                continue

            if col_idx == my_col:
                # Our player: card IDs are real resource types
                new_resources = {r: 0 for r in resource_names}
                for card_id in cards:
                    resource = COLONIST_RESOURCE_TO_CATAN.get(card_id)
                    if resource:
                        new_resources[resource] += 1
                self.translator.state.my_resources = new_resources
            else:
                # Opponent: reconcile card tracker with authoritative total,
                # then use tracked known resources + distributed unknowns
                authoritative_total = len(cards)
                self.card_tracker.reconcile(col_idx, authoritative_total)
                self.translator.state.opponent_resources[col_idx] = (
                    self.card_tracker.get_resources_for_state(col_idx)
                )
                self.card_tracker.log_state(col_idx)

    def _process_dev_card_diff(self, dev_state: Dict) -> None:
        """Process mechanicDevelopmentCardsState from type 91 diff or type 4 init.

        Syncs dev card hand, played cards, bought-this-turn, and used-this-turn
        for our player, plus developmentCardsUsed for ALL players (for largest army).

        Structure: {bankDevelopmentCards: ..., players: {colorIdx: playerData, ...}}
        OR (in some diffs): {colorIdx: playerData, ...} directly.
        """
        my_col = self.translator.state.my_color_index

        # Per-player data may be nested under 'players' key or at top level
        players_dict = dev_state.get("players", dev_state)
        if not isinstance(players_dict, dict):
            return

        for col_str, player_data in players_dict.items():
            # Keys are stringified color indices (e.g. "1", "5")
            try:
                col_idx = int(col_str)
            except (ValueError, TypeError):
                continue
            if not isinstance(player_data, dict):
                continue

            # Sync developmentCardsUsed for ALL players (needed for largest army)
            used_list = player_data.get("developmentCardsUsed")
            if isinstance(used_list, list):
                self.translator.state.all_players_dev_cards_used[col_idx] = list(used_list)

            # For our player, additionally sync hand and turn state
            if col_idx != my_col:
                continue

            logger.info(
                f"Dev card diff for our player (col={col_idx}): "
                f"keys={list(player_data.keys())}"
            )

            # Sync dev card hand from developmentCards.cards
            # NOTE: colonist.io hides all card types as id=10 in the hand array.
            # We count hidden cards and assign them as Knights (most likely type).
            dev_cards_obj = player_data.get("developmentCards")
            if isinstance(dev_cards_obj, dict):
                cards = dev_cards_obj.get("cards")
                if isinstance(cards, list):
                    new_hand = {c: 0 for c in DEVELOPMENT_CARDS}
                    hidden_count = 0
                    for card_id in cards:
                        dev_name = COLONIST_VALUE_TO_DEVCARD.get(card_id)
                        if dev_name and dev_name in new_hand:
                            new_hand[dev_name] += 1
                        elif card_id == 10:
                            hidden_count += 1
                        else:
                            logger.warning(f"Unknown dev card id {card_id} in hand")
                    self.translator.state.unknown_dev_card_count = hidden_count
                    self.translator.state.my_dev_cards = new_hand
                    if hidden_count > 0:
                        logger.info(
                            f"Dev card hand: {hidden_count} hidden, "
                            f"{sum(new_hand.values())} known"
                        )

            # Sync played dev cards — detect newly played cards and remove from known types
            used = player_data.get("developmentCardsUsed")
            if isinstance(used, list):
                old_played = dict(self.translator.state.played_dev_cards)
                new_played = {c: 0 for c in DEVELOPMENT_CARDS}
                for card_id in used:
                    dev_name = COLONIST_VALUE_TO_DEVCARD.get(card_id)
                    if dev_name and dev_name in new_played:
                        new_played[dev_name] += 1
                # Remove newly played cards from known_dev_card_types
                for card_name in DEVELOPMENT_CARDS:
                    newly_played = new_played[card_name] - old_played.get(card_name, 0)
                    for _ in range(newly_played):
                        if card_name in self.translator.state.known_dev_card_types:
                            self.translator.state.known_dev_card_types.remove(card_name)
                self.translator.state.played_dev_cards = new_played

            # Sync dev cards bought this turn — track actual card types
            bought = player_data.get("developmentCardsBoughtThisTurn")
            if isinstance(bought, list):
                # Add newly revealed card types to our knowledge
                prev_count = self.translator.state._bought_this_turn_processed
                if len(bought) > prev_count:
                    for card_id in bought[prev_count:]:
                        dev_name = COLONIST_VALUE_TO_DEVCARD.get(card_id)
                        if dev_name:
                            self.translator.state.known_dev_card_types.append(dev_name)
                            self.translator.state.dev_card_types_bought_this_turn.append(dev_name)
                            logger.info(f"Bought dev card: {dev_name} (id={card_id})")
                        else:
                            logger.info(f"Bought dev card: hidden (id={card_id})")
                    self.translator.state._bought_this_turn_processed = len(bought)
                self.translator.state.dev_cards_bought_this_turn = len(bought)
            elif bought is None and "developmentCardsBoughtThisTurn" in player_data:
                # Explicit null = reset (new turn)
                self.translator.state.dev_cards_bought_this_turn = 0
                self.translator.state._bought_this_turn_processed = 0
                self.translator.state.dev_card_types_bought_this_turn = []

            # Sync has used dev card this turn
            has_used = player_data.get("hasUsedDevelopmentCardThisTurn")
            if has_used is not None:
                self.translator.state.has_used_dev_card_this_turn = bool(has_used)
            elif "hasUsedDevelopmentCardThisTurn" in player_data:
                # Explicit null = reset
                self.translator.state.has_used_dev_card_this_turn = False

    async def _process_trade_diff(self, trade_state: Dict) -> None:
        """Process trade state diffs from type 91."""
        active_offers = trade_state.get("activeOffers")
        if not active_offers:
            return

        my_col_idx = self.translator.state.my_color_index
        my_col_str = str(my_col_idx)

        for trade_id, offer_data in active_offers.items():
            if offer_data is None:
                # Trade was closed/cancelled
                self._active_trade_offers.pop(trade_id, None)
                continue

            if not isinstance(offer_data, dict):
                continue

            # New trade offer
            if "creator" in offer_data:
                self._active_trade_offers[trade_id] = offer_data
                creator = offer_data.get("creator")
                if creator == my_col_idx:
                    # Track the trade_id for our pending proposal
                    if self._pending_our_trade:
                        self._pending_our_trade_id = trade_id
                    continue  # We created this trade, skip responding

                # Check if we need to respond (playerResponses has our color with 0=pending)
                responses = offer_data.get("playerResponses", {})
                if responses.get(my_col_str) == TRADE_RESPONSE_PENDING:
                    await self._safe_execute(
                        lambda tid=trade_id, od=offer_data: self._handle_trade_offer_from_diff(tid, od)
                    )

            # Trade response update (someone responded)
            elif "playerResponses" in offer_data:
                # Merge response update into tracked offer
                existing = self._active_trade_offers.get(trade_id, {})
                existing_responses = existing.get("playerResponses", {})
                existing_responses.update(offer_data["playerResponses"])
                existing["playerResponses"] = existing_responses
                self._active_trade_offers[trade_id] = existing

                # Handle responses to OUR trade proposal
                if self._pending_our_trade and existing.get("creator") == my_col_idx:
                    await self._handle_our_trade_responses(trade_id, existing_responses)

    async def _handle_our_trade_responses(self, trade_id: str, responses: Dict) -> None:
        """Handle responses to a trade WE proposed.

        On first accept, schedules a 2s delayed finalization to wait for more
        responses.  If all respond before the timer, finalizes immediately.
        """
        my_col_str = str(self.translator.state.my_color_index)

        acceptees = [int(pid) for pid, resp in responses.items()
                     if pid != my_col_str and resp == TRADE_RESPONSE_ACCEPT]
        all_responded = all(
            resp != TRADE_RESPONSE_PENDING
            for pid, resp in responses.items()
            if pid != my_col_str
        )

        if not acceptees:
            if all_responded:
                logger.info("All opponents rejected our trade, resuming turn")
                self._clear_pending_trade()
                await self._safe_execute(self._handle_our_turn)
            return

        if all_responded:
            # Everyone answered — finalize now
            await self._execute_pending_trade(trade_id)
            return

        # First accept: schedule delayed finalization (only once)
        if self._first_accept_time == 0.0:
            self._first_accept_time = time.monotonic()
            logger.info(f"First accept on our trade from {acceptees}, waiting 2s for more")
            asyncio.ensure_future(self._delayed_trade_finalize(trade_id))

    async def _delayed_trade_finalize(self, trade_id: str) -> None:
        """Wait 2s after first accept, then execute the trade."""
        await asyncio.sleep(2.0)
        if not self._pending_our_trade:
            return  # Already handled (all responded or watchdog fired)
        logger.info("2s accept wait complete, finalizing trade")
        await self._execute_pending_trade(trade_id)

    async def _execute_pending_trade(self, trade_id: str) -> None:
        """Pick best acceptee from current responses and execute (or cancel)."""
        if not self._pending_our_trade:
            return

        existing = self._active_trade_offers.get(trade_id, {})
        responses = existing.get("playerResponses", {})
        my_col_str = str(self.translator.state.my_color_index)

        acceptees = [int(pid) for pid, resp in responses.items()
                     if pid != my_col_str and resp == TRADE_RESPONSE_ACCEPT]

        if not acceptees:
            logger.info("No acceptees, resuming turn")
            self._clear_pending_trade()
            await self._safe_execute(self._handle_our_turn)
            return

        proposal = self._pending_trade_proposal
        if proposal is None:
            self._clear_pending_trade()
            await self._safe_execute(self._handle_our_turn)
            return

        game = self.translator.reconstruct_game()
        my_color = self.translator.state.my_catan_color
        if not game or not my_color:
            self._clear_pending_trade()
            await self._safe_execute(self._handle_our_turn)
            return

        opponent_vps = self._get_opponent_vps(game)
        c2c = self.translator.state.colonist_to_catan_color

        best = self.bot.pick_trade_acceptee(
            game, my_color, acceptees,
            offered_catan=proposal.get("_offered_catan", []),
            wanted_catan=proposal.get("_wanted_catan", []),
            delta_us=proposal.get("_delta_us", 0.0),
            opponent_vps=opponent_vps,
            colonist_to_catan=c2c,
            vps_to_win=game.vps_to_win,
        )

        if best is not None:
            logger.info(f"Executing trade {trade_id} with player {best}")
            self._send({"action": 14, "data": {
                "tradeId": trade_id, "playerColor": best,
            }})
        else:
            logger.info("No acceptable trade partner, cancelling")

        self._clear_pending_trade()
        await self._safe_execute(self._handle_our_turn)

    def _clear_pending_trade(self) -> None:
        """Clear all pending trade proposal state."""
        self._pending_our_trade = False
        self._pending_our_trade_id = None
        self._pending_trade_proposal = None
        self._first_accept_time = 0.0

    # ------------------------------------------------------------------
    # Turn handlers
    # ------------------------------------------------------------------

    async def _handle_our_turn(self) -> None:
        """Handle our turn based on the current action/turn state."""
        state = self.translator.state
        action_s = state.current_action_state
        turn_s = state.current_turn_state
        logger.debug(
            f"_handle_our_turn: setup={state.is_setup_phase}, "
            f"action_s={action_s}, turn_s={turn_s}"
        )

        if state.is_setup_phase:
            if action_s == COLONIST_ACTION_STATE_SETUP_SETTLEMENT:
                await self._handle_setup_settlement()
            elif action_s == COLONIST_ACTION_STATE_SETUP_ROAD:
                await self._handle_setup_road()
            else:
                # actionState unknown (e.g. reconnect during setup or first
                # game init before any type 91 diff arrives).  Infer from
                # how many of OUR settlements vs roads have been placed:
                #   settlements <= roads  →  need settlement
                #   settlements >  roads  →  need road
                my_col = state.my_color_index
                my_settlements = sum(
                    1 for e in state.build_history
                    if e.color_index == my_col and e.building_type == SETTLEMENT
                )
                my_roads = sum(
                    1 for e in state.build_history
                    if e.color_index == my_col and e.building_type == ROAD
                )
                if my_settlements > my_roads:
                    logger.info(f"Setup: inferred road (action_state={action_s}, {my_settlements}S/{my_roads}R)")
                    await self._handle_setup_road()
                else:
                    logger.info(f"Setup: inferred settlement (action_state={action_s}, {my_settlements}S/{my_roads}R)")
                    await self._handle_setup_settlement()
            return

        # Road Building dev card: place free roads before other action dispatch
        if state.is_road_building and state.free_roads_available > 0:
            await self._handle_road_building_placement()
            return

        # Special action states take priority (can arrive while turnState is still ROLL)
        if action_s == COLONIST_ACTION_STATE_MOVE_ROBBER:
            await self._handle_move_robber()
            return
        if action_s == COLONIST_ACTION_STATE_ROB_VICTIM:
            await self._handle_rob_selection_auto()
            return
        if action_s == COLONIST_ACTION_STATE_DISCARD:
            # Discard is handled via type 13 DiscardPromptMsg
            return

        # Regular game phase
        if turn_s == COLONIST_TURN_STATE_ROLL:
            # Need to roll dice
            if not self._dice_rolled_this_turn:
                self._send({"action": 4})
                self._dice_rolled_this_turn = True
            return

        if turn_s == COLONIST_TURN_STATE_PLAY or turn_s == -1:
            if action_s == COLONIST_ACTION_STATE_MAIN:
                await self._handle_play_turn()

    async def _handle_setup_settlement(self) -> None:
        """Handle initial settlement placement."""
        game = self.translator.reconstruct_game()
        if game is None:
            logger.warning("Setup settlement: reconstruct_game returned None")
            return
        logger.info(f"Setup settlement: {len(game.playable_actions)} playable actions")
        action = self.bot.decide(game, self.translator.state.my_catan_color)
        if action is None:
            logger.warning("Setup settlement: bot returned None")
            return
        if self.action_translator:
            try:
                commands = self.action_translator.translate(action)
                for cmd in commands:
                    cmd["setup"] = True
                    self._send(cmd)
            except Exception as e:
                logger.error(f"Failed to translate setup settlement: {e}")
        else:
            logger.warning("Setup settlement: no action_translator")

    async def _handle_setup_road(self) -> None:
        """Handle initial road placement."""
        game = self.translator.reconstruct_game()
        if game is None:
            logger.warning("Setup road: reconstruct_game returned None")
            return
        logger.info(f"Setup road: {len(game.playable_actions)} playable actions")
        action = self.bot.decide(game, self.translator.state.my_catan_color)
        if action is None:
            logger.warning("Setup road: bot returned None")
            return
        if self.action_translator:
            try:
                commands = self.action_translator.translate(action)
                for cmd in commands:
                    cmd["setup"] = True
                    self._send(cmd)
            except Exception as e:
                logger.error(f"Failed to translate setup road: {e}")

    async def _handle_play_turn(self) -> None:
        """Handle main turn actions (after rolling).

        Decides ONE action per invocation and sends it, then returns.
        When colonist.io confirms the action via type 91 diff, the consumer
        processes it, updates shadow state, and calls _handle_our_turn again.
        """
        # If we have a pending trade proposal, wait for responses or timeout
        if self._pending_our_trade:
            elapsed = time.monotonic() - self._trade_proposal_time
            if elapsed > 5.0:
                logger.info("Trade proposal timed out (5s), proceeding with turn")
                self._clear_pending_trade()
            else:
                return  # Still waiting for responses

        game = self.translator.reconstruct_game()
        my_color = self.translator.state.my_catan_color
        if not my_color or not game:
            return

        # Trade proposal: escalating threshold, up to N per turn
        max_proposals = self.bot._MAX_PROPOSALS_PER_TURN
        if self._propose_trades and self._trades_proposed_this_turn < max_proposals:
            try:
                opponent_vps = self._get_opponent_vps(game)
                proposal = self.bot.propose_trade(
                    game, my_color, opponent_vps, game.vps_to_win,
                    proposals_this_turn=self._trades_proposed_this_turn,
                )
                self._trades_proposed_this_turn += 1
                if proposal:
                    self._send({"action": 11, "data": {
                        "offered": proposal["offered"],
                        "wanted": proposal["wanted"],
                    }})
                    self._pending_our_trade = True
                    self._trade_proposal_time = time.monotonic()
                    self._first_accept_time = 0.0
                    self._pending_trade_proposal = proposal
                    logger.info(
                        f"Proposed trade #{self._trades_proposed_this_turn}: "
                        f"offer {proposal['offered']} for {proposal['wanted']}"
                    )
                    return  # Wait for responses
            except Exception as e:
                logger.error(f"Trade proposal failed: {e}", exc_info=True)

        # Dev card diagnostics — always log so we can trace issues
        ts = self.translator.state
        hidden = ts.unknown_dev_card_count
        known_types = ts.known_dev_card_types
        has_used = ts.has_used_dev_card_this_turn
        dev_actions = [a for a in game.playable_actions
                      if a.action_type.value.startswith("PLAY_")]
        logger.info(
            f"Dev cards: {hidden} hidden, known={known_types}, "
            f"used_this_turn={has_used}, bought_this_turn={ts.dev_cards_bought_this_turn}, "
            f"playable_dev={[str(a.action_type) for a in dev_actions]}, "
            f"total_actions={len(game.playable_actions)}"
        )

        action = self.bot.decide(game, my_color)
        if action is None:
            return

        logger.info(f"Bot chose: {action.action_type} {action.value}")

        # Guard against infinite dev card play retries —
        # if we already attempted a dev card play this turn and ended up back here,
        # skip dev cards and end turn instead.
        DEV_PLAY_ACTIONS = {
            ActionType.PLAY_KNIGHT_CARD,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_ROAD_BUILDING,
        }
        if action.action_type in DEV_PLAY_ACTIONS:
            if self._attempted_dev_play_this_turn:
                logger.info("Skipping dev card play (already attempted this turn), ending turn")
                self._send({"action": 5})  # END_TURN
                return
            self._attempted_dev_play_this_turn = True

        # Store pending state for multi-step card plays
        if action.action_type == ActionType.PLAY_MONOPOLY:
            self._pending_monopoly_resource = action.value
        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            self._pending_yop_resources = action.value
        elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
            self.translator.state.is_road_building = True
            self.translator.state.free_roads_available = 2

        if self.action_translator:
            try:
                commands = self.action_translator.translate(action)
                for cmd in commands:
                    # Inject our color into trade commands so userscript uses correct creator
                    if cmd.get("action") == 11 and isinstance(cmd.get("data"), dict):
                        cmd["data"]["creator"] = self.translator.state.my_color_index
                    self._send(cmd)
            except ValueError as e:
                logger.error(f"Action translation error: {e}")
                self._send({"action": 5})
            except Exception as e:
                logger.error(f"Unexpected translation error: {e}")
                self._send({"action": 5})
        else:
            self._send({"action": 5})

    async def _handle_road_building_placement(self) -> None:
        """Place a road during Road Building dev card play.

        Uses the dev card road protocol: SELECT(edge) → SELECT(null) → CONFIRM_BUILD_DEV(edge)
        (internal action 13), which differs from normal road build flow.
        """
        logger.info(
            f"Road building placement: free_roads={self.translator.state.free_roads_available}, "
            f"available_roads={len(self._available_roads)}"
        )
        game = self.translator.reconstruct_game()
        if not game:
            logger.warning("Road building placement: reconstruct_game returned None")
            return

        my_color = self.translator.state.my_catan_color
        action = self.bot.decide(game, my_color)
        logger.info(f"Road building bot decision: {action}")

        if action is None or action.action_type != ActionType.BUILD_ROAD:
            # Fallback: pick first available road from colonist.io positions
            if self._available_roads:
                edge_index = self._available_roads[0]
                logger.info(f"Road building fallback: picking edge {edge_index}")
                self._send({"action": ACTION_BUILD_ROAD_DEV, "data": edge_index})
            else:
                logger.warning("Road building: no available roads, ending road building")
                self.translator.state.is_road_building = False
                self.translator.state.free_roads_available = 0
            return

        if not self.action_translator:
            logger.warning("Road building placement: no action_translator")
            return

        try:
            edge = action.value
            col_coord = self.action_translator.mapper.catan_edge_to_colonist.get(frozenset(edge))
            if col_coord is None:
                col_coord = self.action_translator.mapper.catan_edge_to_colonist.get(
                    frozenset((edge[1], edge[0]))
                )
            if col_coord is None:
                raise ValueError(f"Unknown edge for road building: {edge}")
            edge_index = self.action_translator._find_edge_index(col_coord)
            logger.info(f"Road building: placing road at edge_index={edge_index} (catan edge={edge})")
            self._send({"action": ACTION_BUILD_ROAD_DEV, "data": edge_index})
        except Exception as e:
            logger.error(f"Failed to translate road building placement: {e}")
            # Fallback: use available roads from colonist.io
            if self._available_roads:
                logger.info(f"Road building fallback after error: edge_index={self._available_roads[0]}")
                self._send({"action": ACTION_BUILD_ROAD_DEV, "data": self._available_roads[0]})

    async def _handle_move_robber(self) -> None:
        """Handle robber movement (actionState=24).

        Uses depth-2 search (not depth-3) to keep decision time <2s.
        Depth 3 after robber is too expensive because the PLAY_TURN branching
        factor explodes: 30+ robber tiles × 20+ post-robber actions × 20+ opponent.
        Depth 2 captures blocking value + one round of play.
        """
        game = self.translator.reconstruct_game()
        if game is None:
            return
        my_color = self.translator.state.my_catan_color

        robber_actions = [
            a for a in game.playable_actions
            if a.action_type == ActionType.MOVE_ROBBER
        ]
        if not robber_actions:
            return

        # Use shallower search for robber — depth 3 takes 20s+
        original_depth = getattr(self.bot._get_player(my_color), 'depth', None)
        player = self.bot._get_player(my_color)
        if original_depth is not None and original_depth > 2:
            player.depth = 2

        action = self.bot.decide(game, my_color)

        # Restore original depth
        if original_depth is not None:
            player.depth = original_depth

        if action is None or action.action_type != ActionType.MOVE_ROBBER:
            action = robber_actions[0]

        if self.action_translator:
            try:
                commands = self.action_translator.translate(action)
                if commands:
                    self._send(commands[0])  # Send move_robber
                    # Store steal color for when rob victim selection arrives
                    if len(commands) > 1:
                        self._pending_robber_steal_color = action.value[1]
            except Exception as e:
                logger.error(f"Failed to translate move robber: {e}")

    async def _handle_rob_selection_auto(self) -> None:
        """Handle rob player selection (actionState=27).

        Uses pending steal color from move_robber decision if available.
        """
        if self._pending_robber_steal_color is not None:
            steal_color = self._pending_robber_steal_color
            self._pending_robber_steal_color = None

            colonist_color = self.translator.state.catanatron_to_colonist_color.get(steal_color)
            if colonist_color is not None:
                self._send({"action": 9, "data": colonist_color})
                return

        # Fallback: pick first opponent (not ourselves)
        my_col = self.translator.state.my_color_index
        for col_idx in self.translator.state.player_order:
            if col_idx != my_col:
                self._send({"action": 9, "data": col_idx})
                return

    async def _handle_rob_selection(self, players_to_select: List[int]) -> None:
        """Handle rob player selection (type 29 prompt)."""
        if not players_to_select:
            return

        # Use pending steal color from move_robber decision
        if self._pending_robber_steal_color is not None:
            steal_color = self._pending_robber_steal_color
            self._pending_robber_steal_color = None

            colonist_color = self.translator.state.catanatron_to_colonist_color.get(steal_color)
            if colonist_color is not None and colonist_color in players_to_select:
                self._send({"action": 9, "data": colonist_color})
                return

        # Fallback: pick first allowed player
        self._send({"action": 9, "data": players_to_select[0]})

    async def _handle_discard(self, amount: int) -> None:
        """Handle discard prompt (type 13)."""
        resources_to_discard = self.bot.decide_discard(
            self.translator.state.my_resources, amount
        )
        if self.action_translator:
            cmd = self.action_translator.translate_discard(resources_to_discard)
            self._send(cmd)
        else:
            from bridge.config import CATAN_RESOURCE_TO_COLONIST
            cards = []
            for resource, count in self.translator.state.my_resources.items():
                if len(cards) >= amount:
                    break
                for _ in range(min(count, amount - len(cards))):
                    cards.append(CATAN_RESOURCE_TO_COLONIST.get(resource, 1))
            self._send({"action": 10, "data": cards})

    async def _handle_trade_offer_from_diff(self, trade_id: str, offer_data: Dict) -> None:
        """Handle a trade offer received via type 91 diff."""
        creator = offer_data.get("creator", -1)
        offered = offer_data.get("offeredResources", [])
        wanted = offer_data.get("wantedResources", [])

        # Reconstruct game for smart trade evaluation
        game = self.translator.reconstruct_game()
        our_color = self.translator.state.my_catan_color
        creator_vps = 0
        vps_to_win = 10
        creator_catan = None
        if game is not None:
            vps_to_win = game.vps_to_win
            creator_catan = self.translator.state.colonist_to_catan_color.get(creator)
            if creator_catan is not None and creator_catan in game.state.color_to_index:
                c_idx = game.state.color_to_index[creator_catan]
                creator_vps = game.state.player_state.get(
                    f"P{c_idx}_ACTUAL_VICTORY_POINTS", 0)

        accept = self.bot.decide_trade_response(
            offered, wanted,
            self.translator.state.my_resources,
            trade_id,
            game=game,
            our_color=our_color,
            creator_vps=creator_vps,
            vps_to_win=vps_to_win,
            creator_color=creator_catan,
        )

        if accept:
            self._send({"action": 6, "data": {"tradeId": trade_id, "creator": creator}})
        else:
            self._send({"action": 7, "data": {"tradeId": trade_id}})

    def _handle_available_actions(self, msg: AvailableActionsMsg) -> None:
        """Store available action positions for decision-making."""
        if msg.action_type == MSG_TYPE_AVAILABLE_SETTLEMENTS:
            self._available_settlements = msg.indices
        elif msg.action_type == MSG_TYPE_AVAILABLE_ROADS:
            self._available_roads = msg.indices
        elif msg.action_type == MSG_TYPE_AVAILABLE_CITIES:
            self._available_cities = msg.indices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _RESOURCE_LETTER = {"WOOD": "w", "BRICK": "b", "SHEEP": "s", "WHEAT": "g", "ORE": "o"}

    def _get_opponent_vps(self, game) -> Dict[int, int]:
        """Get VP counts for all opponents, keyed by colonist color index."""
        result = {}
        my_col = self.translator.state.my_color_index
        c2c = self.translator.state.colonist_to_catan_color
        for col_idx, catan_color in c2c.items():
            if col_idx == my_col:
                continue
            if catan_color in game.state.color_to_index:
                p_idx = game.state.color_to_index[catan_color]
                result[col_idx] = game.state.player_state.get(
                    f"P{p_idx}_ACTUAL_VICTORY_POINTS", 0)
        return result

    def _store_tile_data(self, tile_hex_states: Dict) -> None:
        """Store tile data by index and coord for action descriptions."""
        for idx_str, data in tile_hex_states.items():
            idx = int(idx_str)
            resource = COLONIST_TILE_TO_RESOURCE.get(data.get("type", 0))
            dice_num = data.get("diceNumber", 0) or 0
            x, y = data.get("x", 0), data.get("y", 0)
            tile = {"resource": resource, "dice_num": dice_num}
            self._tile_by_index[idx] = tile
            self._tile_by_coord[(x, y)] = tile

    def _vertex_desc(self, vertex_idx: int) -> str:
        """Compact tile description for a vertex, e.g. '3b 6o 2w'."""
        coord = self._vertex_index_to_coord.get(vertex_idx)
        if not coord:
            return ""
        x, y, z = coord
        if z == 0:
            candidates = [(x, y), (x, y - 1), (x + 1, y - 1)]
        else:
            candidates = [(x, y), (x, y + 1), (x - 1, y + 1)]
        parts = []
        for cx, cy in candidates:
            tile = self._tile_by_coord.get((cx, cy))
            if tile and tile["resource"]:
                letter = self._RESOURCE_LETTER.get(tile["resource"], "?")
                parts.append(f"{tile['dice_num']}{letter}")
        return " ".join(parts)

    def _tile_desc(self, tile_idx: int) -> str:
        """Compact tile description, e.g. 'Brick:3'."""
        tile = self._tile_by_index.get(tile_idx)
        if not tile:
            return f"tile {tile_idx}"
        if not tile["resource"]:
            return "Desert"
        return f"{tile['resource']}:{tile['dice_num']}"

    def _resources_desc(self, resource_ids: list) -> str:
        """Compact resource list, e.g. '2w b'."""
        from collections import Counter
        counts = Counter()
        for rid in resource_ids:
            name = COLONIST_RESOURCE_TO_CATAN.get(rid)
            if name:
                counts[self._RESOURCE_LETTER.get(name, "?")] += 1
        return " ".join(
            f"{n}{l}" if n > 1 else l for l, n in counts.items()
        )

    def _describe_command(self, data: dict) -> str:
        """Generate human-readable description of a bot command."""
        action = data.get("action")
        payload = data.get("data")
        setup = "Setup " if data.get("setup") else ""

        if action == 0:
            return f"{setup}Build road at e{payload}"
        elif action == 1:
            tiles = self._vertex_desc(payload) if isinstance(payload, int) else ""
            return f"{setup}Build settlement at v{payload} ({tiles})" if tiles else f"{setup}Build settlement at v{payload}"
        elif action == 2:
            tiles = self._vertex_desc(payload) if isinstance(payload, int) else ""
            return f"Upgrade city at v{payload} ({tiles})" if tiles else f"Upgrade city at v{payload}"
        elif action == 3:
            return "Buy development card"
        elif action == 4:
            return "Roll dice"
        elif action == 5:
            return "End turn"
        elif action == 6:
            tid = payload.get("tradeId", "?") if isinstance(payload, dict) else payload
            return f"Accept trade {tid}"
        elif action == 7:
            tid = payload.get("tradeId", "?") if isinstance(payload, dict) else payload
            return f"Reject trade {tid}"
        elif action == 8:
            return f"Move robber to {self._tile_desc(payload)}" if isinstance(payload, int) else f"Move robber"
        elif action == 9:
            return f"Rob player P{payload}"
        elif action == 10:
            return f"Discard: {self._resources_desc(payload)}" if isinstance(payload, list) else "Discard"
        elif action == 11:
            if isinstance(payload, dict):
                offered = self._resources_desc(payload.get("offered", []))
                wanted = self._resources_desc(payload.get("wanted", []))
                bank = " (bank)" if payload.get("bankTrade") else ""
                return f"Trade{bank}: give [{offered}] for [{wanted}]"
            return "Create trade"
        elif action == 12:
            card = COLONIST_VALUE_TO_DEVCARD.get(payload, f"card#{payload}") if isinstance(payload, int) else str(payload)
            return f"Play {card}"
        elif action == 13:
            return f"Road Building: place road at e{payload}"
        elif action == 14:
            if isinstance(payload, dict):
                return f"Execute trade {payload.get('tradeId', '?')} with P{payload.get('playerColor', '?')}"
            return "Execute trade"
        return ""

    async def _safe_execute(self, handler) -> None:
        """Wrap handler in timeout with fallback action.

        Anticheat delay acts as a minimum wait time: if the handler takes
        longer than the delay, no extra sleep is added.
        """
        start = time.monotonic()
        try:
            result = handler()
            if inspect.iscoroutine(result):
                await asyncio.wait_for(result, timeout=DECISION_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error(
                f"Handler timed out after {DECISION_TIMEOUT_SECONDS}s. Sending END_TURN fallback."
            )
            self._send({"action": 5})
        except Exception as e:
            logger.error(f"Handler failed: {e}", exc_info=True)
            self._send({"action": 5})

        if self._anticheat:
            elapsed = time.monotonic() - start
            min_delay = random.uniform(self._ac_min_think, self._ac_max_think)
            remaining = min_delay - elapsed
            if remaining > 0:
                logger.debug(f"Anticheat: waited {remaining:.1f}s (think took {elapsed:.1f}s)")
                await asyncio.sleep(remaining)

    def _send(self, data: dict) -> None:
        """Queue a message for sending to the userscript."""
        desc = self._describe_command(data)
        if desc:
            data["desc"] = desc
            logger.info(f">> {desc}")
        try:
            data_json = json.dumps(data)
            self.queue.put_nowait(data_json)
            self.game_logger.log_outgoing("action", data)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to serialize outgoing message: {e}")
        self._last_action_sent_time = time.monotonic()

    def _refresh_action_translator(self) -> None:
        """Re-initialize ActionTranslator when mapper and colors are available."""
        state = self.translator.state
        if state.mapper is None:
            return
        if not state.catanatron_to_colonist_color:
            return

        self.action_translator = ActionTranslator(
            state.mapper,
            state.catanatron_to_colonist_color,
        )
        # Set index maps for vertex/edge lookups
        self.action_translator.set_vertex_index_map(self._vertex_coord_to_idx)
        self.action_translator.set_edge_index_map(self._edge_coord_to_idx)


async def main(bot_type: str = "value", anticheat: bool = None,
               search_depth: int = 2, blend_weight: float = 1e8,
               bc_model_path: str = "robottler/models/value_net_v2.pt",
               trade_strategy: str = "blend",
               propose_trades: bool = False):
    """Start the bridge server."""
    server = BridgeServer(
        bot_type=bot_type, anticheat=anticheat,
        search_depth=search_depth, blend_weight=blend_weight,
        bc_model_path=bc_model_path, trade_strategy=trade_strategy,
        propose_trades=propose_trades,
    )
    async with websockets.serve(server.handle_connection, WS_HOST, WS_PORT):
        logger.info(f"Bridge server running on ws://{WS_HOST}:{WS_PORT} (bot={bot_type})")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="AICatan bridge server")
    parser.add_argument("--no-anticheat", action="store_true", help="Disable anti-cheat delays")
    parser.add_argument("--bot", type=str, default="search",
                        choices=["search", "value", "neural", "random"],
                        help="Bot type (default: search)")
    parser.add_argument("--search-depth", type=int, default=3,
                        help="Search depth for search bot (default: 3)")
    parser.add_argument("--blend-weight", type=float, default=1e8,
                        help="Blend weight for search bot (default: 1e8)")
    parser.add_argument("--bc-model", type=str, default="robottler/models/value_net_v2.pt",
                        help="BC model path for search bot")
    parser.add_argument("--trade-strategy", type=str, default="blend",
                        choices=["blend", "heuristic"],
                        help="Trade evaluation strategy (default: blend)")
    parser.add_argument("--propose-trades", action="store_true",
                        help="Enable bot trade proposals (default: off)")
    args = parser.parse_args()
    anticheat = False if args.no_anticheat else None
    asyncio.run(main(
        bot_type=args.bot, anticheat=anticheat,
        search_depth=args.search_depth, blend_weight=args.blend_weight,
        bc_model_path=args.bc_model, trade_strategy=args.trade_strategy,
        propose_trades=args.propose_trades,
    ))
