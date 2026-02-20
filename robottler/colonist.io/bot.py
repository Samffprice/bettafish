from enum import Enum
from types import SimpleNamespace
import asyncio
import copy
import json
import websockets

from board import Board
from col_events import ColEvents
from dev_cards import DevCards
from resources import Resources
from naive_bot import NaiveBot

PLAYER_COLOR = None
BOARD = None
QUEUE = None

class GameState(Enum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    START_TURN = 2
    PLAYER_TURN = 3
    OPPONENT_TURN = 4
GAME_STATE = GameState.SETUP_SETTLEMENT

BOT = None

async def consumer_handler(websocket, _path):
    async for message in websocket:
        try:
            data = json.loads(
                message, object_hook=lambda d: SimpleNamespace(**d))
            if hasattr(data, "myColor"):
                global PLAYER_COLOR
                PLAYER_COLOR = data.myColor
            elif hasattr(data, "tileState"):  # Board information
                global BOARD
                BOARD = Board(data)
                BOT = NaiveBot(BOARD, PLAYER_COLOR, QUEUE)
            elif hasattr(data, "currentTurnState"):  # Game state information
                global GAME_STATE

                if data.currentTurnState == 1 and data.currentTurnPlayerColor != PLAYER_COLOR:
                    GAME_STATE = GameState.OPPONENT_TURN

                if data.currentTurnPlayerColor == PLAYER_COLOR:
                    # need to be outside the currentTurnState ifs,
                    # because it can happen in both turnstate 1 and 2
                    if data.currentActionState == 23:
                        BOT.move_robber()
                    elif data.currentActionState == 27:
                        asyncio.create_task(BOT.use_road_building())
                    # elif data.currentActionState == 28:
                    #     BOT.handle_event(ColEvents.ROAD_BUILDING)
                    elif data.currentActionState == 29:
                        asyncio.create_task(BOT.use_year_of_plenty())
                    elif data.currentActionState == 30:
                        asyncio.create_task(BOT.use_monopoly())

                    if data.currentTurnState == 0:
                        if data.currentActionState == 1 and \
                           GAME_STATE == GameState.SETUP_SETTLEMENT:
                            BOT.build_setup_settlement()
                            GAME_STATE = GameState.SETUP_ROAD
                        if data.currentActionState == 3 and GAME_STATE == GameState.SETUP_ROAD:
                            BOT.build_setup_road()
                            if len(BOARD.own_settlements) < 2:
                                GAME_STATE = GameState.SETUP_SETTLEMENT
                            else:
                                GAME_STATE = GameState.OPPONENT_TURN
                    if data.currentTurnState == 1:
                        if data.currentActionState == 0 and GAME_STATE == GameState.OPPONENT_TURN:
                            GAME_STATE = GameState.START_TURN
                            asyncio.create_task(BOT.start_turn())
                    if data.currentTurnState == 2:
                        if GAME_STATE == GameState.START_TURN:
                            GAME_STATE = GameState.PLAYER_TURN
                            asyncio.create_task(BOT.play_turn())
            # TODO: remember the player next to tile we place robber on
            # so we can do the logic on the turn logic package
            elif hasattr(data, "allowableActionState"):
                print("hasattr(data, \"allowableActionState\"")
                if data.allowableActionState == 24:
                    print("data.allowableActionState == 24")
                    BOT.rob(data.playersToSelect)
            elif hasattr(data, "selectCardFormat"):
                amount = data.selectCardFormat.amountOfCardsToSelect
                BOT.discard_cards(amount)
            elif hasattr(data, "givingPlayer"):  # Trade information
                if data.givingPlayer == PLAYER_COLOR:
                    for card in data.givingCards:
                        BOARD.resources[Resources(card)] -= 1
                    for card in data.receivingCards:
                        BOARD.resources[Resources(card)] += 1
                if data.receivingPlayer == PLAYER_COLOR:
                    for card in data.givingCards:
                        if card >= 1 and card <= 5:
                            BOARD.resources[Resources(card)] += 1
                        else:
                            BOARD.own_dev_cards[DevCards(card)] += 1
                    for card in data.receivingCards:
                        BOARD.resources[Resources(card)] -= 1
                BOT.handle_event(ColEvents.RECEIVED_CARDS)
            elif hasattr(data, "offeredResources"):  # Active trade offer
                # Check if we're allowed to take this trade and have to respond
                for player_actions in data.actions:
                    if player_actions.player == PLAYER_COLOR:
                        if len(player_actions.allowedTradeActions) > 1:
                            # Respond to trade offer
                            BOT.respond_to_trade(data)
                        break
            elif hasattr(data, "resourceCards"):
                for resource in Resources:
                    BOARD.bank_resources[resource] = data["resourceCards"].count(resource.value)
            # Settlement update (probably upgrading to a city works the same)
            elif isinstance(data, list) and hasattr(data[0], "hexCorner"):
                update_vertex(data[0])
                BOT.handle_event(ColEvents.VERTEX_CHANGED)
            elif isinstance(data, list) and hasattr(data[0], "hexEdge"):
                update_edge(data[0])
                BOT.handle_event(ColEvents.EDGE_CHANGED)
            # Cards being handed out
            elif isinstance(data, list) and hasattr(data[0], "owner"):
                for entry in data:
                    if entry.owner == PLAYER_COLOR:
                        # TODO: fix for cities, probably can just increment with distribitionType
                        BOARD.resources[Resources(entry.card)] += 1
            # Robber move information
            elif isinstance(data, list) and hasattr(data[0], "tilePieceTypes"):
                new_robber_tile = data[1]
                loc = new_robber_tile.hexFace
                BOARD.robber_tile = BOARD.find_tile_index_by_coordinates(
                    loc.x, loc.y)
                print("New robber_tile: {0}".format(BOARD.robber_tile))
                BOT.handle_event(ColEvents.ROBBER_MOVED)
        except:
            pass


async def producer_handler(websocket, _path):
    while True:
        message = await QUEUE.get()
        await websocket.send(message)


async def handler(websocket, path):
    consumer_task = asyncio.ensure_future(
        consumer_handler(websocket, path))
    producer_task = asyncio.ensure_future(
        producer_handler(websocket, path))
    _done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


def update_vertex(new_vertex_info):
    print("update_vertex")

    loc = new_vertex_info.hexCorner
    vertex_index = BOARD.find_vertex_index_by_coordinates(loc.x, loc.y, loc.z)
    vertex = BOARD.vertices[vertex_index]

    if vertex is None:
        raise ValueError("Given coordinates don't exist on the board.")

    vertex.owner = new_vertex_info.owner
    vertex.buildingType = new_vertex_info.buildingType

    if vertex.owner == PLAYER_COLOR:
        # Update own building information
        if vertex.buildingType == 1:
            BOARD.own_settlements.append(vertex_index)
        if vertex.buildingType == 2:
            BOARD.own_settlements.remove(vertex_index)
            BOARD.own_cities.append(vertex_index)

        # Update production
        tiles = BOARD.vertex_tiles[vertex_index]
        for tile_index in tiles:
            tile = BOARD.tiles[tile_index]

            if hasattr(tile, "_diceProbability"):
                BOARD.own_production[Resources(tile.tileType)] += tile._diceProbability

        # Update bank trades
        if vertex.harborType != 0:
            BOARD.own_harbors.add(vertex.harborType)
            if vertex.harborType == 1:
                for resource in Resources:
                    if BOARD.bank_trades[resource] > 3:
                        BOARD.bank_trades[resource] = 3
            else:
                BOARD.bank_trades[Resources(vertex.harborType - 1)] = 2

    # Remove neighbour vertices from future settlement placement
    neighbours = BOARD.adjacency_map[vertex_index]
    for neighbour in neighbours:
        print("Restricting neighbour: {0}".format(neighbour["vertex_index"]))
        BOARD.vertices[neighbour["vertex_index"]].restrictedStartingPlacement = True


def update_edge(new_edge_info):
    loc = new_edge_info.hexEdge
    edge_index = BOARD.find_edge_index_by_coordinates(loc.x, loc.y, loc.z)
    BOARD.edges[edge_index].owner = new_edge_info.owner

QUEUE = asyncio.Queue()
start_server = websockets.serve(handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
