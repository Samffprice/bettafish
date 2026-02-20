""" asdasd
"""
import json
from abc import ABC, abstractmethod

from dev_cards import DevCards

class Bot(ABC):
    board = None
    own_color = None
    _queue = None

    def __init__(self, board, own_color, queue):
        self.board = board
        self.own_color = own_color
        self._queue = queue

    @abstractmethod
    def build_setup_settlement(self):
        pass

    @abstractmethod
    def build_setup_road(self):
        pass

    @abstractmethod
    async def start_turn(self):
        pass

    @abstractmethod
    async def play_turn(self):
        pass

    @abstractmethod
    def move_robber(self):
        pass

    @abstractmethod
    def rob(self, options):
        pass

    @abstractmethod
    def discard_cards(self):
        pass

    @abstractmethod
    def respond_to_trade(self):
        pass

    @abstractmethod
    async def use_road_building(self):
        """ Called when bot has played a road building card and now has to make use of it.
        Needs to call send_build_road two times.
        """

    @abstractmethod
    async def use_monopoly(self):
        """ Called when bot has played a monopoly card and now has to make use of it.
        Needs to call send_select_cards with the chosen resource, where argument selection
        is a list with one element: the resource value.
        """

    @abstractmethod
    async def use_year_of_plenty(self):
        """ Called when bot has played a year of plenty card and now has to make use of it.
        Needs to call send_select_cards with the chosen resource, where argument selection
        is a list with two elements: the resources values.
        """

    @abstractmethod
    def handle_event(self, event_type, **data):
        pass

    def send_build_road(self, edge_index):
        self.send({"action": 0, "data": edge_index})

    def send_build_settlement(self, vertex_index):
        self.send({"action": 1, "data": vertex_index})

    def send_build_city(self, vertex_index):
        self.send({"action": 2, "data": vertex_index})

    def send_buy_dev_card(self):
        self.send({"action": 3})

    def send_throw_dice(self):
        self.send({"action": 4})

    def send_pass_turn(self):
        self.send({"action": 5})

    def send_accept_trade(self, trade_id):
        self.send({"action": 6, "data": trade_id})

    def send_reject_trade(self, trade_id):
        self.send({"action": 7, "data": trade_id})

    def send_move_robber(self, tile_index):
        self.send({"action": 8, "data": tile_index})

    def send_rob(self, player):
        self.send({"action": 9, "data": player})

    def send_select_cards(self, selection):
        self.send({"action": 10, "data": selection})

    def send_create_trade(self, offered, wanted):
        data = {"offered": offered, "wanted": wanted}
        self.send({"action": 11, "data": data})

    def send_play_dev_card(self, card_type):
        self.send({"action": 12, "data": DevCards(card_type).value})

    def send(self, data):
        data_in_json = json.dumps(data)
        self._queue.put_nowait(data_in_json)
