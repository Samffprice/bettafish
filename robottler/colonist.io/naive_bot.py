import asyncio
import copy
from collections import Counter

from abstract_bot import Bot
from col_events import ColEvents
from costs import COSTS
from dev_cards import DevCards
from purchase_types import PurchaseType
from resources import Resources

class NaiveBot(Bot):
    next_purchase = None
    events = []

    def __init__(self, board, own_color, queue):
        super().__init__(board, own_color, queue)

    # overriding abstract method
    def build_setup_settlement(self):
        print("Building settlement")
        settlement_index = self.find_setup_settlement_vertex()
        self.send_build_settlement(settlement_index)

    # overriding abstract method
    def build_setup_road(self):
        print("Building road")
        _, road_index = self.find_next_settlement(self.board.own_settlements[-1], True)

        if road_index is None:
            road_index = self.board.adjacency_map[self.board.own_settlements[-1]][0]["edge_index"]
        self.send_build_road(road_index)

    # overriding abstract method
    async def start_turn(self):
        self.send_throw_dice()

    # overriding abstract method
    async def play_turn(self):
        if self.board.own_dev_cards[DevCards.ROBBER] > 0:
            print("Playing robber")
            self.send_play_dev_card(DevCards.ROBBER)
            await self.wait_event(ColEvents.ROBBER_MOVED)
            await self.wait_event(ColEvents.RECEIVED_CARDS)
            self.board.own_dev_cards[DevCards.ROBBER] -= 1
        elif self.board.own_dev_cards[DevCards.ROAD_BUILDING] > 0:
            print("Playing road building")
            self.send_play_dev_card(DevCards.ROAD_BUILDING)
            await self.wait_event(ColEvents.EDGE_CHANGED)
            await self.wait_event(ColEvents.EDGE_CHANGED)
            self.board.own_dev_cards[DevCards.ROAD_BUILDING] -= 1
        elif self.board.own_dev_cards[DevCards.MONOPOLY] > 0:
            print("Playing monopoly")
            self.send_play_dev_card(DevCards.MONOPOLY)
            await self.wait_event(ColEvents.RECEIVED_CARDS)
            self.board.own_dev_cards[DevCards.MONOPOLY] -= 1
        elif self.board.own_dev_cards[DevCards.YEAR_OF_PLENTY] > 0:
            print("Playing year of plenty")
            self.send_play_dev_card(DevCards.YEAR_OF_PLENTY)
            await self.wait_event(ColEvents.RECEIVED_CARDS)
            self.board.own_dev_cards[DevCards.YEAR_OF_PLENTY] -= 1

        print("Starting turn")
        self.next_purchase = self.calc_next_purchase()
        print("next_purchase: {0}".format(self.next_purchase))

        if self.distance_from_cards(COSTS[self.next_purchase], self.board.resources, False) > 0:
            await self.trade_with_bank()
        if self.distance_from_cards(COSTS[self.next_purchase], self.board.resources, False) == 0:
            settlement_index, road_index = self.find_next_settlement()
            if self.next_purchase == PurchaseType.ROAD:
                self.send_build_road(road_index)
            if self.next_purchase == PurchaseType.SETTLEMENT:
                self.send_build_settlement(settlement_index)
            if self.next_purchase == PurchaseType.CITY:
                self.send_build_city(self.board.own_settlements[0])
            if self.next_purchase == PurchaseType.DEV_CARD:
                self.send_buy_dev_card()

        self.send_pass_turn()

    # overriding abstract method
    def move_robber(self):
        print("Moving robber")
        new_robber_tile_index = self.find_new_robber_tile()
        print("New robber tile index : {0}".format(new_robber_tile_index))
        self.send_move_robber(new_robber_tile_index)
    
    # overriding abstract method
    def rob(self, options):
        print("Robbing")
        self.send_rob(options[0])

    # overriding abstract method
    def discard_cards(self, amount):
        print("Discarding cards")
        cards = self.pick_cards_to_discard(amount)
        self.send_select_cards(cards)

    # TODO: Rewrite to decouple from the colonist.io API
    # overriding abstract method
    def respond_to_trade(self, data):
        print("Responding to offer")
        # Respond to trade offer
        self.next_purchase = self.calc_next_purchase()

        if self.is_favourable_trade(data):
            self.send_accept_trade(data.id)
        else:
            self.send_reject_trade(data.id)

    # overriding abstract method
    async def use_road_building(self):
        print("use_road_building()")
        _, edge_index = self.find_next_settlement()
        self.send_build_road(edge_index)
        await self.wait_event(ColEvents.EDGE_CHANGED)
        _, edge_index = self.find_next_settlement()
        self.send_build_road(edge_index)

    # overriding abstract method
    async def use_monopoly(self):
        print("use_monopoly()")
        selected_resource = max(self.board.resources, key=self.board.resources.get)
        print(selected_resource)
        self.send_select_cards([selected_resource])

    # overriding abstract method
    async def use_year_of_plenty(self):
        print("use_year_of_plenty()")
        needed = COSTS[self.calc_next_purchase()]
        print(needed)
        _, missing = self.calc_missing_cards(needed, self.board.resources)
        print(missing)
        missing_list = [resource.value for resource in missing.keys() for x in range(missing[resource])]
        print(missing_list)
        missing_list = missing_list[:2]
        while len(missing_list) < 2:
            missing_list.append(1)
        print(missing_list)
        self.send_select_cards(missing_list)

    # overriding abstract method
    def handle_event(self, event_type, **data):
        for event in self.events:
            if event["event_type"] == event_type:
                event["object"].set()


    def create_event(self, event_type, **data):
        event = {"event_type" : event_type, "object" : asyncio.Event()}
        event.update(data)
        self.events.append(event)
        return event
    
    async def wait_event(self, event_type, **data):
        event = self.create_event(event_type)
        print("waiting for event")
        await event["object"].wait()
        print("done waiting for event")
        self.events.remove(event)

    # TODO: Rewrite this function to use evaluate_board
    def find_next_settlement(self, settlement_index=None, is_setup=False):
        candidates = []

        if settlement_index is None:
            for index in self.board.own_settlements:
                candidates.extend(self.find_settlement_spots(index, [], 2))
        else:
            candidates.extend(self.find_settlement_spots(settlement_index, [], 2))

        if len(candidates) == 0:
            return None, None

        high_vertex_index = -1
        high_prod = -1
        high_path = []
        for entry in candidates:
            vertex_index = entry["vertex_index"]

            prod = self.board.find_vertex_production_by_index(vertex_index)
            if prod > high_prod and (not is_setup or prod <= 7):
                high_vertex_index = vertex_index
                high_prod = prod
                high_path = entry["path"]

        for edge_index in high_path:
            if self.board.edges[edge_index].owner == 0:
                return high_vertex_index, edge_index
        return high_vertex_index, None
    
    def evaluate_setup(self, production, harbor_types):
        score = 0
        total_production = sum(production.values())
        score += total_production

        for harbor_type in harbor_types:
            if harbor_type == 1:
                score *= 1.25
            else:
                score += production[Resources(harbor_type - 1)] / 1.5

        # Penalty if you have scare resources
        for resource, amount in production.items():
            if amount < 0.15 * total_production:
                penalty_multiplier = 1 - amount / (0.15 * total_production)
                score -= (0.2 * total_production) * penalty_multiplier

        return score

    def find_setup_settlement_vertex(self):
        i = high_index = -1
        high_score = -1
        for i, vertex in enumerate(self.board.vertices):
            if vertex.owner != 0 or vertex.restrictedStartingPlacement is True:
                continue
            
            prod = self.board.find_vertex_production_per_resource_by_index(i)
            for resource in Resources:
                prod[resource] += self.board.own_production[resource]

            harbor_types = set()
            harbor_types.update(self.board.own_harbors)
            if vertex.harborType != 0:
                harbor_types.add(vertex.harborType)

            score = self.evaluate_setup(prod, harbor_types)

            if score > high_score:
                high_score = score
                high_index = i

        return high_index

    def find_highest_producing_vertex(self):
        """ Loops over all vertices to find the highest producing spot
        where its still possible to build a settlement
        """

        i = high_index = -1
        high_prod = -1
        for i, vertex in enumerate(self.board.vertices):
            if vertex.owner != 0 or vertex.restrictedStartingPlacement is True:
                continue

            prod = self.board.find_vertex_production_by_index(i)
            if prod > high_prod:
                high_prod = prod
                high_index = i

        return high_index

    def find_settlement_spots(self, vertex_index, path, degree):
        if degree <= 0:
            vert = self.board.vertices[vertex_index]
            if vert.owner == 0 and not vert.restrictedStartingPlacement:
                return [{"vertex_index": vertex_index, "path": path}]
            else:
                return []

        candidates = []
        for entry in self.board.adjacency_map[vertex_index]:
            vertex = self.board.vertices[entry["vertex_index"]]
            edge = self.board.edges[entry["edge_index"]]
            if (not (len(path) > 0 and entry["edge_index"] == path[-1]) and # Don't travel backwards
            (vertex.owner == 0 or vertex.owner == self.own_color) and       # Make sure we can travel there
            (edge.owner == 0 or edge.owner == self.own_color)):             # Make sure we can travel there
                new_path = copy.deepcopy(path)
                new_path.append((entry["edge_index"]))
                candidates.extend(self.find_settlement_spots(entry["vertex_index"], new_path, degree - 1))

        return candidates

    def calc_cards_after_bank_trades(self, wanted, own_cards):
        print("calc_cards_after_bank_trades")

        if self.distance_from_cards(wanted, own_cards, False) == 0:
            return own_cards

        extra_resources = copy.deepcopy(own_cards)

        for resource, amount in wanted.items():
            extra_resources[resource] -= amount

        dist, missing = self.calc_missing_cards(wanted, own_cards)

        print(list(missing)[0])

        cards_after_trade = copy.deepcopy(own_cards)

        traded = True
        while traded:
            traded = False
            for resource in Resources:
                offer_amount = self.board.bank_trades[resource]
                if extra_resources[resource] >= offer_amount:
                    cards_after_trade[resource] -= offer_amount
                    cards_after_trade[list(missing)[0]] += 1

                    print(cards_after_trade)

                    extra_resources[resource] -= offer_amount
                    missing[list(missing)[0]] -= 1

                    if missing[list(missing)[0]] <= 0:
                        del missing[list(missing)[0]]
                    if len(missing) == 0:
                        traded = False
                        break
                    traded = True
        
        print("cards_after_trade: {0}".format(cards_after_trade))

        return cards_after_trade

    def calc_missing_cards(self, needed, own_cards):
        dist = 0
        missing_cards = {}
        for resource, amount in needed.items():
            diff = max(0, amount - own_cards[resource])
            dist += diff
            if diff > 0:
                missing_cards[resource] = diff
        return dist, missing_cards

    def distance_from_cards(self, needed, own_cards, after_bank_trades=True):
        print("distance_from_cards")

        if after_bank_trades:
            own_cards = self.calc_cards_after_bank_trades(needed, own_cards)
        
        dist = 0
        for resource, amount in needed.items():
            dist += max(0, amount - own_cards[resource])
        return dist

    # TODO: Calculate multiple steps ahead while distance_from_cards == 0
    def is_favourable_trade(self, data):
        resources_after_trade = copy.deepcopy(self.board.resources)
        for card in data.offeredResources.cards:
            resources_after_trade[Resources(card)] += 1
        for card in data.wantedResources.cards:
            resources_after_trade[Resources(card)] -= 1
        needed = COSTS[self.calc_next_purchase()]
        return self.distance_from_cards(needed, resources_after_trade) < self.distance_from_cards(needed, self.board.resources)


    def find_new_robber_tile(self):
        high_index = -1
        high_block = -1

        for i, tile in enumerate(self.board.tiles):
            if self.board.robber_tile == i:
                continue
            if tile.tileType == 0:
                continue  # desert tile doesn't have the _diceProbability attribute

            block = 0
            for vertex_index in self.board.tile_vertices[i]:
                vertex = self.board.vertices[vertex_index]

                if vertex.owner == self.own_color:
                    break
                if vertex.owner != 0:
                    block += tile._diceProbability * vertex.buildingType
            else:  # only runs if execution isnt broken during loop
                if block > high_block:
                    high_block = block
                    high_index = i

        return high_index

    # TODO: Trade for lowest producing resource
    async def trade_with_bank(self):
        print("trade_with_bank()")

        if self.distance_from_cards(COSTS[self.next_purchase], self.board.resources, False) == 0:
            return

        extra_resources = copy.deepcopy(self.board.resources)
        print(COSTS[self.next_purchase])
        for resource, amount in COSTS[self.next_purchase].items():
            extra_resources[resource] -= amount

        dist, missing = self.calc_missing_cards(COSTS[self.next_purchase], self.board.resources)

        print(list(missing)[0].value)
        
        traded = True
        while traded:
            traded = False
            for resource in Resources:
                offer_amount = self.board.bank_trades[resource]
                if extra_resources[resource] >= offer_amount:
                    print("Starting bank trade")

                    offered = [resource.value for x in range(offer_amount)]
                    wanted = [list(missing)[0].value]
                    self.send_create_trade(offered, wanted)

                    await self.wait_event(ColEvents.RECEIVED_CARDS)

                    extra_resources[resource] -= offer_amount
                    missing[list(missing)[0]] -= 1

                    if missing[list(missing)[0]] <= 0:
                        del missing[list(missing)[0]]
                    if len(missing) == 0:
                        return
                    traded = True

    # TODO: Add check if we still have a road
    # TODO: Differentiate between functional roads and non-functional roads
    # TODO: Check for non-functional roads as well
    def can_build_road(self, board=None, cards=True):
        """ Checks if it is possible to build a road.

        Keyword arguments:
        board -- the board
        cards -- include check if bot has enough cards to build it directly
        """
        board = board or self.board
        if self.find_next_settlement()[0] is not None: return False
        if cards:
            if self.distance_from_cards(COSTS[PurchaseType.SETTLEMENT], board.resources) > 0:
                return False
        return True

    def can_build_settlement(self, board=None, cards=True):
        """ Checks if it is possible to build a settlement.

        Keyword arguments:
        board -- the board
        cards -- include check if bot has enough cards to build it directly
        """
        print("can_build_settlement()")

        board = board or self.board
        if self.find_next_settlement()[1] is not None: return False
        if board.own_settlements == 5: return False
        if cards:
            if self.distance_from_cards(COSTS[PurchaseType.SETTLEMENT], board.resources) > 0:
                return False
        return True

    def can_build_city(self, board=None, cards=True):
        """ Checks if it is possible to build a city.

        Keyword arguments:
        board -- the board
        cards -- include check if bot has enough cards to build it directly
        """
        print("can_build_city()")

        board = board or self.board

        print(board)

        if len(board.own_settlements) == 0: return False
        if board.own_cities == 4: return False
        if cards:
            if self.distance_from_cards(COSTS[PurchaseType.CITY], board.resources) > 0:
                return False
        return True

    def can_buy_dev_card(self, board=None, cards=True):
        """ Checks if it is possible to buy a development card.

        Keyword arguments:
        board -- the board
        cards -- include check if bot has enough cards to buy it directly
        """
        board = board or self.board
        if board.bank_dev_cards == 0: return False
        if cards:
            if self.distance_from_cards(COSTS[PurchaseType.DEV_CARD], board.resources) > 0:
                return False
        return True

    def calc_next_purchase(self, board=None):
        board = board or self.board
        print("calc_next_purchase()")

        # First check if we can build a city or settlement
        if self.can_build_city():
            return PurchaseType.CITY
        if self.can_build_settlement():
            return PurchaseType.SETTLEMENT

        dist_from_city = self.distance_from_cards(COSTS[PurchaseType.CITY], board.resources)

        # High change of having to discard cards
        if sum(board.resources.values()) >= 6:
            if dist_from_city < 1 and self.can_build_city(cards=False):
                cards_after_dev_card = dict(Counter(board.resources) - Counter(COSTS[PurchaseType.DEV_CARD]))
                if self.can_buy_dev_card() and \
                   self.distance_from_cards(PurchaseType.DEV_CARD, cards_after_dev_card) == \
                   self.distance_from_cards(PurchaseType.DEV_CARD, board.resources):
                    return PurchaseType.DEV_CARD

                cards_after_road = dict(Counter(board.resources) - Counter(COSTS[PurchaseType.ROAD]))
                if self.can_build_road() and \
                   self.distance_from_cards(PurchaseType.DEV_CARD, cards_after_road) == \
                   self.distance_from_cards(PurchaseType.DEV_CARD, board.resources):
                    return PurchaseType.ROAD
                return PurchaseType.CITY

            if self.can_buy_dev_card():
                return PurchaseType.DEV_CARD
            if self.can_build_road():
                return PurchaseType.ROAD
            
        # Check if we should expand
        if dist_from_city < 1:
            return PurchaseType.CITY
        if len(self.board.own_settlements) + len(self.board.own_cities) < 4:
            print("settlement or road")
            if self.find_next_settlement()[1] is None:
                print("settlement")
                return PurchaseType.SETTLEMENT
            else:
                print("road")
                return PurchaseType.ROAD

        # Buy a dev card
        print("dev card")
        return PurchaseType.DEV_CARD


    def pick_cards_to_discard(self, amount):
        next_cards = COSTS[self.calc_next_purchase()]

        non_discarded_cards = copy.deepcopy(self.board.resources)
        discarded_cards = []
        extra_cards = True
        # 1. round robin cards not necessary for next buy
        # 2. round robin cards necessary for next buy
        while len(discarded_cards) < amount:
            extra_card_found = False
            for resource in Resources:
                if extra_cards:
                    if (resource not in next_cards or \
                        non_discarded_cards[resource] > next_cards[resource]) and \
                    non_discarded_cards[resource] > 0:
                        discarded_cards.append(resource.value)
                        non_discarded_cards[resource] -= 1
                        extra_card_found = True
                else:
                    if non_discarded_cards[resource] > 0:
                        discarded_cards.append(resource.value)
                        non_discarded_cards[resource] -= 1

                if len(discarded_cards) == amount:
                    break
            if extra_card_found is False:
                extra_cards = False
        return discarded_cards
