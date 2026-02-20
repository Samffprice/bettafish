"""Tests for bridge/protocol.py - colonist.io message parser."""
import json
import pytest

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


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

class TestParseHeartbeat:
    def test_parse_heartbeat(self):
        raw = json.dumps({"timestamp": 1234567890})
        result = parse_message(raw)
        assert isinstance(result, HeartbeatMsg)
        assert result.timestamp == 1234567890


class TestParseGameInit:
    def test_parse_game_init(self):
        raw = json.dumps({
            "type": 4,
            "payload": {
                "playerColor": 2,
                "playOrder": [3, 2, 1, 4],
                "gameState": {
                    "mapState": {"tileHexStates": {}},
                    "bankState": {},
                },
            },
            "sequence": 1,
        })
        result = parse_message(raw)
        assert isinstance(result, GameInitMsg)
        assert result.player_color == 2
        assert result.play_order == [3, 2, 1, 4]
        assert "mapState" in result.game_state

    def test_parse_type_4_without_game_fields_returns_unknown(self):
        """Type 4 without playerColor/playOrder/gameState is lobby info, not game init."""
        raw = json.dumps({
            "type": 4,
            "payload": {"someOtherField": True},
        })
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)


class TestParseGameStateDiff:
    def test_parse_game_state_diff(self):
        raw = json.dumps({
            "type": 91,
            "payload": {
                "diff": {
                    "currentState": {"turnState": 1, "actionState": 0},
                },
                "timeLeftInState": 30.5,
            },
            "sequence": 5,
        })
        result = parse_message(raw)
        assert isinstance(result, GameStateDiffMsg)
        assert "currentState" in result.diff
        assert result.time_left == 30.5

    def test_parse_type_91_without_diff_returns_unknown(self):
        raw = json.dumps({
            "type": 91,
            "payload": {"notADiff": True},
        })
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)


class TestParseResourceDistribution:
    def test_parse_resource_distribution(self):
        raw = json.dumps({
            "type": 28,
            "payload": [
                {"owner": 1, "tileIndex": 5, "distributionType": 0, "card": 2},
                {"owner": 3, "tileIndex": 8, "distributionType": 0, "card": 4},
            ],
            "sequence": 10,
        })
        result = parse_message(raw)
        assert isinstance(result, ResourceDistributionMsg)
        assert len(result.distributions) == 2
        assert result.distributions[0]["owner"] == 1
        assert result.distributions[0]["card"] == 2

    def test_parse_empty_resource_distribution(self):
        raw = json.dumps({"type": 28, "payload": []})
        result = parse_message(raw)
        assert isinstance(result, ResourceDistributionMsg)
        assert len(result.distributions) == 0


class TestParseTradeExecution:
    def test_parse_trade_execution(self):
        raw = json.dumps({
            "type": 43,
            "payload": {
                "givingPlayer": 1,
                "receivingPlayer": 2,
                "givingCards": [1, 1],
                "receivingCards": [3],
            },
            "sequence": 15,
        })
        result = parse_message(raw)
        assert isinstance(result, TradeExecutionMsg)
        assert result.giving_player == 1
        assert result.receiving_player == 2
        assert result.giving_cards == [1, 1]
        assert result.receiving_cards == [3]


class TestParseDiscardPrompt:
    def test_parse_discard_prompt(self):
        raw = json.dumps({
            "type": 13,
            "payload": {
                "selectCardFormat": {
                    "amountOfCardsToSelect": 4,
                    "validCardsToSelect": [1, 2, 3, 4, 5, 1, 2, 3],
                },
            },
        })
        result = parse_message(raw)
        assert isinstance(result, DiscardPromptMsg)
        assert result.amount == 4
        assert len(result.valid_cards) == 8


class TestParseRobVictimPrompt:
    def test_parse_rob_victim_prompt(self):
        raw = json.dumps({
            "type": 29,
            "payload": {
                "playersToSelect": [2, 4],
                "allowableActionState": 27,
            },
        })
        result = parse_message(raw)
        assert isinstance(result, RobVictimPromptMsg)
        assert result.players_to_select == [2, 4]


class TestParseAvailableActions:
    def test_parse_available_settlements(self):
        raw = json.dumps({"type": 30, "payload": [5, 12, 28]})
        result = parse_message(raw)
        assert isinstance(result, AvailableActionsMsg)
        assert result.action_type == 30
        assert result.indices == [5, 12, 28]

    def test_parse_available_roads(self):
        raw = json.dumps({"type": 31, "payload": [7, 14, 33]})
        result = parse_message(raw)
        assert isinstance(result, AvailableActionsMsg)
        assert result.action_type == 31
        assert result.indices == [7, 14, 33]

    def test_parse_available_cities(self):
        raw = json.dumps({"type": 32, "payload": [5, 12]})
        result = parse_message(raw)
        assert isinstance(result, AvailableActionsMsg)
        assert result.action_type == 32

    def test_parse_available_dev_card(self):
        raw = json.dumps({"type": 33, "payload": []})
        result = parse_message(raw)
        assert isinstance(result, AvailableActionsMsg)
        assert result.action_type == 33
        assert result.indices == []

    def test_parse_playable_dev_cards(self):
        raw = json.dumps({"type": 59, "payload": [7, 9]})
        result = parse_message(raw)
        assert isinstance(result, AvailableActionsMsg)
        assert result.action_type == 59
        assert result.indices == [7, 9]


class TestParseEndTurn:
    def test_parse_end_turn(self):
        raw = json.dumps({"type": 80, "payload": None})
        result = parse_message(raw)
        assert isinstance(result, EndTurnMsg)


class TestParseGameOver:
    def test_parse_game_over(self):
        raw = json.dumps({
            "type": 45,
            "payload": {
                "endGameState": {"winner": 2, "scores": [8, 10, 5, 4]},
            },
        })
        result = parse_message(raw)
        assert isinstance(result, GameOverMsg)
        assert result.end_game_state["winner"] == 2


class TestParseUnknown:
    def test_string_type_returns_unknown(self):
        """Connection messages (type='Connected') should be unknown."""
        raw = json.dumps({"type": "Connected"})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_unhandled_type_number_returns_unknown(self):
        raw = json.dumps({"type": 6, "payload": {}})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_unknown_dict_no_type(self):
        raw = json.dumps({"some_unknown_field": "value", "another": 42})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_empty_dict_returns_unknown(self):
        raw = json.dumps({})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)


# ---------------------------------------------------------------------------
# Failure path tests (security/robustness)
# ---------------------------------------------------------------------------

class TestProtocolFailurePaths:
    def test_malformed_json_returns_unknown(self):
        result = parse_message("this is not json {{{}")
        assert isinstance(result, UnknownMsg)

    def test_empty_string_returns_unknown(self):
        result = parse_message("")
        assert isinstance(result, UnknownMsg)

    def test_json_array_of_numbers_returns_unknown(self):
        result = parse_message("[1, 2, 3]")
        assert isinstance(result, UnknownMsg)

    def test_json_string_returns_unknown(self):
        result = parse_message('"just a string"')
        assert isinstance(result, UnknownMsg)

    def test_json_null_returns_unknown(self):
        result = parse_message("null")
        assert isinstance(result, UnknownMsg)

    def test_json_integer_returns_unknown(self):
        result = parse_message("42")
        assert isinstance(result, UnknownMsg)

    def test_json_boolean_returns_unknown(self):
        result = parse_message("true")
        assert isinstance(result, UnknownMsg)

    def test_missing_required_fields_returns_unknown(self):
        result = parse_message(json.dumps({"foo": "bar", "baz": 123}))
        assert isinstance(result, UnknownMsg)

    def test_very_large_message_does_not_crash(self):
        large = json.dumps({"data": "x" * 100000})
        result = parse_message(large)
        assert result is not None

    def test_type_91_non_dict_payload_returns_unknown(self):
        raw = json.dumps({"type": 91, "payload": "not_a_dict"})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_4_non_dict_payload_returns_unknown(self):
        raw = json.dumps({"type": 4, "payload": [1, 2, 3]})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_13_missing_select_card_format_returns_unknown(self):
        raw = json.dumps({"type": 13, "payload": {"wrongKey": True}})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_43_non_dict_payload_returns_unknown(self):
        raw = json.dumps({"type": 43, "payload": "not_a_dict"})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_28_non_list_payload_returns_unknown(self):
        raw = json.dumps({"type": 28, "payload": {"wrong": True}})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_45_non_dict_payload_returns_unknown(self):
        raw = json.dumps({"type": 45, "payload": "not_a_dict"})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)

    def test_type_29_non_dict_payload_returns_unknown(self):
        raw = json.dumps({"type": 29, "payload": [1, 2]})
        result = parse_message(raw)
        assert isinstance(result, UnknownMsg)
