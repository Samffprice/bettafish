"""Integration tests for the bridge server."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bridge.server import BridgeServer


class TestIntegrationBasic:
    @pytest.mark.asyncio
    async def test_server_creates(self):
        server = BridgeServer(bot_type="random")
        assert server is not None
        assert server.translator is not None
        assert server.bot is not None

    @pytest.mark.asyncio
    async def test_game_init_message_routing(self):
        """GameInitMsg should set the player color and player order."""
        server = BridgeServer(bot_type="random")
        server.game_logger.start_game = MagicMock()
        server.game_logger.log_incoming = MagicMock()
        server.game_logger.end_game = MagicMock()
        server.game_logger.log_outgoing = MagicMock()

        from bridge.protocol import GameInitMsg
        msg = GameInitMsg(
            player_color=2,
            play_order=[3, 2, 1, 4],
            game_state={
                "mapState": {
                    "tileHexStates": {},
                    "tileCornerStates": {},
                    "tileEdgeStates": {},
                },
            },
        )
        await server._handle_message(msg)

        assert server.translator.state.my_color_index == 2
        assert server.translator.state.player_order == [3, 2, 1, 4]


class TestIntegrationFailurePaths:
    @pytest.mark.asyncio
    async def test_websocket_disconnect_during_consumer(self):
        """Simulate WebSocket disconnect: consumer exits cleanly,
        handle_connection cleans up all tasks."""
        import websockets.exceptions

        server = BridgeServer(bot_type="random")
        server.game_logger.start_game = MagicMock()
        server.game_logger.end_game = MagicMock()
        server.game_logger.log_incoming = MagicMock()
        server.game_logger.log_outgoing = MagicMock()
        server.game_logger.log_security_event = MagicMock()

        mock_ws = AsyncMock()
        # Simulate: one message then disconnect (use a heartbeat which is harmless)
        mock_ws.__aiter__ = MagicMock(return_value=iter([
            json.dumps({"timestamp": 1234567890}),
        ]))

        # handle_connection should not raise
        await server.handle_connection(mock_ws)
        # Logger should be cleaned up
        server.game_logger.end_game.assert_called()

    @pytest.mark.asyncio
    async def test_handler_timeout_sends_fallback(self):
        """If a handler takes too long, _safe_execute sends END_TURN."""
        server = BridgeServer(bot_type="random")

        async def slow_handler():
            await asyncio.sleep(100)

        sent = []
        original_send = server._send

        def capture_send(data):
            sent.append(data)

        server._send = capture_send

        with patch("bridge.server.DECISION_TIMEOUT_SECONDS", 0.05):
            await server._safe_execute(slow_handler)

        assert any(d.get("action") == 5 for d in sent), "Should send END_TURN fallback"

    @pytest.mark.asyncio
    async def test_malformed_message_does_not_crash_consumer(self):
        """Consumer processes malformed messages without crashing."""
        server = BridgeServer(bot_type="random")
        server.game_logger.start_game = MagicMock()
        server.game_logger.log_incoming = MagicMock()
        server.game_logger.log_security_event = MagicMock()
        server.game_logger.end_game = MagicMock()

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=iter([
            "not valid json {{{",
            json.dumps({"unknown_field_xyz": True}),
        ]))

        # Consumer should process all messages without raising
        await server._consumer(mock_ws)
        # Security events should be logged
        assert server.game_logger.log_security_event.called

    @pytest.mark.asyncio
    async def test_producer_handles_connection_closed(self):
        """Producer exits cleanly if WebSocket is closed during send."""
        import websockets.exceptions

        server = BridgeServer(bot_type="random")
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = websockets.exceptions.ConnectionClosed(None, None)
        server.queue.put_nowait(json.dumps({"action": 5}))

        # Should exit without raising
        await server._producer(mock_ws)

    @pytest.mark.asyncio
    async def test_safe_execute_handles_sync_function(self):
        """_safe_execute should handle regular (non-async) functions too."""
        server = BridgeServer(bot_type="random")

        calls = []

        def sync_handler():
            calls.append(1)

        await server._safe_execute(sync_handler)
        # Note: sync handlers are wrapped in run_in_executor, so this tests
        # the executor path

    @pytest.mark.asyncio
    async def test_unknown_message_logs_security_event(self):
        """Unknown messages should trigger security event logging."""
        server = BridgeServer(bot_type="random")
        server.game_logger.start_game = MagicMock()
        server.game_logger.log_incoming = MagicMock()
        server.game_logger.log_security_event = MagicMock()

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=iter([
            json.dumps({"totally_unknown": "message"}),
        ]))

        await server._consumer(mock_ws)
        server.game_logger.log_security_event.assert_called()


class TestBridgeServerBuildIndexMaps:
    def test_build_index_maps_from_init(self):
        """_build_index_maps_from_init should create coordinate-to-index lookups."""
        server = BridgeServer(bot_type="random")

        map_state = {
            "tileCornerStates": {
                "0": {"x": 0, "y": 0, "z": 0},
                "1": {"x": 1, "y": 0, "z": 1},
            },
            "tileEdgeStates": {
                "0": {"x": 0, "y": 0, "z": 0},
                "1": {"x": 1, "y": 0, "z": 1},
            },
        }

        server._build_index_maps_from_init(map_state)
        assert server._vertex_coord_to_idx[(0, 0, 0)] == 0
        assert server._vertex_coord_to_idx[(1, 0, 1)] == 1
        assert server._edge_coord_to_idx[(0, 0, 0)] == 0
        assert server._edge_coord_to_idx[(1, 0, 1)] == 1
        assert server._vertex_index_to_coord[0] == (0, 0, 0)
        assert server._edge_index_to_coord[1] == (1, 0, 1)


class TestGameLoggerIntegration:
    def test_game_logger_start_end_cycle(self):
        """GameLogger start/end cycle should work without errors."""
        import tempfile, os
        from bridge.game_logger import GameLogger
        from bridge import config

        # Use a temp directory
        original_log_dir = config.LOG_DIR
        config.LOG_DIR = tempfile.mkdtemp()

        try:
            gl = GameLogger()
            gl.start_game()
            assert gl._file is not None
            gl.log_incoming("TestMsg", '{"test": 1}')
            gl.log_outgoing("action", {"action": 5})
            gl.log_security_event("test_event", "test details")
            gl.end_game()
            assert gl._file is None

            # Verify file was written
            log_files = os.listdir(config.LOG_DIR)
            assert len(log_files) == 1
            assert log_files[0].endswith(".jsonl")

        finally:
            config.LOG_DIR = original_log_dir

    def test_game_logger_end_game_idempotent(self):
        """end_game() should be safe to call multiple times."""
        import tempfile
        from bridge.game_logger import GameLogger
        from bridge import config

        original_log_dir = config.LOG_DIR
        config.LOG_DIR = tempfile.mkdtemp()

        try:
            gl = GameLogger()
            gl.start_game()
            gl.end_game()
            gl.end_game()  # Second call - should not raise
            assert gl._file is None
        finally:
            config.LOG_DIR = original_log_dir
