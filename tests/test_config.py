"""Tests for bridge/config.py - configuration and _safe_int validation."""
import os
import pytest
from unittest.mock import patch


class TestSafeInt:
    """Tests for the _safe_int() helper function in config.py."""

    def test_safe_int_valid_value_accepted(self):
        """_safe_int should return a valid integer within the allowed range."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            result = _safe_int("TEST_VAR", default=10, min_val=1, max_val=100)
        assert result == 42

    def test_safe_int_invalid_string_uses_default(self):
        """_safe_int should return the default when env var is not a valid integer."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "not_a_number"}):
            result = _safe_int("TEST_VAR", default=5, min_val=1, max_val=100)
        assert result == 5

    def test_safe_int_out_of_range_uses_default(self):
        """_safe_int should return the default when value is outside [min_val, max_val]."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "200"}):
            result = _safe_int("TEST_VAR", default=10, min_val=1, max_val=100)
        assert result == 10

    def test_safe_int_below_min_uses_default(self):
        """_safe_int should return the default when value is below min_val."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "0"}):
            result = _safe_int("TEST_VAR", default=8, min_val=1, max_val=100)
        assert result == 8

    def test_safe_int_env_var_unset_uses_default(self):
        """_safe_int should return the default when env var is not set."""
        from bridge.config import _safe_int
        env_copy = {k: v for k, v in os.environ.items() if k != "TEST_UNSET_VAR"}
        with patch.dict(os.environ, env_copy, clear=True):
            result = _safe_int("TEST_UNSET_VAR", default=99, min_val=1, max_val=1000)
        assert result == 99

    def test_safe_int_boundary_min_value_accepted(self):
        """_safe_int should accept a value exactly equal to min_val."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            result = _safe_int("TEST_VAR", default=10, min_val=1, max_val=100)
        assert result == 1

    def test_safe_int_boundary_max_value_accepted(self):
        """_safe_int should accept a value exactly equal to max_val."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "100"}):
            result = _safe_int("TEST_VAR", default=10, min_val=1, max_val=100)
        assert result == 100

    def test_safe_int_float_string_uses_default(self):
        """_safe_int should return the default when value is a float string."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": "3.14"}):
            result = _safe_int("TEST_VAR", default=7, min_val=1, max_val=100)
        assert result == 7

    def test_safe_int_empty_string_uses_default(self):
        """_safe_int should return the default when env var is empty string."""
        from bridge.config import _safe_int
        with patch.dict(os.environ, {"TEST_VAR": ""}):
            result = _safe_int("TEST_VAR", default=3, min_val=1, max_val=100)
        assert result == 3
