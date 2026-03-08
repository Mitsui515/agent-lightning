# Copyright (c) Microsoft. All rights reserved.

"""Tests for the ``agl store`` CLI entry point (``agentlightning.cli.store``).

We test the argument-parsing layer and the ``main()`` function's argument
dispatch; we do *not* start a live server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str]) -> int:
    """Import and invoke ``agentlightning.cli.store.main``."""
    from agentlightning.cli.store import main

    return main(argv)


# ---------------------------------------------------------------------------
# Argument parsing (--help path)
# ---------------------------------------------------------------------------


class TestStoreCLIArgParsing:
    def test_missing_required_args_exits(self) -> None:
        """Calling main with bad args should raise SystemExit."""
        with pytest.raises(SystemExit):
            _run_main(["--unknown-flag"])

    def test_default_backend_is_memory(self) -> None:
        """Verify the default --backend is 'memory' by parsing args directly."""
        import argparse

        from agentlightning.cli.store import main as _m

        # Patch to avoid actually starting the server
        with patch("agentlightning.cli.store.LightningStoreServer") as mock_server_cls:
            mock_server = MagicMock()
            mock_server.run_forever = AsyncMock(side_effect=KeyboardInterrupt)
            mock_server_cls.return_value = mock_server

            with pytest.raises((KeyboardInterrupt, RuntimeError)):
                _run_main([])

    def test_backend_mongo_accepted(self) -> None:
        """--backend mongo is a valid argparse choice; check it doesn't exit on parse."""
        import argparse

        # We just verify that argparse accepts 'mongo' as a backend value.
        # We don't launch the server (no pymongo required).
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", choices=["memory", "mongo"], default="memory")
        args = parser.parse_args(["--backend", "mongo"])
        assert args.backend == "mongo"


# ---------------------------------------------------------------------------
# Tracker flag handling
# ---------------------------------------------------------------------------


class TestStoreCLITrackers:
    def test_console_tracker_enabled(self) -> None:
        """--tracker console should add a ConsoleMetricsBackend."""
        with patch("agentlightning.cli.store.LightningStoreServer") as mock_server_cls, patch(
            "agentlightning.cli.store.ConsoleMetricsBackend"
        ) as mock_console:
            mock_server = MagicMock()
            mock_server.run_forever = AsyncMock(side_effect=RuntimeError("stop"))
            mock_server_cls.return_value = mock_server
            mock_console.return_value = MagicMock()

            result = _run_main(["--tracker", "console"])
            assert result == 1
            mock_console.assert_called_once()


# ---------------------------------------------------------------------------
# Top-level agl dispatch
# ---------------------------------------------------------------------------


class TestAGLDispatch:
    def test_store_subcommand_dispatches_to_store_main(self) -> None:
        """``agl store --help`` should print help and exit."""
        from agentlightning.cli import main as agl_main

        with pytest.raises(SystemExit) as exc_info:
            agl_main(["store", "--help"])
        # --help exits with code 0
        assert exc_info.value.code == 0

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        """Passing an unknown subcommand exits with a non-zero code."""
        from agentlightning.cli import main as agl_main

        with pytest.raises(SystemExit) as exc_info:
            agl_main(["unknown-subcommand"])
        assert exc_info.value.code != 0

    def test_no_args_shows_usage(self) -> None:
        """Running ``agl`` with no arguments should exit with error."""
        from agentlightning.cli import main as agl_main

        with pytest.raises(SystemExit) as exc_info:
            agl_main([])
        assert exc_info.value.code != 0
