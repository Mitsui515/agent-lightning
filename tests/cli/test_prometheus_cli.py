# Copyright (c) Microsoft. All rights reserved.

"""Tests for the ``agl prometheus`` CLI entry point.

We test the helper functions that can be exercised without starting a live
server, the ``create_prometheus_app()`` factory, and the ``main()`` argument
dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentlightning.cli.prometheus import create_prometheus_app, ensure_prometheus_dir


# ---------------------------------------------------------------------------
# ensure_prometheus_dir
# ---------------------------------------------------------------------------


class TestEnsurePrometheusDir:
    def test_raises_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        with pytest.raises(ValueError, match="PROMETHEUS_MULTIPROC_DIR"):
            ensure_prometheus_dir()

    def test_creates_directory_if_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        new_dir = str(tmp_path / "prom_metrics")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", new_dir)
        result = ensure_prometheus_dir()
        assert result == new_dir
        assert os.path.isdir(new_dir)

    def test_returns_directory_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        result = ensure_prometheus_dir()
        assert result == str(tmp_path)


# ---------------------------------------------------------------------------
# create_prometheus_app
# ---------------------------------------------------------------------------


class TestCreatePrometheusApp:
    def test_valid_path_returns_app(self) -> None:
        from fastapi import FastAPI

        app = create_prometheus_app("/v1/prometheus")
        assert isinstance(app, FastAPI)

    def test_path_without_leading_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            create_prometheus_app("v1/prometheus")

    def test_root_path_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be '/'"):
            create_prometheus_app("/")

    def test_app_has_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        app = create_prometheus_app("/v1/prometheus")
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_custom_metrics_path(self) -> None:
        app = create_prometheus_app("/metrics/v2")
        # App should be created without raising.
        assert app is not None


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------


class TestPrometheusMainDispatch:
    def test_help_exits_zero(self) -> None:
        from agentlightning.cli.prometheus import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_prometheus_dir_returns_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() should raise ValueError when PROMETHEUS_MULTIPROC_DIR is not set."""
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        from agentlightning.cli.prometheus import main

        with pytest.raises(ValueError, match="PROMETHEUS_MULTIPROC_DIR"):
            main([])
