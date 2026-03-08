# Copyright (c) Microsoft. All rights reserved.

"""Tests for agentlightning.types.resources module.

Covers:
- LLM resource creation and serialisation
- ProxyLLM URL generation and access-warning behaviour
- PromptTemplate formatting
- ResourcesUpdate versioning helpers
- NamedResources discriminated-union round-trips
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pytest

from agentlightning.types import LLM, NamedResources, PromptTemplate, ResourcesUpdate
from agentlightning.types.resources import ProxyLLM, Resource


# ---------------------------------------------------------------------------
# Resource base
# ---------------------------------------------------------------------------


class TestResourceBase:
    def test_resource_has_resource_type_field(self) -> None:
        """Resource subclasses must populate resource_type."""
        llm = LLM(endpoint="http://localhost:8000/v1", model="gpt-4o")
        assert llm.resource_type == "llm"

    def test_llm_default_sampling_parameters(self) -> None:
        """sampling_parameters defaults to empty dict."""
        llm = LLM(endpoint="http://localhost:8000/v1", model="llama3")
        assert llm.sampling_parameters == {}

    def test_llm_optional_api_key(self) -> None:
        llm_no_key = LLM(endpoint="http://localhost:8000/v1", model="llama3")
        llm_with_key = LLM(endpoint="http://localhost:8000/v1", model="llama3", api_key="sk-test")
        assert llm_no_key.api_key is None
        assert llm_with_key.api_key == "sk-test"

    def test_llm_get_base_url_returns_endpoint(self) -> None:
        endpoint = "http://localhost:8000/v1"
        llm = LLM(endpoint=endpoint, model="llama3")
        assert llm.get_base_url() == endpoint


# ---------------------------------------------------------------------------
# ProxyLLM URL construction
# ---------------------------------------------------------------------------


class TestProxyLLM:
    def _make_proxy(self, endpoint: str = "http://localhost:8000/v1") -> ProxyLLM:
        return ProxyLLM(endpoint=endpoint, model="proxy-model")

    def test_get_base_url_without_ids_returns_endpoint(self) -> None:
        proxy = self._make_proxy()
        url = proxy.get_base_url(None, None)
        assert url == "http://localhost:8000/v1"

    def test_get_base_url_injects_rollout_and_attempt(self) -> None:
        proxy = self._make_proxy("http://localhost:8000/v1")
        url = proxy.get_base_url("ro-abc", "at-xyz")
        assert "ro-abc" in url
        assert "at-xyz" in url
        assert url.endswith("/v1")

    def test_get_base_url_works_without_trailing_v1(self) -> None:
        proxy = self._make_proxy("http://localhost:8000")
        url = proxy.get_base_url("ro-abc", "at-xyz")
        assert "ro-abc" in url
        assert "at-xyz" in url
        # No /v1 added when it wasn't present.
        assert not url.endswith("/v1")

    def test_get_base_url_strips_trailing_slash(self) -> None:
        proxy = self._make_proxy("http://localhost:8000/v1/")
        url = proxy.get_base_url("ro-r", "at-a")
        assert url.endswith("/v1")

    def test_get_base_url_raises_if_only_one_id_provided(self) -> None:
        proxy = self._make_proxy()
        with pytest.raises(ValueError, match="must be strings or all be empty"):
            proxy.get_base_url("ro-abc", None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be strings or all be empty"):
            proxy.get_base_url(None, "at-xyz")  # type: ignore[arg-type]

    def test_direct_endpoint_access_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Accessing `.endpoint` after initialisation should log a warning."""
        proxy = self._make_proxy()
        with caplog.at_level(logging.WARNING, logger="agentlightning.types.resources"):
            _ = proxy.endpoint
        assert any("endpoint" in msg for msg in caplog.messages)

    def test_with_attempted_rollout_returns_plain_llm(self) -> None:
        from agentlightning.types import AttemptedRollout, Attempt, Rollout

        proxy = self._make_proxy()
        attempt = Attempt(
            rollout_id="ro-x",
            attempt_id="at-y",
            sequence_id=1,
            start_time=0.0,
            status="running",
        )
        rollout = AttemptedRollout(
            rollout_id="ro-x",
            input={"x": 1},
            start_time=0.0,
            status="running",
            attempt=attempt,
        )
        result = proxy.with_attempted_rollout(rollout)
        assert isinstance(result, LLM)
        assert not isinstance(result, ProxyLLM)
        assert "ro-x" in result.endpoint
        assert "at-y" in result.endpoint

    def test_resource_type_is_proxy_llm(self) -> None:
        proxy = self._make_proxy()
        assert proxy.resource_type == "proxy_llm"


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_f_string_formatting(self) -> None:
        tmpl = PromptTemplate(template="Hello, {name}!", engine="f-string")
        assert tmpl.format(name="World") == "Hello, World!"

    def test_f_string_multiple_placeholders(self) -> None:
        tmpl = PromptTemplate(
            template="Task: {task}. Attempt {n} of {total}.",
            engine="f-string",
        )
        result = tmpl.format(task="classify", n=2, total=3)
        assert result == "Task: classify. Attempt 2 of 3."

    def test_unsupported_engine_raises(self) -> None:
        tmpl = PromptTemplate(template="{{ greeting }}", engine="jinja")
        with pytest.raises(NotImplementedError):
            tmpl.format(greeting="hi")

    def test_resource_type_is_prompt_template(self) -> None:
        tmpl = PromptTemplate(template="x", engine="f-string")
        assert tmpl.resource_type == "prompt_template"


# ---------------------------------------------------------------------------
# NamedResources discriminated union round-trip
# ---------------------------------------------------------------------------


class TestNamedResourcesRoundTrip:
    def test_llm_roundtrip(self) -> None:
        """LLM serialises and deserialises through NamedResources."""
        from pydantic import TypeAdapter

        ta: TypeAdapter[NamedResources] = TypeAdapter(NamedResources)
        original: NamedResources = {
            "main": LLM(endpoint="http://localhost:8000/v1", model="gpt-4o")
        }
        encoded = ta.dump_json(original)
        recovered = ta.validate_json(encoded)
        assert recovered["main"].model == "gpt-4o"  # type: ignore[union-attr]
        assert isinstance(recovered["main"], LLM)

    def test_proxy_llm_roundtrip(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[NamedResources] = TypeAdapter(NamedResources)
        original: NamedResources = {
            "proxy": ProxyLLM(endpoint="http://localhost:8000/v1", model="proxy-model")
        }
        encoded = ta.dump_json(original)
        recovered = ta.validate_json(encoded)
        assert isinstance(recovered["proxy"], ProxyLLM)

    def test_prompt_template_roundtrip(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[NamedResources] = TypeAdapter(NamedResources)
        original: NamedResources = {
            "sys": PromptTemplate(template="You are {role}.", engine="f-string")
        }
        encoded = ta.dump_json(original)
        recovered = ta.validate_json(encoded)
        assert isinstance(recovered["sys"], PromptTemplate)
        assert recovered["sys"].template == "You are {role}."  # type: ignore[union-attr]

    def test_mixed_resources_roundtrip(self) -> None:
        from pydantic import TypeAdapter

        ta: TypeAdapter[NamedResources] = TypeAdapter(NamedResources)
        original: NamedResources = {
            "llm": LLM(endpoint="http://example.com/v1", model="llm-model"),
            "prompt": PromptTemplate(template="Hello {name}", engine="f-string"),
        }
        encoded = ta.dump_json(original)
        recovered = ta.validate_json(encoded)
        assert isinstance(recovered["llm"], LLM)
        assert isinstance(recovered["prompt"], PromptTemplate)


# ---------------------------------------------------------------------------
# ResourcesUpdate
# ---------------------------------------------------------------------------


class TestResourcesUpdate:
    def test_creation(self) -> None:
        update = ResourcesUpdate(
            resources_id="res-001",
            create_time=1000.0,
            update_time=1001.0,
            version=1,
            resources={"llm": LLM(endpoint="http://localhost/v1", model="m")},
        )
        assert update.resources_id == "res-001"
        assert update.version == 1
        assert "llm" in update.resources

    def test_serialisation(self) -> None:
        update = ResourcesUpdate(
            resources_id="res-002",
            create_time=0.0,
            update_time=0.0,
            version=2,
            resources={},
        )
        data = update.model_dump()
        assert data["resources_id"] == "res-002"
        assert data["version"] == 2
