"""Tests for the pure API-resolution and schema helpers."""

from __future__ import annotations

import pytest

from custom_components.azure_ai_foundry.const import (
    MODEL_FAMILY_AUTO,
    MODEL_FAMILY_OPENAI,
    MODEL_FAMILY_OTHER,
    is_reasoning_deployment,
    resolve_api,
)
from types import SimpleNamespace

import voluptuous as vol

from custom_components.azure_ai_foundry.entity import (
    _UNSUPPORTED_TOOL_SCHEMA_KEYS,
    _adjust_schema,
    _decode_tool_arguments,
    _format_tool_parameters,
)


@pytest.mark.parametrize(
    ("family", "deployment", "expected"),
    [
        # Explicit overrides win regardless of deployment name.
        (MODEL_FAMILY_OPENAI, "deepseek-r1", "responses"),
        (MODEL_FAMILY_OTHER, "gpt-4o", "chat"),
        # Auto-detect from the deployment-name prefix.
        (MODEL_FAMILY_AUTO, "gpt-4o-mini", "responses"),
        (MODEL_FAMILY_AUTO, "o3-mini", "responses"),
        (MODEL_FAMILY_AUTO, "chatgpt-4o", "responses"),
        (MODEL_FAMILY_AUTO, "deepseek-r1", "chat"),
        (MODEL_FAMILY_AUTO, "llama-3.1", "chat"),
        (MODEL_FAMILY_AUTO, "mistral-large", "chat"),
        (MODEL_FAMILY_AUTO, "phi-4", "chat"),
        (MODEL_FAMILY_AUTO, "my-custom-deploy", "chat"),
    ],
)
def test_resolve_api(family: str, deployment: str, expected: str) -> None:
    """The right API is chosen per family and deployment name."""
    assert resolve_api(family, deployment) == expected


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        ("o1-mini", True),
        ("o3", True),
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-4o", False),
        ("deepseek-r1", False),
    ],
)
def test_is_reasoning_deployment(deployment: str, expected: bool) -> None:
    """Reasoning deployments are detected from their prefix."""
    assert is_reasoning_deployment(deployment) is expected


def test_decode_tool_arguments_strips_empty_values() -> None:
    """Blank optional slots are removed so HA intents don't reject them."""
    raw = (
        '{"name": "Kleine lamp bureau", "area": "Bureau", "floor": "", '
        '"domain": ["light"], "device_class": [], "brightness": 0, '
        '"on": false, "extra": {}}'
    )
    assert _decode_tool_arguments(raw) == {
        "name": "Kleine lamp bureau",
        "area": "Bureau",
        "domain": ["light"],
        # 0 and False are meaningful values and must be preserved.
        "brightness": 0,
        "on": False,
    }


def test_format_tool_parameters_strips_top_level_unsupported_keys() -> None:
    """Top-level anyOf/oneOf/etc. are removed (e.g. HassStartTimer 400)."""
    # "At least one of hours/minutes/seconds", like HassStartTimer, which
    # converts to a schema with a top-level anyOf the function-tool API rejects.
    schema = vol.Schema(
        vol.All(
            {
                vol.Optional("name"): str,
                vol.Optional("hours"): int,
                vol.Optional("minutes"): int,
                vol.Optional("seconds"): int,
            },
            vol.Any(
                vol.Schema({vol.Required("hours"): int}, extra=vol.ALLOW_EXTRA),
                vol.Schema({vol.Required("minutes"): int}, extra=vol.ALLOW_EXTRA),
                vol.Schema({vol.Required("seconds"): int}, extra=vol.ALLOW_EXTRA),
            ),
        )
    )
    tool = SimpleNamespace(parameters=schema)
    llm_api = SimpleNamespace(custom_serializer=None)

    result = _format_tool_parameters(tool, llm_api)

    assert result["type"] == "object"
    assert "properties" in result
    assert not any(key in result for key in _UNSUPPORTED_TOOL_SCHEMA_KEYS)


def test_reasoning_item_round_trips_before_function_call() -> None:
    """A reasoning model's state is replayed before the call it produced."""
    from openai.types.responses import ResponseReasoningItem

    from homeassistant.components import conversation
    from homeassistant.helpers import llm

    from custom_components.azure_ai_foundry.entity import (
        _convert_content_to_response_input,
    )

    content = conversation.AssistantContent(
        agent_id="x",
        content=None,
        tool_calls=[
            llm.ToolInput(id="call_1", tool_name="GetLiveContext", tool_args={})
        ],
        native=ResponseReasoningItem(
            id="rs_123", type="reasoning", summary=[], encrypted_content="ENC"
        ),
    )

    items = _convert_content_to_response_input(content)

    assert [item["type"] for item in items] == ["reasoning", "function_call"]
    assert items[0]["id"] == "rs_123"
    assert items[0]["encrypted_content"] == "ENC"


def test_adjust_schema_enforces_strict_mode() -> None:
    """Objects get additionalProperties:false and all keys required."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            },
        },
    }
    _adjust_schema(schema)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "nested", "items"}
    assert schema["properties"]["nested"]["additionalProperties"] is False
    assert schema["properties"]["nested"]["required"] == ["value"]
    assert (
        schema["properties"]["items"]["items"]["additionalProperties"] is False
    )
    assert schema["properties"]["items"]["items"]["required"] == ["x"]
