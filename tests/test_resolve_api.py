"""Tests for the pure API-resolution and schema helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import voluptuous as vol

from homeassistant.util import slugify

from custom_components.litellm_conversation.const import (
    MODEL_FAMILY_AUTO,
    MODEL_FAMILY_OPENAI,
    MODEL_FAMILY_OTHER,
    is_anthropic_deployment,
    is_reasoning_deployment,
    resolve_api,
)
from custom_components.litellm_conversation.entity import (
    _UNSUPPORTED_TOOL_SCHEMA_KEYS,
    _adjust_schema,
    _decode_tool_arguments,
    _format_structured_output,
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


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        ("claude-sonnet-4-6", True),
        ("claude-3-5-sonnet", True),
        ("Claude-Opus", True),
        ("gpt-4o-mini", False),
        ("deepseek-r1", False),
    ],
)
def test_is_anthropic_deployment(deployment: str, expected: bool) -> None:
    """Claude deployments are detected so they raise a clear error."""
    assert is_anthropic_deployment(deployment) is expected


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

    from custom_components.litellm_conversation.entity import (
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


def test_adjust_schema_strict_mode_with_optional_fields() -> None:
    """All props become required; originally-optional ones become nullable."""
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "note": {"type": "string"},  # optional -> should become nullable
        },
    }
    _adjust_schema(schema)

    assert schema["additionalProperties"] is False
    # Strict mode: every property must be listed in required.
    assert set(schema["required"]) == {"name", "note"}
    # Originally-required field keeps its plain type.
    assert schema["properties"]["name"]["type"] == "string"
    # Originally-optional field is made nullable to preserve optionality.
    assert schema["properties"]["note"]["type"] == ["string", "null"]


def test_adjust_schema_recurses_objects_and_arrays() -> None:
    """Nested objects and array items are adjusted too."""
    schema = {
        "type": "object",
        "properties": {
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

    assert schema["properties"]["nested"]["additionalProperties"] is False
    assert schema["properties"]["nested"]["required"] == ["value"]
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["required"] == ["x"]


def test_format_structured_output_slugifies_name() -> None:
    """A task name with spaces becomes an API-safe schema name."""
    result = _format_structured_output(
        "My Task Name", vol.Schema({vol.Required("x"): str}), None
    )
    assert result["name"] == slugify("My Task Name")
    assert result["strict"] is True
    assert result["schema"]["type"] == "object"
