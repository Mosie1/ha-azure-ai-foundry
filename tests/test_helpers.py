"""Tests for the pure entity helpers."""

from __future__ import annotations

from types import SimpleNamespace

import voluptuous as vol

from homeassistant.util import slugify

from custom_components.litellm_conversation.entity import (
    _UNSUPPORTED_TOOL_SCHEMA_KEYS,
    _adjust_schema,
    _decode_tool_arguments,
    _format_structured_output,
    _format_tool_parameters,
)


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
    assert set(schema["required"]) == {"name", "note"}
    assert schema["properties"]["name"]["type"] == "string"
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
