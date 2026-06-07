"""Shared base entity and Chat Completions request handling.

The integration talks to a LiteLLM proxy's OpenAI-compatible endpoint, so a
single Chat Completions code path serves every model the proxy is configured
for (Azure OpenAI, Foundry models, Claude, ...).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import TYPE_CHECKING, Any

import openai
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MODEL,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if TYPE_CHECKING:
    from . import LiteLLMConversationConfigEntry


def _decode_tool_arguments(arguments: str) -> Any:
    """Parse tool-call arguments, raising a friendly error on bad JSON.

    Models often fill optional parameters with blank values (e.g.
    ``"floor": ""`` or ``"device_class": []``). Home Assistant intents reject
    those as invalid slot info, so empty values are stripped from the top-level
    arguments object before the tool is called.
    """
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(
            f"Unexpected tool argument response: {err}"
        ) from err

    if isinstance(data, dict):
        return {
            key: value
            for key, value in data.items()
            if value is not None and value != "" and value != [] and value != {}
        }
    return data


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Recursively make a JSON schema compatible with strict mode.

    Strict structured output requires every property listed in ``required``.
    To preserve optionality, properties that were *not* originally required are
    made nullable (their type becomes ``[type, "null"]``) before being added to
    ``required`` -- mirroring the official ``open_router`` integration. We also
    set ``additionalProperties: false`` on objects.
    """
    if schema.get("type") == "object":
        if "properties" not in schema:
            return
        schema["additionalProperties"] = False
        required = schema.setdefault("required", [])
        for prop, prop_schema in schema["properties"].items():
            _adjust_schema(prop_schema)
            if prop not in required:
                # Optional field: keep it optional by allowing null, then mark
                # it required (strict mode requires every property in `required`).
                prop_type = prop_schema.get("type")
                if isinstance(prop_type, str):
                    prop_schema["type"] = [prop_type, "null"]
                elif isinstance(prop_type, list) and "null" not in prop_type:
                    prop_schema["type"] = [*prop_type, "null"]
                required.append(prop)
    elif schema.get("type") == "array" and "items" in schema:
        _adjust_schema(schema["items"])


def _format_structured_output(
    name: str, schema: vol.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Convert a voluptuous schema into a strict JSON schema definition."""
    converted: dict[str, Any] = convert(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )
    _adjust_schema(converted)
    return {"name": slugify(name or "output"), "schema": converted, "strict": True}


# Keys the function-tool schema validator rejects at the top level. Some HA
# intents (e.g. HassStartTimer, which requires at least one of hours/minutes/
# seconds) convert to a schema with one of these at the top level, which the
# Chat Completions API rejects with a 400.
_UNSUPPORTED_TOOL_SCHEMA_KEYS = ("oneOf", "anyOf", "allOf", "enum", "not")


def _format_tool_parameters(
    tool: llm.Tool, llm_api: llm.APIInstance
) -> dict[str, Any]:
    """Convert a tool's parameters, stripping unsupported top-level keys."""
    schema = convert(tool.parameters, custom_serializer=llm_api.custom_serializer)
    if any(key in schema for key in _UNSUPPORTED_TOOL_SCHEMA_KEYS):
        schema = {
            key: value
            for key, value in schema.items()
            if key not in _UNSUPPORTED_TOOL_SCHEMA_KEYS
        }
    return schema


def _format_tools(
    llm_api: llm.APIInstance | None,
) -> list[dict[str, Any]] | None:
    """Format HA LLM tools for the Chat Completions API."""
    if not llm_api:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": _format_tool_parameters(tool, llm_api),
            },
        }
        for tool in llm_api.tools
    ]


def _convert_content_to_chat_message(
    content: conversation.Content,
) -> dict[str, Any] | None:
    """Convert HA chat-log content to a Chat Completions message dict."""
    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_call_id,
            "content": json.dumps(content.tool_result),
        }

    if content.role == "system" and content.content:
        return {"role": "system", "content": content.content}

    if content.role == "user" and content.content:
        return {"role": "user", "content": content.content}

    if content.role == "assistant":
        message: dict[str, Any] = {"role": "assistant", "content": content.content}
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.tool_args),
                    },
                }
                for tool_call in content.tool_calls
            ]
        return message

    return None


async def _transform_chat_message(
    message: Any,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a Chat Completions response message into a HA delta."""
    data: conversation.AssistantContentDeltaDict = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        data["tool_calls"] = [
            llm.ToolInput(
                id=tool_call.id,
                tool_name=tool_call.function.name,
                tool_args=_decode_tool_arguments(tool_call.function.arguments),
            )
            for tool_call in message.tool_calls
            if tool_call.type == "function"
        ]
    yield data


class LiteLLMConversationBaseLLMEntity(Entity):
    """Shared base for the conversation and AI task entities."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: LiteLLMConversationConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="LiteLLM",
            model=subentry.data.get(CONF_MODEL, RECOMMENDED_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _client(self) -> openai.AsyncOpenAI:
        return self.entry.runtime_data

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log via the Chat Completions API."""
        options = self.subentry.data
        messages = [
            msg
            for content in chat_log.content
            if (msg := _convert_content_to_chat_message(content))
        ]

        model_args: dict[str, Any] = {
            "model": options.get(CONF_MODEL, RECOMMENDED_MODEL),
            "messages": messages,
            "user": chat_log.conversation_id,
            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
        }
        # Reasoning models reject temperature/top_p, so when a reasoning effort
        # is configured we send that instead. The proxy (with drop_params)
        # normalizes anything the target model doesn't accept.
        if effort := options.get(CONF_REASONING_EFFORT):
            model_args["reasoning_effort"] = effort
        else:
            model_args["temperature"] = options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            )
            model_args["top_p"] = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)

        if tools := _format_tools(chat_log.llm_api):
            model_args["tools"] = tools

        if structure:
            if TYPE_CHECKING:
                assert structure_name is not None
            model_args["response_format"] = {
                "type": "json_schema",
                "json_schema": _format_structured_output(
                    structure_name, structure, chat_log.llm_api
                ),
            }

        client = self._client
        for _iteration in range(max_iterations):
            try:
                result = await client.chat.completions.create(**model_args)
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError(
                    "LiteLLM proxy authentication error"
                ) from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to the LiteLLM proxy: %s", err)
                raise HomeAssistantError(
                    f"Error talking to the LiteLLM proxy: {err}"
                ) from err

            choice = result.choices[0]
            if choice.finish_reason == "length":
                raise HomeAssistantError(
                    "The response was truncated (max tokens reached). "
                    "Try increasing max tokens."
                )

            added_content = False
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_chat_message(choice.message)
            ):
                added_content = True
                if (msg := _convert_content_to_chat_message(content)) is not None:
                    model_args["messages"].append(msg)

            if not chat_log.unresponded_tool_results:
                break
            if not added_content:
                raise HomeAssistantError("The LiteLLM proxy returned an empty response.")
