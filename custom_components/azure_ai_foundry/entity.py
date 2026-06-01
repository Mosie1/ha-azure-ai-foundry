"""Shared base entity and dual-API request handling for Azure AI Foundry."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import TYPE_CHECKING, Any, Literal

import openai
from openai.types.responses import ResponseReasoningItem, ResponseReasoningItemParam
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from .const import (
    CONF_DEPLOYMENT_NAME,
    CONF_MAX_TOKENS,
    CONF_MODEL_FAMILY,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    MAX_TOOL_ITERATIONS,
    MODEL_FAMILY_AUTO,
    RECOMMENDED_DEPLOYMENT_NAME,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    is_reasoning_deployment,
    resolve_api,
)

if TYPE_CHECKING:
    from . import AzureAIFoundryConfigEntry


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

    Azure/OpenAI strict structured output requires ``additionalProperties:
    false`` and every property listed in ``required`` on each object.
    """
    if schema.get("type") == "object":
        if "properties" not in schema:
            return
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"])
        for child in schema["properties"].values():
            _adjust_schema(child)
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
    return {"name": name, "schema": converted, "strict": True}


# Keys the function-tool schema validator rejects at the top level. Some HA
# intents (e.g. HassStartTimer, which requires at least one of hours/minutes/
# seconds) convert to a schema with one of these at the top level, which both
# the Chat Completions and Responses APIs reject with a 400.
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


def _format_tools_responses(
    llm_api: llm.APIInstance | None,
) -> list[dict[str, Any]] | None:
    """Format HA LLM tools for the Responses API."""
    if not llm_api:
        return None
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": _format_tool_parameters(tool, llm_api),
        }
        for tool in llm_api.tools
    ]


# --- Chat Completions conversion -------------------------------------------


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


# --- Responses conversion ---------------------------------------------------


def _convert_content_to_response_input(
    content: conversation.Content,
) -> list[dict[str, Any]]:
    """Convert HA chat-log content to Responses API input item(s)."""
    if isinstance(content, conversation.ToolResultContent):
        return [
            {
                "type": "function_call_output",
                "call_id": content.tool_call_id,
                "output": json.dumps(content.tool_result),
            }
        ]

    items: list[dict[str, Any]] = []

    # Re-emit the reasoning item first so it chronologically precedes the
    # message/function call it produced. This preserves a reasoning model's
    # state across tool-call turns (mirrors openai_conversation).
    if (
        isinstance(content, conversation.AssistantContent)
        and isinstance(content.native, ResponseReasoningItem)
        and content.native.encrypted_content
    ):
        items.append(
            ResponseReasoningItemParam(
                type="reasoning",
                id=content.native.id,
                summary=[],
                encrypted_content=content.native.encrypted_content,
            )
        )

    if content.content:
        role = "developer" if content.role == "system" else content.role
        items.append({"type": "message", "role": role, "content": content.content})

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        items.extend(
            {
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.tool_name,
                "arguments": json.dumps(tool_call.tool_args),
            }
            for tool_call in content.tool_calls
        )

    return items


async def _transform_response_output(
    response: Any,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a Responses API response into a HA delta."""
    text_parts: list[str] = []
    tool_calls: list[llm.ToolInput] = []
    reasoning: ResponseReasoningItem | None = None

    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    text_parts.append(part.text)
        elif item.type == "function_call":
            tool_calls.append(
                llm.ToolInput(
                    id=item.call_id,
                    tool_name=item.name,
                    tool_args=_decode_tool_arguments(item.arguments),
                )
            )
        elif item.type == "reasoning" and reasoning is None:
            # Keep the reasoning item so it can be sent back next turn,
            # preserving the model's reasoning state (reasoning models only).
            reasoning = item

    data: conversation.AssistantContentDeltaDict = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        data["tool_calls"] = tool_calls
    if reasoning is not None:
        data["native"] = reasoning
    yield data


# --- Base entity ------------------------------------------------------------


class AzureAIFoundryBaseLLMEntity(Entity):
    """Shared base for Azure AI Foundry conversation and AI task entities."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: AzureAIFoundryConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Microsoft",
            model=subentry.data.get(
                CONF_DEPLOYMENT_NAME, RECOMMENDED_DEPLOYMENT_NAME
            ),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _client(self) -> openai.AsyncOpenAI:
        return self.entry.runtime_data

    def _resolve_api(self) -> Literal["responses", "chat"]:
        """Return which Azure OpenAI API this agent should use."""
        options = self.subentry.data
        return resolve_api(
            options.get(CONF_MODEL_FAMILY, MODEL_FAMILY_AUTO),
            options.get(CONF_DEPLOYMENT_NAME, ""),
        )

    def _apply_model_params(
        self, model_args: dict[str, Any], api: Literal["responses", "chat"]
    ) -> None:
        """Apply token / temperature / reasoning parameters per API and model."""
        options = self.subentry.data
        deployment = options.get(CONF_DEPLOYMENT_NAME, "")
        max_tokens = options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        reasoning = is_reasoning_deployment(deployment)

        if api == "responses":
            model_args["max_output_tokens"] = max_tokens
            # Manage reasoning state ourselves rather than server-side.
            model_args["store"] = False
            if reasoning:
                model_args["reasoning"] = {
                    "effort": options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    )
                }
                # Ask for the encrypted reasoning so it can be replayed on the
                # next turn, preserving the model's state across tool calls.
                model_args["include"] = ["reasoning.encrypted_content"]
        else:  # chat
            if reasoning:
                model_args["max_completion_tokens"] = max_tokens
            else:
                model_args["max_tokens"] = max_tokens

        if not reasoning:
            model_args["temperature"] = options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            )
            model_args["top_p"] = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log via the resolved API."""
        if self._resolve_api() == "responses":
            await self._async_handle_responses_api(
                chat_log, structure_name, structure, max_iterations
            )
        else:
            await self._async_handle_chat_completions_api(
                chat_log, structure_name, structure, max_iterations
            )

    async def _async_handle_chat_completions_api(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None,
        structure: vol.Schema | None,
        max_iterations: int,
    ) -> None:
        """Handle the chat log using the Chat Completions API."""
        options = self.subentry.data
        messages = [
            msg
            for content in chat_log.content
            if (msg := _convert_content_to_chat_message(content))
        ]

        model_args: dict[str, Any] = {
            "model": options.get(CONF_DEPLOYMENT_NAME, RECOMMENDED_DEPLOYMENT_NAME),
            "messages": messages,
            "user": chat_log.conversation_id,
        }
        self._apply_model_params(model_args, "chat")

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
                raise HomeAssistantError("Azure AI Foundry authentication error") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Azure AI Foundry: %s", err)
                raise HomeAssistantError(
                    f"Error talking to Azure AI Foundry: {err}"
                ) from err

            choice = result.choices[0]
            if choice.finish_reason == "length":
                raise HomeAssistantError(
                    "Azure AI Foundry response was truncated (max tokens reached). "
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
                raise HomeAssistantError(
                    "Azure AI Foundry returned an empty response."
                )

    async def _async_handle_responses_api(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None,
        structure: vol.Schema | None,
        max_iterations: int,
    ) -> None:
        """Handle the chat log using the Responses API."""
        options = self.subentry.data
        input_items: list[dict[str, Any]] = []
        for content in chat_log.content:
            input_items.extend(_convert_content_to_response_input(content))

        model_args: dict[str, Any] = {
            "model": options.get(CONF_DEPLOYMENT_NAME, RECOMMENDED_DEPLOYMENT_NAME),
            "input": input_items,
            "user": chat_log.conversation_id,
        }
        self._apply_model_params(model_args, "responses")

        if tools := _format_tools_responses(chat_log.llm_api):
            model_args["tools"] = tools

        if structure:
            if TYPE_CHECKING:
                assert structure_name is not None
            fmt = _format_structured_output(
                structure_name, structure, chat_log.llm_api
            )
            model_args["text"] = {"format": {"type": "json_schema", **fmt}}

        client = self._client
        for _iteration in range(max_iterations):
            try:
                response = await client.responses.create(**model_args)
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Azure AI Foundry authentication error") from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Azure AI Foundry: %s", err)
                raise HomeAssistantError(
                    f"Error talking to Azure AI Foundry: {err}"
                ) from err

            if getattr(response, "status", None) == "incomplete":
                reason = getattr(
                    getattr(response, "incomplete_details", None), "reason", None
                )
                raise HomeAssistantError(
                    "Azure AI Foundry returned an incomplete response "
                    f"({reason or 'unknown reason'}). For reasoning models this "
                    "usually means the reasoning used up the token budget — try "
                    "increasing max tokens or using a non-reasoning model."
                )

            added_content = False
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_response_output(response)
            ):
                added_content = True
                model_args["input"].extend(
                    _convert_content_to_response_input(content)
                )

            if not chat_log.unresponded_tool_results:
                break
            if not added_content:
                raise HomeAssistantError(
                    "Azure AI Foundry returned an empty response. For reasoning "
                    "models, try increasing max tokens or using a non-reasoning model."
                )
