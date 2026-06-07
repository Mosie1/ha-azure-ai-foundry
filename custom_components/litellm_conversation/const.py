"""Constants for the LiteLLM Conversation integration."""

from __future__ import annotations

import logging
from typing import Final, Literal

from homeassistant.const import CONF_LLM_HASS_API  # noqa: F401  (re-exported for convenience)
from homeassistant.helpers import llm

DOMAIN: Final = "litellm_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Connection / main config entry
CONF_ENDPOINT: Final = "endpoint"
# CONF_API_KEY is reused from homeassistant.const

# The integration targets Azure's version-less OpenAI v1 endpoint
# (https://<resource>.openai.azure.com/openai/v1/). "preview" opts into the
# newest features, including the Responses API.
AZURE_API_VERSION: Final = "preview"

# Subentry types
SUBENTRY_TYPE_CONVERSATION: Final = "conversation"
SUBENTRY_TYPE_AI_TASK_DATA: Final = "ai_task_data"

# Per-subentry options
CONF_DEPLOYMENT_NAME: Final = "deployment_name"
CONF_MODEL_FAMILY: Final = "model_family"
CONF_RECOMMENDED: Final = "recommended"
CONF_PROMPT: Final = "prompt"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_REASONING_EFFORT: Final = "reasoning_effort"
CONF_IMAGE_DEPLOYMENT: Final = "image_deployment"
CONF_IMAGE_SIZE: Final = "image_size"
CONF_IMAGE_QUALITY: Final = "image_quality"

# Model family selector values
MODEL_FAMILY_AUTO: Final = "auto"
MODEL_FAMILY_OPENAI: Final = "openai"
MODEL_FAMILY_OTHER: Final = "other"

# Deployment-name prefixes that resolve to the OpenAI Responses API under "auto"
OPENAI_FAMILY_PREFIXES: Final = ("gpt", "o1", "o3", "o4", "chatgpt")
# Deployment-name prefixes that need reasoning handling (no temperature/top_p)
REASONING_PREFIXES: Final = ("o1", "o3", "o4", "gpt-5")
# Anthropic Claude models on Foundry use the native Messages API (not the
# OpenAI-compatible surface), which this integration does not support yet.
ANTHROPIC_PREFIXES: Final = ("claude",)

# Recommended defaults
RECOMMENDED_DEPLOYMENT_NAME: Final = "gpt-4o-mini"
RECOMMENDED_MAX_TOKENS: Final = 3000
RECOMMENDED_TEMPERATURE: Final = 1.0
RECOMMENDED_TOP_P: Final = 1.0
RECOMMENDED_REASONING_EFFORT: Final = "low"
RECOMMENDED_IMAGE_SIZE: Final = "1024x1024"
RECOMMENDED_IMAGE_QUALITY: Final = "standard"

REASONING_EFFORT_OPTIONS: Final = ["low", "medium", "high"]
IMAGE_SIZE_OPTIONS: Final = ["1024x1024", "1792x1024", "1024x1792"]
IMAGE_QUALITY_OPTIONS: Final = ["standard", "hd"]

MAX_TOOL_ITERATIONS: Final = 10

DEFAULT_CONVERSATION_NAME: Final = "LiteLLM Conversation Conversation"
DEFAULT_AI_TASK_NAME: Final = "LiteLLM Conversation Task"

RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_DEPLOYMENT_NAME: RECOMMENDED_DEPLOYMENT_NAME,
    CONF_MODEL_FAMILY: MODEL_FAMILY_AUTO,
}

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_DEPLOYMENT_NAME: RECOMMENDED_DEPLOYMENT_NAME,
    CONF_MODEL_FAMILY: MODEL_FAMILY_AUTO,
}


def resolve_api(family: str, deployment: str) -> Literal["responses", "chat"]:
    """Decide which Azure OpenAI API to use for a deployment.

    ``family`` is the user-selected model family override; ``deployment`` is the
    Azure deployment name. Under ``auto`` we guess from the deployment-name
    prefix, defaulting to Chat Completions for anything that is not obviously an
    OpenAI model.
    """
    if family == MODEL_FAMILY_OPENAI:
        return "responses"
    if family == MODEL_FAMILY_OTHER:
        return "chat"
    if deployment.lower().startswith(OPENAI_FAMILY_PREFIXES):
        return "responses"
    return "chat"


def is_reasoning_deployment(deployment: str) -> bool:
    """Return True if the deployment name looks like a reasoning model."""
    return deployment.lower().startswith(REASONING_PREFIXES)


def is_anthropic_deployment(deployment: str) -> bool:
    """Return True if the deployment name looks like an Anthropic Claude model."""
    return deployment.lower().startswith(ANTHROPIC_PREFIXES)
