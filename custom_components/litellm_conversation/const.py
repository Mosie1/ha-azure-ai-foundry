"""Constants for the LiteLLM Conversation integration."""

from __future__ import annotations

import logging
from typing import Final

from homeassistant.const import CONF_LLM_HASS_API  # noqa: F401  (re-exported for convenience)
from homeassistant.helpers import llm

DOMAIN: Final = "litellm_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Connection / main config entry. The integration talks to a LiteLLM proxy's
# OpenAI-compatible endpoint, so it only needs a base URL and an API key
# (the proxy master key or a virtual key).
CONF_BASE_URL: Final = "base_url"
# CONF_API_KEY is reused from homeassistant.const

# Subentry types
SUBENTRY_TYPE_CONVERSATION: Final = "conversation"
SUBENTRY_TYPE_AI_TASK_DATA: Final = "ai_task_data"

# Per-subentry options
CONF_MODEL: Final = "model"
CONF_RECOMMENDED: Final = "recommended"
CONF_PROMPT: Final = "prompt"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_REASONING_EFFORT: Final = "reasoning_effort"
CONF_IMAGE_MODEL: Final = "image_model"
CONF_IMAGE_SIZE: Final = "image_size"
CONF_IMAGE_QUALITY: Final = "image_quality"

# Recommended defaults
RECOMMENDED_MODEL: Final = "gpt-4o-mini"
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

DEFAULT_CONVERSATION_NAME: Final = "LiteLLM Conversation"
DEFAULT_AI_TASK_NAME: Final = "LiteLLM Task"

RECOMMENDED_CONVERSATION_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_MODEL: RECOMMENDED_MODEL,
}

RECOMMENDED_AI_TASK_OPTIONS: Final = {
    CONF_RECOMMENDED: True,
    CONF_MODEL: RECOMMENDED_MODEL,
}
