"""Fixtures for Azure AI Foundry tests."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from custom_components.azure_ai_foundry.const import (
    CONF_API_VERSION,
    CONF_ENDPOINT,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_API_VERSION,
    RECOMMENDED_CONVERSATION_OPTIONS,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
)

from pytest_homeassistant_custom_component.common import MockConfigEntry


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading of the custom integration in all tests."""
    yield


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a mock config entry with one conversation and one AI task agent."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Azure AI Foundry",
        data={
            CONF_ENDPOINT: "https://example.openai.azure.com",
            CONF_API_KEY: "test-key",
            CONF_API_VERSION: RECOMMENDED_API_VERSION,
        },
        subentries_data=[
            {
                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                "data": RECOMMENDED_CONVERSATION_OPTIONS,
                "title": "Conversation",
                "unique_id": None,
            },
            {
                "subentry_type": SUBENTRY_TYPE_AI_TASK_DATA,
                "data": RECOMMENDED_AI_TASK_OPTIONS,
                "title": "Task",
                "unique_id": None,
            },
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_client() -> Generator[MagicMock]:
    """Patch the AsyncAzureOpenAI client used by the integration."""
    with patch(
        "custom_components.azure_ai_foundry.openai.AsyncAzureOpenAI"
    ) as mock_ctor:
        client = mock_ctor.return_value
        client.models.list = AsyncMock(return_value=[])
        client.with_options.return_value = client
        client.chat.completions.create = AsyncMock()
        client.responses.create = AsyncMock()
        client.images.generate = AsyncMock()
        yield client
