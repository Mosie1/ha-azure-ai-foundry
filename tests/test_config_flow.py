"""Tests for the Azure AI Foundry config flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.azure_ai_foundry.const import (
    CONF_API_VERSION,
    CONF_ENDPOINT,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
)

USER_INPUT = {
    CONF_ENDPOINT: "https://example.openai.azure.com",
    CONF_API_KEY: "test-key",
    CONF_API_VERSION: "2024-10-21",
}


@pytest.fixture(autouse=True)
def mark_conversation_loaded(hass: HomeAssistant) -> None:
    """Mark the `conversation` dependency as set up.

    Starting a config flow processes the integration's manifest dependencies;
    the real `conversation` component can't fully initialise in the minimal
    test environment, and these tests don't exercise it.
    """
    hass.config.components.add("conversation")


async def test_user_flow_success(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """A valid endpoint + key creates an entry with two default agents."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM

    with patch(
        "custom_components.azure_ai_foundry.async_setup_entry", return_value=True
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], USER_INPUT
        )
        await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == USER_INPUT

    entry = result["result"]
    subentry_types = {s.subentry_type for s in entry.subentries.values()}
    assert subentry_types == {
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK_DATA,
    }


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (
            openai.AuthenticationError(
                "bad key", response=MagicMock(status_code=401), body=None
            ),
            "invalid_auth",
        ),
        (
            openai.APIConnectionError(request=MagicMock()),
            "cannot_connect",
        ),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    mock_client: MagicMock,
    side_effect: Exception,
    expected_error: str,
) -> None:
    """Connection / auth errors are surfaced and the form is shown again."""
    mock_client.with_options.return_value.models.list.side_effect = side_effect

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}
