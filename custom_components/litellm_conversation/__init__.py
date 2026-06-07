"""The LiteLLM Conversation integration."""

from __future__ import annotations

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client

from .const import CONF_BASE_URL, LOGGER

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)

type LiteLLMConversationConfigEntry = ConfigEntry[openai.AsyncOpenAI]


def _build_client(hass: HomeAssistant, data: dict) -> openai.AsyncOpenAI:
    """Create an OpenAI client pointed at the user's LiteLLM proxy.

    The LiteLLM proxy exposes an OpenAI-compatible API, so the standard
    ``AsyncOpenAI`` client works directly against its base URL; the proxy
    handles provider routing (Azure OpenAI, Foundry models, Claude, ...).
    """
    return openai.AsyncOpenAI(
        base_url=data[CONF_BASE_URL],
        api_key=data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: LiteLLMConversationConfigEntry
) -> bool:
    """Set up LiteLLM Conversation from a config entry."""
    client = _build_client(hass, dict(entry.data))

    # Validate the proxy URL and credentials by listing the proxy's models.
    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid LiteLLM proxy credentials: %s", err)
        raise ConfigEntryAuthFailed("Invalid authentication") from err
    except openai.APIConnectionError as err:
        raise ConfigEntryNotReady("Unable to connect to the LiteLLM proxy") from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(str(err)) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_migrate_entry(
    hass: HomeAssistant, entry: LiteLLMConversationConfigEntry
) -> bool:
    """Migrate an old config entry.

    No migrations are needed yet; this exists so future schema changes can be
    handled non-breakingly via VERSION / MINOR_VERSION bumps.
    """
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: LiteLLMConversationConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(
    hass: HomeAssistant, entry: LiteLLMConversationConfigEntry
) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
