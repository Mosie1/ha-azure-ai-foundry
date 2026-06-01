"""The Azure AI Foundry integration."""

from __future__ import annotations

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client

from .const import AZURE_API_VERSION, CONF_ENDPOINT, LOGGER

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)

type AzureAIFoundryConfigEntry = ConfigEntry[openai.AsyncOpenAI]


def _build_client(hass: HomeAssistant, data: dict) -> openai.AsyncOpenAI:
    """Create an Azure AI Foundry client targeting the OpenAI v1 endpoint.

    The classic ``AzureOpenAI`` client routes to ``/openai/...?api-version=...``
    and rewrites chat calls to ``/deployments/<model>/...``; that surface has no
    Responses API. We therefore point a plain ``AsyncOpenAI`` client at the
    version-less v1 endpoint, which serves both Chat Completions and Responses.
    """
    endpoint = data[CONF_ENDPOINT].rstrip("/")
    return openai.AsyncOpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=data[CONF_API_KEY],
        default_query={"api-version": AZURE_API_VERSION},
        http_client=get_async_client(hass),
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: AzureAIFoundryConfigEntry
) -> bool:
    """Set up Azure AI Foundry from a config entry."""
    client = _build_client(hass, dict(entry.data))

    # Validate the endpoint and credentials. Azure does not reliably expose a
    # deployment listing, so a NotFoundError still means the endpoint is
    # reachable and the key was accepted.
    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid Azure AI Foundry credentials: %s", err)
        raise ConfigEntryAuthFailed("Invalid authentication") from err
    except openai.NotFoundError:
        # Endpoint reachable, credentials accepted, no listing available.
        pass
    except openai.APIConnectionError as err:
        raise ConfigEntryNotReady("Unable to connect to Azure AI Foundry") from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(str(err)) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: AzureAIFoundryConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(
    hass: HomeAssistant, entry: AzureAIFoundryConfigEntry
) -> None:
    """Reload the entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
