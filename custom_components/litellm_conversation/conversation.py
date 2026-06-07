"""Conversation agent for LiteLLM Conversation."""

from __future__ import annotations

from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import LiteLLMConversationConfigEntry
from .const import CONF_LLM_HASS_API, CONF_PROMPT, SUBENTRY_TYPE_CONVERSATION
from .entity import LiteLLMConversationBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LiteLLMConversationConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities from their subentries."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue
        async_add_entities(
            [LiteLLMConversationConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMConversationConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    LiteLLMConversationBaseLLMEntity,
):
    """LiteLLM Conversation conversation agent."""

    def __init__(
        self, entry: LiteLLMConversationConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register the agent when added."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the agent when removed."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a user message."""
        options = self.subentry.data
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(self.entry.domain),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
