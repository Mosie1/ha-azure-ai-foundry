"""AI Task entity for LiteLLM Conversation."""

from __future__ import annotations

import base64
from json import JSONDecodeError

import openai

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import LiteLLMConversationConfigEntry
from .const import (
    CONF_IMAGE_MODEL,
    CONF_IMAGE_QUALITY,
    CONF_IMAGE_SIZE,
    LOGGER,
    RECOMMENDED_IMAGE_QUALITY,
    RECOMMENDED_IMAGE_SIZE,
    SUBENTRY_TYPE_AI_TASK_DATA,
)
from .entity import LiteLLMConversationBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LiteLLMConversationConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities from their subentries."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK_DATA:
            continue
        async_add_entities(
            [LiteLLMConversationTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LiteLLMConversationTaskEntity(
    ai_task.AITaskEntity,
    LiteLLMConversationBaseLLMEntity,
):
    """LiteLLM Conversation AI Task entity."""

    def __init__(
        self, entry: LiteLLMConversationConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        # TODO: advertise SUPPORT_ATTACHMENTS once attachment content is
        # converted into image/file inputs (see README roadmap).
        features = ai_task.AITaskEntityFeature.GENERATE_DATA
        if subentry.data.get(CONF_IMAGE_MODEL):
            features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE
        self._attr_supported_features = features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            LOGGER.error(
                "Failed to parse structured response: %s. Response: %s", err, text
            )
            raise HomeAssistantError(
                "Error parsing the structured response"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task via the configured image model."""
        options = self.subentry.data
        image_model = options.get(CONF_IMAGE_MODEL)
        if not image_model:
            raise HomeAssistantError("No image model configured")

        size = options.get(CONF_IMAGE_SIZE, RECOMMENDED_IMAGE_SIZE)
        quality = options.get(CONF_IMAGE_QUALITY, RECOMMENDED_IMAGE_QUALITY)

        try:
            response = await self._client.images.generate(
                model=image_model,
                prompt=task.instructions,
                size=size,
                quality=quality,
                response_format="b64_json",
                n=1,
            )
        except openai.AuthenticationError as err:
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError("LiteLLM proxy authentication error") from err
        except openai.OpenAIError as err:
            LOGGER.error("Error generating image: %s", err)
            raise HomeAssistantError(f"Error generating image: {err}") from err

        image = response.data[0]
        if not image.b64_json:
            raise HomeAssistantError("No image returned")

        width: int | None = None
        height: int | None = None
        if "x" in size:
            raw_width, _, raw_height = size.partition("x")
            if raw_width.isdigit() and raw_height.isdigit():
                width, height = int(raw_width), int(raw_height)

        # gpt-image models can return a non-PNG format; honor it when reported.
        output_format = getattr(response, "output_format", None) or getattr(
            image, "output_format", None
        )
        mime_type = f"image/{output_format}" if output_format else "image/png"

        return ai_task.GenImageTaskResult(
            image_data=base64.b64decode(image.b64_json),
            conversation_id=chat_log.conversation_id,
            mime_type=mime_type,
            width=width,
            height=height,
            model=image_model,
            revised_prompt=getattr(image, "revised_prompt", None),
        )
