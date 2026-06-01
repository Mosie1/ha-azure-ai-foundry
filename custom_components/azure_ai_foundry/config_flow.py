"""Config flow for the Azure AI Foundry integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType

from . import _build_client
from .const import (
    CONF_DEPLOYMENT_NAME,
    CONF_ENDPOINT,
    CONF_IMAGE_DEPLOYMENT,
    CONF_IMAGE_QUALITY,
    CONF_IMAGE_SIZE,
    CONF_LLM_HASS_API,
    CONF_MAX_TOKENS,
    CONF_MODEL_FAMILY,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    IMAGE_QUALITY_OPTIONS,
    IMAGE_SIZE_OPTIONS,
    LOGGER,
    MODEL_FAMILY_AUTO,
    MODEL_FAMILY_OPENAI,
    MODEL_FAMILY_OTHER,
    REASONING_EFFORT_OPTIONS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_DEPLOYMENT_NAME,
    RECOMMENDED_IMAGE_QUALITY,
    RECOMMENDED_IMAGE_SIZE,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    resolve_api,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ENDPOINT): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Required(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)

MODEL_FAMILY_SELECTOR = SelectSelector(
    SelectSelectorConfig(
        options=[MODEL_FAMILY_AUTO, MODEL_FAMILY_OPENAI, MODEL_FAMILY_OTHER],
        mode=SelectSelectorMode.DROPDOWN,
        translation_key=CONF_MODEL_FAMILY,
    )
)


async def _validate_connection(hass, data: dict[str, Any]) -> None:
    """Try to connect to the endpoint. Raises openai errors on failure."""
    client = _build_client(hass, data)
    try:
        await client.with_options(timeout=10.0).models.list()
    except openai.NotFoundError:
        # Endpoint reachable and credentials accepted, listing unsupported.
        pass


class AzureAIFoundryConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Azure AI Foundry."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match({CONF_ENDPOINT: user_input[CONF_ENDPOINT]})
            try:
                await _validate_connection(self.hass, user_input)
            except openai.AuthenticationError:
                errors["base"] = "invalid_auth"
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.OpenAIError:
                LOGGER.exception("Unexpected error validating Azure AI Foundry")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title="Azure AI Foundry",
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                            "data": RECOMMENDED_CONVERSATION_OPTIONS,
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                        {
                            "subentry_type": SUBENTRY_TYPE_AI_TASK_DATA,
                            "data": RECOMMENDED_AI_TASK_OPTIONS,
                            "title": DEFAULT_AI_TASK_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle re-authentication."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm re-authentication by re-entering the API key."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()
        if user_input is not None:
            data = {**reauth_entry.data, CONF_API_KEY: user_input[CONF_API_KEY]}
            try:
                await _validate_connection(self.hass, data)
            except openai.AuthenticationError:
                errors["base"] = "invalid_auth"
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.OpenAIError:
                LOGGER.exception("Unexpected error during reauth")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    reauth_entry, data_updates={CONF_API_KEY: user_input[CONF_API_KEY]}
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry types supported by this integration."""
        return {
            SUBENTRY_TYPE_CONVERSATION: AzureAIFoundrySubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK_DATA: AzureAIFoundrySubentryFlowHandler,
        }


class AzureAIFoundrySubentryFlowHandler(ConfigSubentryFlow):
    """Flow to create or reconfigure a conversation / AI task agent."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return True when creating a new subentry."""
        return self.source == "user"

    @property
    def _is_conversation(self) -> bool:
        return self._subentry_type == SUBENTRY_TYPE_CONVERSATION

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create a new subentry."""
        if self._is_conversation:
            self.options = dict(RECOMMENDED_CONVERSATION_OPTIONS)
        else:
            self.options = dict(RECOMMENDED_AI_TASK_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Collect name, prompt, LLM API and the recommended toggle."""
        # The parent entry must be loaded to reconfigure a subentry.
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        # Drop any stored LLM API ids that are no longer available.
        if suggested_apis := self.options.get(CONF_LLM_HASS_API):
            if isinstance(suggested_apis, str):
                suggested_apis = [suggested_apis]
            valid_apis = {api.id for api in llm.async_get_apis(self.hass)}
            self.options[CONF_LLM_HASS_API] = [
                api for api in suggested_apis if api in valid_apis
            ]

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            self.options.update(user_input)
            if user_input.get(CONF_RECOMMENDED):
                return self._async_finish()
            return await self.async_step_advanced()

        schema: VolDictType = {}
        if self._is_new:
            default_name = (
                DEFAULT_CONVERSATION_NAME
                if self._is_conversation
                else DEFAULT_AI_TASK_NAME
            )
            schema[vol.Required(CONF_NAME, default=default_name)] = str

        if self._is_conversation:
            schema[
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": self.options.get(CONF_PROMPT)
                    },
                )
            ] = TemplateSelector()
            schema[
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={
                        "suggested_value": self.options.get(CONF_LLM_HASS_API)
                    },
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=api.id, label=api.name)
                        for api in llm.async_get_apis(self.hass)
                    ],
                    multiple=True,
                )
            )

        schema[
            vol.Required(
                CONF_RECOMMENDED, default=self.options.get(CONF_RECOMMENDED, False)
            )
        ] = bool

        return self.async_show_form(
            step_id="init", data_schema=vol.Schema(schema)
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Collect deployment, model family and generation parameters."""
        if user_input is not None:
            self.options.update(user_input)
            return await self.async_step_model()

        schema: VolDictType = {
            vol.Required(
                CONF_DEPLOYMENT_NAME,
                default=self.options.get(
                    CONF_DEPLOYMENT_NAME, RECOMMENDED_DEPLOYMENT_NAME
                ),
            ): str,
            vol.Required(
                CONF_MODEL_FAMILY,
                default=self.options.get(CONF_MODEL_FAMILY, MODEL_FAMILY_AUTO),
            ): MODEL_FAMILY_SELECTOR,
            vol.Optional(
                CONF_MAX_TOKENS,
                default=self.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
            ): int,
            vol.Optional(
                CONF_TEMPERATURE,
                default=self.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_TOP_P,
                default=self.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        }

        return self.async_show_form(
            step_id="advanced", data_schema=vol.Schema(schema)
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Collect model-specific options (reasoning, image generation)."""
        if user_input is not None:
            # Drop empty optional values so they fall back to defaults.
            self.options.update(
                {k: v for k, v in user_input.items() if v not in (None, "")}
            )
            return self._async_finish()

        schema: VolDictType = {}

        family = self.options.get(CONF_MODEL_FAMILY, MODEL_FAMILY_AUTO)
        deployment = self.options.get(CONF_DEPLOYMENT_NAME, "")
        if resolve_api(family, deployment) == "responses":
            schema[
                vol.Optional(
                    CONF_REASONING_EFFORT,
                    description={
                        "suggested_value": self.options.get(
                            CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                        )
                    },
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=REASONING_EFFORT_OPTIONS,
                    translation_key=CONF_REASONING_EFFORT,
                )
            )

        if not self._is_conversation:
            schema[
                vol.Optional(
                    CONF_IMAGE_DEPLOYMENT,
                    description={
                        "suggested_value": self.options.get(CONF_IMAGE_DEPLOYMENT)
                    },
                )
            ] = str
            schema[
                vol.Optional(
                    CONF_IMAGE_SIZE,
                    default=self.options.get(CONF_IMAGE_SIZE, RECOMMENDED_IMAGE_SIZE),
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=IMAGE_SIZE_OPTIONS, custom_value=True
                )
            )
            schema[
                vol.Optional(
                    CONF_IMAGE_QUALITY,
                    default=self.options.get(
                        CONF_IMAGE_QUALITY, RECOMMENDED_IMAGE_QUALITY
                    ),
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=IMAGE_QUALITY_OPTIONS, custom_value=True
                )
            )

        if not schema:
            return self._async_finish()

        return self.async_show_form(
            step_id="model", data_schema=vol.Schema(schema)
        )

    def _async_finish(self) -> SubentryFlowResult:
        """Create or update the subentry from collected options."""
        title = self.options.pop(CONF_NAME, None)
        if self._is_new:
            return self.async_create_entry(
                title=title
                or (
                    DEFAULT_CONVERSATION_NAME
                    if self._is_conversation
                    else DEFAULT_AI_TASK_NAME
                ),
                data=self.options,
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=self.options,
        )
