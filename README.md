# LiteLLM Conversation for Home Assistant

A Home Assistant custom integration that powers **Conversation** (Assist) and **AI Tasks** through a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) — an OpenAI-compatible gateway you run yourself.

The integration is a thin client: it points the `openai` SDK at your LiteLLM proxy and speaks Chat Completions. The proxy handles all provider routing, so a single Home Assistant integration can reach **any model your proxy is configured for** — Azure OpenAI, Azure AI Foundry models (DeepSeek, Llama, Mistral, Phi), Anthropic Claude, OpenAI, local models, and more.

> Independent project, not affiliated with or endorsed by LiteLLM/BerriAI, Microsoft, OpenAI, or Anthropic.

## Why a proxy?

Putting LiteLLM in front of your providers keeps Home Assistant simple and provider-agnostic: routing, credentials, model aliases, and parameter quirks all live in the proxy's `config.yaml`. The integration only needs a URL and a key, and it stays a clean wrapper around one SDK (the same pattern as Home Assistant's official `open_router` integration).

## Features

- 🗣️ **Conversation agent** for Assist, with Home Assistant LLM tool/function calling.
- 🧩 **AI Task – data** with JSON-schema structured output.
- 🖼️ **AI Task – image** generation (when your proxy exposes an image model).
- 🧱 Multiple agents per proxy via config subentries, each with its own model and settings.
- 🧠 Optional reasoning-effort passthrough for reasoning models.

## Requirements

- Home Assistant 2025.7 or newer.
- A running **LiteLLM proxy** reachable from Home Assistant, with at least one model in its `model_list`.

## Step 1 — Run a LiteLLM proxy

Example `config.yaml` routing to Azure AI Foundry (including a Claude model) plus any other provider:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: azure/gpt-4o-mini
      api_base: https://YOUR-RESOURCE.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-10-21"
  - model_name: deepseek-r1
    litellm_params:
      model: azure_ai/deepseek-r1
      api_base: https://YOUR-RESOURCE.services.ai.azure.com
      api_key: os.environ/AZURE_AI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: azure_ai/claude-sonnet-4-5
      api_base: https://YOUR-RESOURCE.services.ai.azure.com/anthropic
      api_key: os.environ/AZURE_AI_API_KEY

litellm_settings:
  drop_params: true   # let the proxy drop params a given model doesn't accept

general_settings:
  master_key: sk-1234   # use a strong secret; or issue virtual keys
```

Minimal `docker-compose.yml`:

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    volumes:
      - ./config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml"]
    environment:
      AZURE_API_KEY: "..."
      AZURE_AI_API_KEY: "..."
```

`drop_params: true` is recommended so reasoning models (which reject `temperature`/`top_p`) and other model-specific constraints are handled by the proxy rather than failing.

## Step 2 — Install the integration (HACS)

1. In HACS, add this repository as a **custom repository** (category: *Integration*).
2. Install **LiteLLM Conversation**, then restart Home Assistant.
3. **Settings → Devices & Services → Add Integration → LiteLLM Conversation**.

## Step 3 — Configure

When you add the integration you provide:

| Field | Example | Notes |
| --- | --- | --- |
| Base URL | `http://homeassistant.local:4000` | Your LiteLLM proxy URL (host + port). |
| API key | `sk-1234` | The proxy master key or a virtual key. |

After the entry is created, add one or more **agents** (subentries):

- **Conversation** — a voice/chat agent for Assist.
- **AI Task** — used by the `ai_task.generate_data` / `ai_task.generate_image` actions.

For each agent you pick a **model** (the list is populated from your proxy's `/models`) and set generation options (max tokens, temperature, top-p, optional reasoning effort). AI-Task agents can additionally set an **image model** to enable image generation.

## Notes & limitations

- **Tool calling / structured output** work to the extent the chosen model and your proxy support them. Claude-via-Foundry tool calling, in particular, depends on the LiteLLM version and proxy configuration.
- Image generation uses `client.images.generate` against the proxy and requires a proxy model that supports it.
- The integration delivers the full response at once (no token streaming yet — see roadmap).

## Roadmap / TODO

- **Streaming responses** (token-by-token).
- **Attachment support** — convert attachment content (images, PDFs) into image inputs, then advertise `AITaskEntityFeature.SUPPORT_ATTACHMENTS`.
- **Adaptive guidance / "memory"** — persist lessons from recurring tool-call mistakes (e.g. models over-filling optional intent slots) into the prompt over time; likely a separate, provider-agnostic integration.

## Official integration / brand assets

For a future Home Assistant core submission, brand assets (`icon.png`, `logo.png`) must be contributed to the [home-assistant/brands](https://github.com/home-assistant/brands) repository under `custom_integrations/litellm_conversation/`. Because the integration depends only on the `openai` SDK (a thin wrapper around one SDK), core submission remains viable.

## License

[Apache-2.0](LICENSE)
