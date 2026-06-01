# Azure AI Foundry for Home Assistant

A Home Assistant custom integration that uses [Azure AI Foundry](https://ai.azure.com/) (Azure OpenAI / Azure AI model deployments) to power **Conversation** (Assist) and **AI Tasks**.

It targets the modern Home Assistant entity APIs (chat log, LLM tool calling, AI Task structured data & image generation) and supports the full range of models you can deploy in Azure AI Foundry:

- **OpenAI-family** deployments (GPT-4o, GPT-4.1, o-series, GPT-5, …) are served through the **Responses API**.
- **Other model families** (DeepSeek-R1, Llama, Mistral, Phi, …) are served through the **Chat Completions API**.

The integration auto-detects which API to use from the deployment name, with a per-agent override.

> This is an independent project and is not affiliated with or endorsed by Microsoft or OpenAI.

## Features

- 🗣️ **Conversation agent** for Assist, with Home Assistant LLM tool/function calling and streaming responses.
- 🧩 **AI Task – data** with JSON-schema structured output.
- 🖼️ **AI Task – image** generation via a DALL·E / gpt-image deployment.
- 🔀 **Dual API** (Responses + Chat Completions) so both OpenAI and open-weight models work.
- 🧠 Reasoning-model aware (o-series / GPT-5): reasoning effort, correct token parameters.
- 🧱 Multiple agents per Azure resource via config subentries, each with its own model and settings.

## Requirements

- Home Assistant 2025.7 or newer.
- An Azure AI Foundry / Azure OpenAI resource with at least one **deployment**.
- The resource **endpoint URL** and an **API key**.

## Installation (HACS)

1. In HACS, add this repository as a **custom repository** (category: *Integration*).
2. Install **Azure AI Foundry**.
3. Restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration → Azure AI Foundry**.

## Configuration

When you add the integration you provide:

| Field | Example | Notes |
| --- | --- | --- |
| Endpoint | `https://my-resource.openai.azure.com` | Your Azure resource endpoint. The integration appends `/openai/v1/` automatically. |
| API key | `xxxxxxxx…` | From the Azure portal → *Keys and Endpoint*. |

The integration talks to Azure's version-less **OpenAI v1 endpoint** (`<endpoint>/openai/v1/`), so there is no API-version field to manage — this is what makes the Responses API (used for OpenAI-family models) work.

After the entry is created, add one or more **agents** (subentries):

- **Conversation** — a voice/chat agent for Assist.
- **AI Task** — used by the `ai_task.generate_data` / `ai_task.generate_image` services.

For each agent you set:

- **Deployment name** — the **Azure deployment name** (not the model name). This is what is sent as `model=`.
- **Model family** — `auto` (default), `openai`, or `other`. Use this to force the Responses vs Chat Completions path when the deployment name does not start with a recognized OpenAI prefix.
- Generation settings (max tokens, temperature, top-p, reasoning effort).
- For AI Task agents only: an optional **image deployment** (e.g. `dall-e-3` or `gpt-image-1`) to enable image generation.

> **Deployment name vs model name:** Azure references models by the *deployment* you created, which can be named anything. Always enter the deployment name. The integration cannot reliably list deployments, so this is a free-text field.

## Notes & limitations

- The integration uses the Azure **OpenAI v1 endpoint** with `api-version=preview`. The Responses API (and therefore OpenAI-family deployments) is only available in [certain regions](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses) — if a `gpt-…` deployment returns `404 Resource not found`, either your region/resource doesn't support the Responses API or you can set that agent's **Model family** to *Other (Chat Completions)*.
- Image generation uses `client.images.generate` against your image deployment; size/quality options differ between `dall-e-3` and `gpt-image-1` — custom values are allowed and any API error is surfaced.
- Microsoft Entra ID authentication is not yet supported (API key only).

## Roadmap / TODO

- **Attachment support** — convert attachment content (images, PDFs) into
  image/file inputs for both the Responses and Chat Completions APIs, then
  re-advertise `AITaskEntityFeature.SUPPORT_ATTACHMENTS` on the AI Task entity
  (and handle attachments on the conversation path). The official OpenAI
  integration's `async_prepare_files_for_prompt` is a good reference.
- **Microsoft Entra ID authentication** (currently API key only).
- **Streaming responses** (the integration currently returns the full message
  at once rather than token-by-token).

## Official integration / brand assets

For a future Home Assistant core submission, brand assets (`icon.png`, `logo.png`) must be contributed to the [home-assistant/brands](https://github.com/home-assistant/brands) repository under `custom_integrations/azure_ai_foundry/`.

## License

[Apache-2.0](LICENSE)
