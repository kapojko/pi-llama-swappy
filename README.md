# pi-llama-swappy

A minimal [Pi](https://pi.dev) extension that adds a **llama-swappy** provider.

[llama-swappy](https://github.com/kapojko/llama-swappy) is an OpenAI-compatible
proxy that starts the requested llama.cpp model on demand and swaps it out
when a different model is requested. This extension registers it as a Pi
provider:

- **No custom streaming code** — llama-swappy speaks OpenAI chat completions,
  so all request handling and streaming delegate to Pi's built-in
  `openai-completions` implementation.
- **Dynamic model catalog** — on startup (and on `/model`, `/login`, or
  `pi update --models`) the provider fetches its model list and per-model
  settings (`contextSize`, `maxTokens`, `reasoning`) from the llama-swappy
  self-description endpoint `GET <endpoint>/llama-swappy/info`.
- **Cost is zero** — models run locally.

## Configuration

Configure everything through Pi's built-in `/login` command:

```
/login llama-swappy
```

You will be prompted for:

1. **Endpoint** — the llama-swappy listen address, e.g. `http://127.0.0.1:12380`
2. **API key** — leave empty if the server does not require one

The endpoint and key are stored in Pi's credential storage
(`~/.pi/agent/auth.json`). You can also configure ambiently without
`/login` by setting environment variables `LLAMA_SWAPPY_ENDPOINT` (required)
and `LLAMA_SWAPPY_API_KEY` (optional).

## Install

```bash
pi install git:github.com/kapojko/pi-llama-swappy
```

Then pick your provider and model:

```
/model llama-swappy <model-key>
```

## Development

Load the extension directly, no build step needed:

```bash
pi --extension ./extensions/llama-swappy/index.ts
```

After editing a discovered extension in a running session, use `/reload`.
