/**
 * pi-llama-swappy: minimal Pi provider extension for llama-swappy
 * (https://github.com/.../llama-swappy), an OpenAI-compatible proxy that
 * starts and swaps llama.cpp models on demand.
 *
 * Configuration is done through Pi's built-in `/login` command:
 *   /login llama-swappy
 *
 * On startup (and on `/model` / `pi update --models`) the provider fetches
 * its catalog from the llama-swappy self-description endpoint:
 *   GET <endpoint>/llama-swappy/info
 */
import {
  createProvider,
  type ApiKeyCredential,
  type Model,
} from "@earendil-works/pi-ai";
import { getApiProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER_ID = "llama-swappy";
const PROVIDER_NAME = "llama-swappy";

/** Provider-scoped env key that stores the endpoint inside the credential. */
const CRED_ENV_ENDPOINT = "LLAMA_SWAPPY_ENDPOINT";
/** Ambient env fallbacks. */
const ENV_ENDPOINT = "LLAMA_SWAPPY_ENDPOINT";
const ENV_API_KEY = "LLAMA_SWAPPY_API_KEY";

const INFO_PATH = "/llama-swappy/info";

const DEFAULT_CONTEXT_WINDOW = 8192;

interface SwappyInfoModel {
  key: string;
  name?: string;
  contextSize?: number;
  maxTokens?: number;
  reasoning?: boolean;
  input?: string[];
}

interface SwappyInfo {
  version?: string;
  models?: SwappyInfoModel[];
}

function normalizeEndpoint(endpoint: string | undefined): string | undefined {
  const trimmed = endpoint?.trim();
  if (!trimmed) return undefined;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) {
    throw new Error(
      `Invalid llama-swappy endpoint: "${trimmed}" (expected e.g. http://127.0.0.1:12380)`,
    );
  }
  return trimmed.replace(/\/+$/, "");
}

/** Resolve the active endpoint: stored credential first, then ambient env. */
function getEndpoint(credential: ApiKeyCredential | undefined): string | undefined {
  return (
    normalizeEndpoint(credential?.env?.[CRED_ENV_ENDPOINT]) ??
    normalizeEndpoint(process.env[ENV_ENDPOINT])
  );
}

function getApiKey(credential: ApiKeyCredential | undefined): string | undefined {
  return credential?.key?.trim() || process.env[ENV_API_KEY]?.trim() || undefined;
}

/** Fetch the llama-swappy self-description. */
async function fetchSwappyInfo(
  endpoint: string,
  apiKey: string | undefined,
  signal: AbortSignal,
): Promise<SwappyInfo> {
  const url = `${endpoint}${INFO_PATH}`;
  const response = await fetch(url, {
    signal,
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
  });
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status} ${response.statusText}`);
  }
  const data = (await response.json()) as SwappyInfo;
  if (!Array.isArray(data.models)) {
    throw new Error(`Unexpected /llama-swappy/info response: no "models" array`);
  }
  return data;
}

/** Map llama-swappy model metadata to Pi model definitions. */
function toPiModels(info: SwappyInfo, endpoint: string): Model<"openai-completions">[] {
  return info.models!.map((m) => {
    const contextWindow = m.contextSize ?? DEFAULT_CONTEXT_WINDOW;
    return {
      id: m.key,
      name: m.name ?? m.key,
      api: "openai-completions" as const,
      provider: PROVIDER_ID,
      baseUrl: endpoint,
      reasoning: m.reasoning ?? false,
      input: (m.input ?? ["text"]) as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: m.maxTokens ?? Math.min(32768, Math.floor(contextWindow / 2)),
    };
  });
}

export default function (pi: ExtensionAPI) {
  // llama-swappy speaks OpenAI chat completions; delegate all request
  // conversion and streaming to Pi's built-in implementation.
  const api = getApiProvider("openai-completions");
  if (!api) {
    throw new Error("openai-completions API implementation not found in pi-ai");
  }

  pi.registerProvider(
    createProvider({
      id: PROVIDER_ID,
      name: PROVIDER_NAME,
      models: [],
      api,
      auth: {
        apiKey: {
          name: "llama-swappy endpoint",
          login: async (interaction) => {
            const endpointRaw = await interaction.prompt({
              type: "text",
              message: "llama-swappy endpoint (e.g. http://127.0.0.1:12380)",
            });
            const endpoint = normalizeEndpoint(endpointRaw);
            if (!endpoint) {
              throw new Error("Endpoint is required");
            }
            interaction.notify({
              type: "info",
              message: `Fetching models from ${endpoint}${INFO_PATH}...`,
            });
            const apiKey = (
              await interaction.prompt({
                type: "secret",
                message: "API key (leave empty if the server does not require one)",
              })
            )?.trim() ?? "";

            // Verify connectivity before persisting, so /login does not
            // save an endpoint that cannot be reached.
            await fetchSwappyInfo(endpoint, apiKey || undefined, interaction.signal);

            const credential: ApiKeyCredential = {
              type: "api_key",
              key: apiKey,
              env: { [CRED_ENV_ENDPOINT]: endpoint },
            };
            interaction.notify({
              type: "info",
              message: "llama-swappy configured. Models are available now.",
            });
            return credential;
          },
          check: async ({ credential, signal }) => {
            signal.throwIfAborted();
            if (getEndpoint(credential as ApiKeyCredential | undefined)) {
              return { type: "api_key", source: "stored credential" };
            }
            if (normalizeEndpoint(process.env[ENV_ENDPOINT])) {
              return { type: "api_key", source: ENV_ENDPOINT };
            }
            return undefined;
          },
          resolve: async ({ credential, signal }) => {
            signal.throwIfAborted();
            const endpoint = getEndpoint(credential as ApiKeyCredential | undefined);
            if (!endpoint) {
              return undefined;
            }
            const apiKey = getApiKey(credential as ApiKeyCredential | undefined);
            return {
              auth: {
                baseUrl: endpoint,
                ...(apiKey ? { apiKey } : {}),
              },
              env: { [CRED_ENV_ENDPOINT]: endpoint },
              source: "stored credential",
            };
          },
        },
      },
      fetchModels: async (context) => {
        const credential = context.credential as ApiKeyCredential | undefined;
        const endpoint = getEndpoint(credential);
        if (!endpoint) {
          throw new Error(
            "llama-swappy is not configured: run /login llama-swappy",
          );
        }
        const info = await fetchSwappyInfo(
          endpoint,
          getApiKey(credential),
          context.signal,
        );
        return toPiModels(info, endpoint);
      },
    }),
  );
}
