/**
 * pi-llama-swappy: Pi provider extension for llama-swappy
 * (https://github.com/.../llama-swappy), an OpenAI-compatible proxy that
 * starts and swaps llama.cpp models on demand.
 *
 * One provider instance ("llama-swappy") is always registered, configured as
 * before via `/login llama-swappy` or ambient env. Additional instances can be
 * managed with `/llama-swappy providers`; each is registered as its own
 * provider (`llama-swappy-<name>`) with its own `/login` entry, and the list
 * is persisted in `<agent-dir>/llama-swappy.json`.
 *
 * llama-swappy serves three roots:
 *   GET <base>/llama-swappy/info   — self-description (model catalog)
 *   POST <base>/llama-swappy/unload — unload the currently loaded model
 *   <base>/v1/...                  — OpenAI-compatible chat completions proxy
 *
 * The endpoint entered at /login is normalized to <base>/v1, so Pi's
 * built-in openai-completions implementation requests
 * <base>/v1/chat/completions. The catalog is always fetched from
 * <base>/llama-swappy/info.
 */
import {
  createProvider,
  type ApiKeyCredential,
  type Model,
  type Provider,
} from "@earendil-works/pi-ai";
import { getApiProvider } from "@earendil-works/pi-ai/compat";
import type {
  ExtensionAPI,
  ExtensionCommandContext,
} from "@earendil-works/pi-coding-agent";
import {
  PROVIDER_ID,
  configFilePath,
  instanceNameForProvider,
  loadInstances,
  providerIdFor,
  saveInstances,
  validateName,
  type SwappyInstance,
} from "./config.ts";

/** Provider-scoped env key that stores the endpoint inside the credential. */
const CRED_ENV_ENDPOINT = "LLAMA_SWAPPY_ENDPOINT";
/** Ambient env fallbacks (main instance only). */
const ENV_ENDPOINT = "LLAMA_SWAPPY_ENDPOINT";
const ENV_API_KEY = "LLAMA_SWAPPY_API_KEY";

const INFO_PATH = "/llama-swappy/info";
const UNLOAD_PATH = "/llama-swappy/unload";

const DEFAULT_CONTEXT_WINDOW = 8192;

interface SwappyInfoModel {
  key: string;
  name?: string;
  contextSize?: number;
  slots?: number;
  maxTokens?: number;
  reasoning?: boolean;
  input?: string[];
}

interface SwappyCurrentModel {
  key: string;
  name?: string;
  secondsToUnload?: number;
}

interface SwappyInfo {
  version?: string;
  current?: SwappyCurrentModel | null;
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
  const noSlash = trimmed.replace(/\/+$/, "");
  // llama-swappy's OpenAI proxy lives under /v1 and pi's
  // openai-completions appends /chat/completions to the baseUrl, so the
  // model baseUrl must end with /v1. Accept both input styles.
  return noSlash.endsWith("/v1") ? noSlash : `${noSlash}/v1`;
}

/** Catalog/control URL: these endpoints live at the server root, not under /v1. */
function controlUrl(endpoint: string, path: string): string {
  return `${endpoint.replace(/\/v1$/, "")}${path}`;
}

function infoUrl(endpoint: string): string {
  return controlUrl(endpoint, INFO_PATH);
}

/**
 * Extra instances configured through /llama-swappy providers, persisted in
 * <agent-dir>/llama-swappy.json. Loaded once at startup and kept in sync by
 * the providers manager.
 */
let extraInstances: SwappyInstance[] = loadInstances();

/** Resolve the active endpoint: stored credential first, then config/env. */
function getEndpoint(
  credential: ApiKeyCredential | undefined,
  instance: SwappyInstance | undefined,
): string | undefined {
  return (
    normalizeEndpoint(credential?.env?.[CRED_ENV_ENDPOINT]) ??
    normalizeEndpoint(instance?.endpoint) ??
    normalizeEndpoint(process.env[ENV_ENDPOINT])
  );
}

function getApiKey(
  credential: ApiKeyCredential | undefined,
  instance: SwappyInstance | undefined,
): string | undefined {
  return (
    credential?.key?.trim() ||
    instance?.apiKey?.trim() ||
    process.env[ENV_API_KEY]?.trim() ||
    undefined
  );
}

/** Hard cap for control requests, so a black-holed connection cannot hang the UI. */
const REQUEST_TIMEOUT_MS = 15_000;

/** Combine a caller signal with a timeout for one request. */
function requestSignal(signal: AbortSignal): AbortSignal {
  return AbortSignal.any([signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]);
}

/** Fetch the llama-swappy self-description. */
async function fetchSwappyInfo(
  endpoint: string,
  apiKey: string | undefined,
  signal: AbortSignal,
): Promise<SwappyInfo> {
  const url = infoUrl(endpoint);
  const response = await fetch(url, {
    signal: requestSignal(signal),
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

/** Unload the currently loaded model on one instance. */
async function unloadSwappy(
  endpoint: string,
  apiKey: string | undefined,
  signal: AbortSignal,
): Promise<string | null> {
  const url = controlUrl(endpoint, UNLOAD_PATH);
  const response = await fetch(url, {
    method: "POST",
    signal: requestSignal(signal),
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
  });
  if (!response.ok) {
    throw new Error(
      `POST ${url} failed: ${response.status} ${response.statusText}`,
    );
  }
  const data = (await response.json()) as { unloaded?: string | null };
  return data.unloaded ?? null;
}

/**
 * Resolve the active endpoint and API key for command handlers through the
 * model registry, which re-runs the provider auth resolution (stored
 * credential first, then instance config / ambient env) exactly like
 * request-time auth.
 */
async function resolveSwappyAuth(
  ctx: ExtensionCommandContext,
  providerId: string,
  instance: SwappyInstance | undefined,
): Promise<{ endpoint: string | undefined; apiKey: string | undefined }> {
  let endpoint: string | undefined;
  let apiKey: string | undefined;
  try {
    const auth = await ctx.modelRegistry.getProviderAuth(providerId);
    endpoint =
      normalizeEndpoint(auth?.env?.[CRED_ENV_ENDPOINT]) ??
      normalizeEndpoint(auth?.auth?.baseUrl);
    apiKey = auth?.auth?.apiKey;
  } catch {
    // Registry not ready or provider unregistered; fall through to config/env.
  }
  return {
    endpoint: endpoint ?? getEndpoint(undefined, instance),
    apiKey: apiKey || getApiKey(undefined, instance),
  };
}

function formatUptime(seconds: number): string {
  const m = Math.max(0, Math.round(seconds / 60));
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
}

/** Format the model catalog for /llama-swappy list. */
function formatModelList(info: SwappyInfo, label?: string): string {
  const models = info.models ?? [];
  const lines = models.map((m) => {
    const marker = info.current?.key === m.key ? "●" : " ";
    const bits: string[] = [];
    if (m.contextSize) bits.push(`ctx ${m.contextSize}`);
    if (m.slots) bits.push(`slots ${m.slots}`);
    if (m.maxTokens) bits.push(`max ${m.maxTokens}`);
    if (m.reasoning) bits.push("reasoning");
    const name = m.name && m.name !== m.key ? ` — ${m.name}` : "";
    const meta = bits.length ? ` (${bits.join(", ")})` : "";
    return `${marker} ${m.key}${name}${meta}`;
  });
  const header = label ? `${label} models:` : "llama-swappy models:";
  return [header, ...lines].join("\n");
}

/** Map llama-swappy model metadata to Pi model definitions. */
function toPiModels(
  info: SwappyInfo,
  endpoint: string,
  providerId: string,
): Model<"openai-completions">[] {
  return info.models!.map((m) => {
    const contextWindow = m.contextSize ?? DEFAULT_CONTEXT_WINDOW;
    return {
      id: m.key,
      name: m.name ?? m.key,
      api: "openai-completions" as const,
      provider: providerId,
      baseUrl: endpoint,
      reasoning: m.reasoning ?? false,
      input: (m.input ?? ["text"]) as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens: m.maxTokens ?? Math.min(32768, Math.floor(contextWindow / 2)),
    };
  });
}

/**
 * Show a message to the user. In TUI/RPC this is a notification; in print
 * mode ui.notify is a no-op, so fall back to stdout/stderr.
 */
function emit(
  ctx: ExtensionCommandContext,
  message: string,
  type: "info" | "warning" | "error" = "info",
) {
  if (ctx.hasUI) {
    ctx.ui.notify(message, type);
  } else if (type === "error") {
    console.error(message);
  } else {
    console.log(message);
  }
}

/** Human-readable instance label for messages. */
function describeTarget(providerId: string): string {
  const name = instanceNameForProvider(providerId);
  return name ? `${PROVIDER_ID} (${name})` : PROVIDER_ID;
}

const USAGE = [
  "usage: /llama-swappy <subcommand>",
  "  providers — manage llama-swappy instances (add, delete, inspect)",
  "  list      — show the model catalog (● = loaded on the server)",
  "  info      — server version, loaded model, selected Pi model",
  "  unload    — unload the currently loaded model to free VRAM",
  "",
  "With several instances configured, list/info/unload ask which server to",
  "target (or accept a typed name: /llama-swappy info local, \"all\" for every",
  "instance).",
].join("\n");

const SUBCOMMANDS = [
  { value: "providers", label: "providers", description: "Manage llama-swappy instances" },
  { value: "list", label: "list", description: "Show the model catalog" },
  { value: "info", label: "info", description: "Server version, loaded model, Pi's selection" },
  { value: "unload", label: "unload", description: "Unload the loaded model (free VRAM)" },
];

/** Subcommands that accept an instance target as second token. */
const TARGETED_SUBCOMMANDS = new Set(["list", "info", "unload"]);

interface SwappyTarget {
  /** "all" aggregates over every registered instance. */
  name: string; // display name: "main", instance name, or "all"
  providerId: string;
  instance: SwappyInstance | undefined;
}

/** All configured targets in display order: main first, then extras. */
function allTargets(): SwappyTarget[] {
  return [
    { name: "main", providerId: PROVIDER_ID, instance: undefined },
    ...extraInstances.map((i) => ({
      name: i.name,
      providerId: providerIdFor(i.name),
      instance: i,
    })),
  ];
}

/** Resolve a typed or selected target name ("main" | instance | "all"). */
function resolveTargets(name: string | undefined): SwappyTarget[] | string {
  const targets = allTargets();
  if (!name) return targets.length === 1 ? [targets[0]] : targets; // caller decides
  const needle = name.trim().toLowerCase();
  if (needle === "all") return targets;
  const match = targets.find((t) => t.name.toLowerCase() === needle);
  return match ? [match] : `Unknown instance "${name}". Configured: ${targets.map((t) => t.name).join(", ")}, all.`;
}

/** Ask the user which instance to target, when there is a choice. */
async function selectTargets(
  ctx: ExtensionCommandContext,
  sub: string,
): Promise<SwappyTarget[] | undefined> {
  const targets = allTargets();
  if (targets.length === 1) return [targets[0]];
  if (!ctx.hasUI) {
    emit(
      ctx,
      `Multiple llama-swappy instances configured (${targets.map((t) => t.name).join(", ")}). ` +
        `Run /llama-swappy ${sub} <name> (or "all") — selection lists need an interactive UI.`,
      "warning",
    );
    return undefined;
  }
  const options = [...targets.map((t) => t.name), "all"];
  const choice = await ctx.ui.select(
    `llama-swappy ${sub} — which instance?`,
    options,
  );
  if (choice === undefined) return undefined;
  return resolveTargets(choice) as SwappyTarget[];
}

/**
 * Build a Provider for the main instance (instance === undefined) or an
 * extra instance. All instances share the same auth/login flow; each extra
 * instance is its own provider ID, so `/login llama-swappy-<name>` and
 * `/model llama-swappy-<name>/...` work out of the box.
 */
function createSwappyProvider(
  api: ReturnType<typeof getApiProvider>,
  instance: SwappyInstance | undefined,
): Provider<"openai-completions"> {
  const id = instance ? providerIdFor(instance.name) : PROVIDER_ID;
  const title = instance ? `${PROVIDER_ID} (${instance.name})` : PROVIDER_ID;

  return createProvider({
    id,
    name: title,
    models: [],
    api,
    auth: {
      apiKey: {
        name: `${title} endpoint`,
        login: async (interaction) => {
          const endpointRaw = await interaction.prompt({
            type: "text",
            message:
              `${title} endpoint (e.g. http://127.0.0.1:12380) — /v1 is appended automatically`,
          });
          const endpoint = normalizeEndpoint(endpointRaw);
          if (!endpoint) {
            throw new Error("Endpoint is required");
          }
          interaction.notify({
            type: "info",
            message: `Fetching models from ${infoUrl(endpoint)}...`,
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
            message: `${title} configured. Models are available now.`,
          });
          return credential;
        },
        check: async ({ credential, signal }) => {
          signal.throwIfAborted();
          if (getEndpoint(credential as ApiKeyCredential | undefined, instance)) {
            return { type: "api_key", source: "stored credential" };
          }
          if (!instance && normalizeEndpoint(process.env[ENV_ENDPOINT])) {
            return { type: "api_key", source: ENV_ENDPOINT };
          }
          return undefined;
        },
        resolve: async ({ credential, signal }) => {
          signal.throwIfAborted();
          const endpoint = getEndpoint(credential as ApiKeyCredential | undefined, instance);
          if (!endpoint) {
            return undefined;
          }
          // pi's openai-completions requires an API key or Authorization
          // header (its own sentinel for header-auth is "unused"). Keyless
          // llama-swappy servers ignore the header, so send the sentinel
          // instead of failing with "No API key for provider".
          const apiKey = getApiKey(credential as ApiKeyCredential | undefined, instance) ?? "unused";
          return {
            auth: {
              baseUrl: endpoint,
              apiKey,
            },
            env: { [CRED_ENV_ENDPOINT]: endpoint },
            source: "stored credential",
          };
        },
      },
    },
    fetchModels: async (context) => {
      const credential = context.credential as ApiKeyCredential | undefined;
      const endpoint = getEndpoint(credential, instance);
      if (!endpoint) {
        throw new Error(
          `${title} is not configured: run /login ${id}`,
        );
      }
      const info = await fetchSwappyInfo(
        endpoint,
        getApiKey(credential, instance),
        context.signal,
      );
      return toPiModels(info, endpoint, id);
    },
  });
}

export default function (pi: ExtensionAPI) {
  // llama-swappy speaks OpenAI chat completions; delegate all request
  // conversion and streaming to Pi's built-in implementation.
  const api = getApiProvider("openai-completions");
  if (!api) {
    throw new Error("openai-completions API implementation not found in pi-ai");
  }

  // Main instance plus every configured extra instance. registerProvider()
  // during extension load is applied once the runner binds its context.
  pi.registerProvider(createSwappyProvider(api, undefined));
  for (const instance of extraInstances) {
    pi.registerProvider(createSwappyProvider(api, instance));
  }

  /** Register one extra instance provider, remember it, and fetch its catalog. */
  async function addInstance(
    ctx: ExtensionCommandContext,
    instance: SwappyInstance,
  ): Promise<void> {
    extraInstances = [...extraInstances, instance];
    saveInstances(extraInstances);
    pi.registerProvider(createSwappyProvider(api, instance));
    // Fetch and persist the model catalog now, so /model lists the new
    // provider immediately (startup refresh is offline-only). The add flow
    // already verified connectivity, so a failure here is unexpected.
    try {
      const result = await ctx.modelRegistry.refresh({
        providers: [providerIdFor(instance.name)],
        allowNetwork: true,
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      const error = result.errors.get(providerIdFor(instance.name));
      if (error) throw error;
    } catch (err) {
      if ((err as Error)?.name === "TimeoutError" || (err as Error)?.name === "AbortError") return;
      emit(
        ctx,
        `llama-swappy (${instance.name}) registered, but its model catalog could not be fetched: ${errorMessage(err)}. It will retry on /login or /model.`,
        "warning",
      );
    }
  }

  /** Remove an extra instance provider and forget it. */
  function removeInstance(name: string) {
    extraInstances = extraInstances.filter(
      (i) => i.name.toLowerCase() !== name.toLowerCase(),
    );
    saveInstances(extraInstances);
    pi.unregisterProvider(providerIdFor(name));
  }

  // ------------------------------------------------------------------
  // /llama-swappy providers — instance manager
  // ------------------------------------------------------------------

  /** Interactive "add instance" flow. Returns true when an instance was added. */
  async function addInstanceFlow(ctx: ExtensionCommandContext): Promise<boolean> {
    for (;;) {
      const name = await ctx.ui.input(
        "Instance name — registers provider llama-swappy-<name>",
        "e.g. remote",
      );
      if (name === undefined) return false;
      const nameError = validateName(name, extraInstances);
      if (nameError) {
        emit(ctx, nameError, "error");
        continue;
      }

      const endpointRaw = await ctx.ui.input(
        `Endpoint for "${name.trim()}" (e.g. http://127.0.0.1:12380) — /v1 is appended automatically`,
        "http://",
      );
      if (endpointRaw === undefined) return false;
      let endpoint: string;
      try {
        endpoint = normalizeEndpoint(endpointRaw) ?? "";
      } catch (err) {
        emit(ctx, (err as Error).message, "error");
        continue;
      }
      if (!endpoint) {
        emit(ctx, "Endpoint is required.", "error");
        continue;
      }

      const apiKey = await ctx.ui.input(
        "API key (optional — stored in plain text in llama-swappy.json; use /login llama-swappy-" +
          name.trim() +
          " to store it securely instead)",
        "leave empty if not required",
      );
      if (apiKey === undefined) return false;

      emit(ctx, `Verifying ${infoUrl(endpoint)}...`);
      try {
        const info = await fetchSwappyInfo(
          endpoint,
          apiKey.trim() || undefined,
          new AbortController().signal,
        );
        const instance: SwappyInstance = {
          name: name.trim(),
          endpoint,
          apiKey: apiKey.trim() || undefined,
        };
        addInstance(ctx, instance);
        emit(
          ctx,
          `Added llama-swappy (${instance.name}) — ${info.models?.length ?? 0} models available as llama-swappy-${instance.name}/<model>.`,
        );
        return true;
      } catch (err) {
        // A 401/403 is likely just a missing API key: keep the instance with
        // endpoint only and let the user complete auth via /login.
        const message = err instanceof Error ? err.message : String(err);
        if (/\b(401|403)\b/.test(message)) {
          const instance: SwappyInstance = { name: name.trim(), endpoint };
          addInstance(ctx, instance);
          emit(
            ctx,
            `Added llama-swappy (${instance.name}) but the server requires auth. Run /login llama-swappy-${instance.name} to set the API key.`,
            "warning",
          );
          return true;
        }
        emit(
          ctx,
          `Could not reach ${infoUrl(endpoint)}: ${message}. Instance not added.`,
          "error",
        );
        // Loop back to re-prompt.
      }
    }
  }

  /** Interactive "delete instance" flow. Returns true when one was deleted. */
  async function deleteInstanceFlow(ctx: ExtensionCommandContext): Promise<boolean> {
    if (extraInstances.length === 0) {
      emit(ctx, "No extra llama-swappy instances to delete (main is permanent).", "warning");
      return false;
    }
    const choice = await ctx.ui.select(
      "Delete which llama-swappy instance?",
      extraInstances.map((i) => i.name),
    );
    if (choice === undefined) return false;
    const providerId = providerIdFor(choice);
    if (
      ctx.model?.provider === providerId &&
      !(await ctx.ui.confirm(
        "Delete instance",
        `"${choice}" is the currently selected model's provider. Delete anyway?`,
      ))
    ) {
      return false;
    }
    removeInstance(choice);
    emit(
      ctx,
      `Deleted llama-swappy (${choice}). Its auth.json entry may remain; /logout ${providerId} removes it.`,
    );
    return true;
  }

  async function providersCommand(ctx: ExtensionCommandContext) {
    if (!ctx.hasUI) {
      emit(
        ctx,
        [
          `llama-swappy instances (${extraInstances.length + 1} total; main is implicit):`,
          ...allTargets().map(
            (t) =>
              `  ${t.name.padEnd(12)} → ${t.instance?.endpoint ?? "(from /login llama-swappy or env)"}`,
          ),
          "",
          `Interactive management needs a UI. To add an instance by hand, edit ${configFilePath()}:`,
          `  { "instances": [ { "name": "local", "endpoint": "http://host:port" } ] }`,
          "then /reload.",
        ].join("\n"),
      );
      return;
    }

    for (;;) {
      const targets = allTargets();
      const options = [
        ...targets.map((t) => `${t.name}${t.instance ? ` — ${t.instance.endpoint}` : " — configured via /login or env"}`),
        "Add instance…",
        "Delete instance…",
        "Done",
      ];
      const choice = await ctx.ui.select("llama-swappy providers", options);
      if (choice === undefined || choice === "Done") return;

      if (choice === "Add instance…") {
        await addInstanceFlow(ctx);
        continue;
      }
      if (choice === "Delete instance…") {
        await deleteInstanceFlow(ctx);
        continue;
      }

      // Instance row: show a short summary.
      const name = choice.split(" — ")[0];
      const target = targets.find((t) => t.name === name);
      if (!target) continue;
      const providerId = target.providerId;
      let authNote = "not configured";
      try {
        const auth = await ctx.modelRegistry.getProviderAuth(providerId);
        if (auth?.auth?.baseUrl) {
          authNote = `credential in auth.json (${auth.auth.baseUrl})`;
        } else if (target.instance) {
          authNote = `llama-swappy.json${target.instance.apiKey ? " (with API key)" : " (no API key)"}`;
        }
      } catch {
        // registry not ready
      }
      emit(
        ctx,
        [
          `llama-swappy (${target.name})`,
          `  provider:  ${providerId}`,
          `  endpoint:  ${target.instance?.endpoint ?? "(stored via /login or env)"}`,
          `  auth:      ${authNote}`,
          `  models:    select via /model ${providerId}/<model>; auth via /login ${providerId}`,
        ].join("\n"),
      );
    }
  }

  // ------------------------------------------------------------------
  // /llama-swappy <list|info|unload> [name|all]
  // ------------------------------------------------------------------

  /** Format /llama-swappy info for one target. */
  async function infoFor(
    ctx: ExtensionCommandContext,
    target: SwappyTarget,
    apiKey: string | undefined,
  ): Promise<string> {
    const { endpoint } = await resolveSwappyAuth(ctx, target.providerId, target.instance);
    if (!endpoint) {
      return `${describeTarget(target.providerId)}: not configured (run /login ${target.providerId})`;
    }
    const info = await fetchSwappyInfo(endpoint, apiKey, new AbortController().signal);
    const loaded = info.current
      ? `${info.current.name ?? info.current.key} (unloads in ~${formatUptime(info.current.secondsToUnload ?? 0)} of idle)`
      : "none";
    const selected =
      ctx.model?.provider === target.providerId
        ? `${ctx.model.provider}/${ctx.model.id}`
        : (ctx.model ? `${ctx.model.provider}/${ctx.model.id} (not ${describeTarget(target.providerId)})` : "none");
    return [
      `${describeTarget(target.providerId)} @ ${infoUrl(endpoint)}`,
      `server version: ${info.version ?? "unknown"}`,
      `configured models: ${info.models?.length ?? 0}`,
      `loaded on server: ${loaded}`,
      `selected in pi:  ${selected}`,
    ].join("\n");
  }

  /** Unload one target, returning a human-readable result line. */
  async function unloadFor(
    target: SwappyTarget,
    endpoint: string,
    apiKey: string | undefined,
  ): Promise<string> {
    const unloaded = await unloadSwappy(endpoint, apiKey, new AbortController().signal);
    const label = describeTarget(target.providerId);
    return unloaded ? `${label}: unloaded ${unloaded}.` : `${label}: no model was loaded.`;
  }

  pi.registerCommand("llama-swappy", {
    description: "Manage models on llama-swappy servers (providers, list, info, unload)",
    getArgumentCompletions: (argumentText) => {
      const text = argumentText.trimStart().toLowerCase();
      if (!text.includes(" ")) {
        const matches = SUBCOMMANDS.filter((s) => s.value.startsWith(text));
        return matches.length ? matches : SUBCOMMANDS;
      }
      // Second token: complete instance names (and "all") for targeted
      // subcommands. pi-tui replaces the whole argument text with item.value,
      // so the value repeats the subcommand.
      const match = text.match(/^(list|info|unload)\s+(\S*)$/);
      if (!match) return null;
      const [, sub, namePrefix] = match;
      const names = [...allTargets().map((t) => t.name), "all"];
      const items = names
        .filter((n) => n.startsWith(namePrefix))
        .map((n) => ({
          value: `${sub} ${n}`,
          label: n,
          description: n === "all" ? "Every configured instance" : `Instance ${n}`,
        }));
      return items.length ? items : null;
    },
    handler: async (args, ctx) => {
      const [sub, second] = args.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!sub) {
        emit(ctx, USAGE, "info");
        return;
      }

      try {
        if (sub === "providers") {
          await providersCommand(ctx);
          return;
        }

        if (TARGETED_SUBCOMMANDS.has(sub)) {
          let targets: SwappyTarget[] | undefined;
          if (second) {
            const resolved = resolveTargets(second);
            if (typeof resolved === "string") {
              emit(ctx, `llama-swappy: ${resolved}`, "error");
              return;
            }
            targets = resolved;
          } else {
            targets = await selectTargets(ctx, sub);
            if (!targets) return; // cancelled or non-interactive hint emitted
          }

          // Resolve auth for every target up front; missing endpoints are
          // reported per target instead of aborting the batch.
          const auths = await Promise.all(
            targets.map((t) => resolveSwappyAuth(ctx, t.providerId, t.instance)),
          );
          const blocks: string[] = [];
          for (let i = 0; i < targets.length; i++) {
            const target = targets[i];
            const { endpoint, apiKey } = auths[i];
            if (!endpoint) {
              blocks.push(
                `${describeTarget(target.providerId)}: not configured (run /login ${target.providerId})`,
              );
              continue;
            }
            try {
              if (sub === "list") {
                const info = await fetchSwappyInfo(endpoint, apiKey, new AbortController().signal);
                const label =
                  targets.length > 1 ? describeTarget(target.providerId) : undefined;
                blocks.push(formatModelList(info, label));
              } else if (sub === "info") {
                blocks.push(await infoFor(ctx, target, apiKey));
              } else {
                blocks.push(await unloadFor(target, endpoint, apiKey));
              }
            } catch (err) {
              blocks.push(`${describeTarget(target.providerId)}: ${errorMessage(err)}`);
            }
          }
          emit(ctx, blocks.join("\n\n"), "info");
          return;
        }

        emit(ctx, `Unknown subcommand "${sub}"\n${USAGE}`, "error");
      } catch (err) {
        emit(ctx, `llama-swappy: ${errorMessage(err)}`, "error");
      }
    },
  });
}

/** Normalize error text, including fetch timeout/abort names. */
function errorMessage(err: unknown): string {
  let message = err instanceof Error ? err.message : String(err);
  if (err instanceof Error && err.name === "TimeoutError") {
    message = `request timed out after ${REQUEST_TIMEOUT_MS / 1000}s`;
  } else if (err instanceof Error && err.name === "AbortError") {
    message = "request aborted";
  }
  return message;
}
