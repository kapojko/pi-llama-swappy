/**
 * Instance-list persistence for pi-llama-swappy.
 *
 * The main provider ("llama-swappy") is implicit and never stored here.
 * Extra instances are stored in <agent-dir>/llama-swappy.json:
 *
 *   { "instances": [ { "name": "local", "endpoint": "http://127.0.0.1:12400", "apiKey": "..." } ] }
 *
 * Endpoints are stored normalized (with /v1 suffix). API keys are optional;
 * they may also be configured via `/login llama-swappy-<name>`, which stores
 * them in auth.json and takes precedence over this file.
 */
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

/** The implicit, always-registered main provider. */
export const PROVIDER_ID = "llama-swappy";
const CONFIG_FILE = "llama-swappy.json";
/** Display/selector names that may not be used for extra instances. */
export const RESERVED_NAMES = new Set(["main", "all"]);

export interface SwappyInstance {
  name: string;
  endpoint: string;
  apiKey?: string;
}

/** Provider ID for an extra instance: "llama-swappy-<name>". */
export function providerIdFor(name: string): string {
  return `${PROVIDER_ID}-${name}`;
}

/** Instance name for a provider ID, or undefined for the main provider. */
export function instanceNameForProvider(providerId: string): string | undefined {
  if (providerId === PROVIDER_ID) return undefined;
  return providerId.startsWith(`${PROVIDER_ID}-`)
    ? providerId.slice(PROVIDER_ID.length + 1)
    : undefined;
}

/** Validate a user-supplied instance name. Returns an error message or undefined. */
export function validateName(
  name: string,
  existing: readonly SwappyInstance[],
): string | undefined {
  const trimmed = name.trim();
  if (!trimmed) return "Instance name is required.";
  if (trimmed.length > 32) return "Instance name must be at most 32 characters.";
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(trimmed)) {
    return "Instance name may contain letters, digits, '_' and '-', and must start with a letter or digit.";
  }
  if (RESERVED_NAMES.has(trimmed.toLowerCase())) {
    return `"${trimmed}" is a reserved name.`;
  }
  if (existing.some((i) => i.name.toLowerCase() === trimmed.toLowerCase())) {
    return `Instance "${trimmed}" already exists.`;
  }
  return undefined;
}

/** Resolve pi's agent directory the same way pi's config.js does. */
export function getAgentDir(): string {
  const fromEnv = process.env.PI_CODING_AGENT_DIR?.trim();
  if (fromEnv) return path.resolve(fromEnv);
  return path.join(os.homedir(), ".pi", "agent");
}

export function configFilePath(): string {
  return path.join(getAgentDir(), CONFIG_FILE);
}

/** Load extra instances from the config file. Tolerates a missing or broken file. */
export function loadInstances(): SwappyInstance[] {
  try {
    const raw = fs.readFileSync(configFilePath(), "utf8");
    const parsed = JSON.parse(raw) as { instances?: unknown };
    if (!Array.isArray(parsed.instances)) return [];
    const seen = new Set<string>();
    const instances: SwappyInstance[] = [];
    for (const entry of parsed.instances) {
      const instance = coerceInstance(entry);
      if (!instance) continue;
      const key = instance.name.toLowerCase();
      if (seen.has(key) || RESERVED_NAMES.has(key)) continue;
      seen.add(key);
      instances.push(instance);
    }
    return instances;
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") {
      console.error(`llama-swappy: could not read ${configFilePath()}: ${err}`);
    }
    return [];
  }
}

/** Persist extra instances to the config file (atomically, creating the dir). */
export function saveInstances(instances: readonly SwappyInstance[]): void {
  const file = configFilePath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const payload = JSON.stringify(
    { instances: instances.map((i) => ({ ...i, apiKey: i.apiKey || undefined })) },
    null,
    2,
  );
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, payload, "utf8");
  fs.renameSync(tmp, file);
}

function coerceInstance(entry: unknown): SwappyInstance | undefined {
  if (typeof entry !== "object" || entry === null) return undefined;
  const { name, endpoint, apiKey } = entry as Record<string, unknown>;
  if (typeof name !== "string" || typeof endpoint !== "string") return undefined;
  if (!name.trim() || !endpoint.trim()) return undefined;
  return {
    name: name.trim(),
    endpoint: endpoint.trim(),
    apiKey: typeof apiKey === "string" && apiKey.trim() ? apiKey.trim() : undefined,
  };
}
