/** The page's own settings: where it finds the engine, where readings come from, and what a new conversation is
 * sent with. They are kept in this browser (localStorage) and nowhere else.
 *
 * Which model endpoint the engine talks to is not one of them: that is the engine's, asked for and changed through
 * the engine (engine.ts), so the command line and every page agree on it. An API key is not a setting at all: it
 * stays in the engine's environment and never reaches the page. */

export type ReadingsFrom = "server" | "sample" | "off";

export interface PageSettings {
  /** where the engine listens */
  engineUrl: string;
  /** measured readings from the readings server, made-up samples, or none */
  readings: ReadingsFrom;
  /** where the readings server listens */
  readingsUrl: string;
  /** sent first in each new conversation; empty for none */
  system: string;
  /** the most tokens a reply may run to; null leaves it to the endpoint */
  maxTokens: number | null;
  /** how freely the endpoint samples; null leaves it to the endpoint */
  temperature: number | null;
}

export const ENGINE_DEFAULT = "http://127.0.0.1:8787";
export const READINGS_DEFAULT = "http://127.0.0.1:8788";
const KEY = "undertone.settings";

/** What the page starts with: the usual addresses, unless it was built or started with others (VITE_ENGINE_URL, and
 * VITE_READINGS: an address, "sample" or "off"). */
export function defaults(env: Record<string, string | undefined> = import.meta.env): PageSettings {
  const readings = env.VITE_READINGS;
  return {
    engineUrl: address(env.VITE_ENGINE_URL) ?? ENGINE_DEFAULT,
    readings: readings === "sample" || readings === "off" ? readings : "server",
    readingsUrl: (readings && readings !== "sample" && readings !== "off" ? address(readings) : null) ?? READINGS_DEFAULT,
    system: "",
    maxTokens: null,
    temperature: null,
  };
}

/** An http(s) address without its trailing slash, or null when the text is not one. */
export function address(text: string | undefined | null): string | null {
  try {
    const url = new URL((text ?? "").trim());
    return url.protocol === "http:" || url.protocol === "https:" ? url.href.replace(/\/+$/, "") : null;
  } catch {
    return null;
  }
}

/** What was kept, checked piece by piece: anything missing or unusable falls back to its default, so settings
 * written by an older page (or by hand) can never leave this one unable to start. */
export function load(env?: Record<string, string | undefined>): PageSettings {
  const base = defaults(env);
  let kept: Partial<Record<keyof PageSettings, unknown>> = {};
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    if (parsed && typeof parsed === "object") kept = parsed as typeof kept;
  } catch {
    /* nothing kept, or storage is closed to this page */
  }
  const number = (v: unknown, least: number, most: number) => (typeof v === "number" && Number.isFinite(v) && v >= least && v <= most ? v : null);
  const tokens = number(kept.maxTokens, 1, 1_000_000);
  return {
    engineUrl: address(kept.engineUrl as string) ?? base.engineUrl,
    readings: kept.readings === "server" || kept.readings === "sample" || kept.readings === "off" ? kept.readings : base.readings,
    readingsUrl: address(kept.readingsUrl as string) ?? base.readingsUrl,
    system: typeof kept.system === "string" ? kept.system : base.system,
    maxTokens: tokens === null ? null : Math.round(tokens),
    temperature: number(kept.temperature, 0, 2),
  };
}

/** Keep the settings in this browser. Only what differs from the defaults is written, so a default that changes
 * later reaches a page that never chose otherwise. */
export function save(settings: PageSettings, env?: Record<string, string | undefined>): void {
  const base = defaults(env);
  const changed = Object.fromEntries(Object.entries(settings).filter(([k, v]) => v !== base[k as keyof PageSettings]));
  try {
    if (Object.keys(changed).length) localStorage.setItem(KEY, JSON.stringify(changed));
    else localStorage.removeItem(KEY);
  } catch {
    /* storage is closed to this page: the settings last until it is closed */
  }
}

/** What a chat request carries beside the message: the system message when the conversation is new, and whichever
 * sampling settings were set. Nothing at all when nothing was. */
export function chatExtras(settings: PageSettings, starting: boolean): { system?: string; params?: Record<string, number> } {
  const params: Record<string, number> = {};
  if (settings.maxTokens !== null) params.max_tokens = settings.maxTokens;
  if (settings.temperature !== null) params.temperature = settings.temperature;
  return {
    ...(starting && settings.system.trim() ? { system: settings.system.trim() } : {}),
    ...(Object.keys(params).length ? { params } : {}),
  };
}
