/** The page's connection for chat and history: the undertone engine's local service (core/engine, `undertone-engine serve`).
 * The page never talks to a model endpoint itself and never holds a key; the engine does both. What the page shows as
 * history is what the engine has kept as transcripts. */

/** One call to the endpoint, as the engine records it (the fields this page reads). */
export interface Transcript {
  id: string;
  conversation: string;
  turn: number;
  created: string;
  endpoint: { host: string; served_model: string };
  model?: string;
  request: { messages: { role: string; content: string }[] };
  response: { content: string; reasoning?: string; finish_reason?: string };
  /** the token ids of the call and where they came from; absent when there are none */
  tokens?: { prompt_ids: number[]; completion_ids?: number[]; source: "server" | "rebuilt" };
}

/** A conversation at a glance, for the history list. */
export interface Summary {
  id: string;
  turns: number;
  created: string;
  updated: string;
  title: string;
}

/** `endpoint` and `model` are null while the engine has no model endpoint to talk to. */
export interface Health {
  ok: boolean;
  endpoint: string | null;
  model: string | null;
}

/** Which model endpoint the engine talks to, and what it knows of the model: the engine's own settings. */
export interface EngineSettings {
  endpoint: { base_url: string; model: string; host: string; token_ids: boolean } | null;
  /** where the endpoint came from: given when the engine started, kept from an earlier change, or set from a page */
  source: "start" | "kept" | "page" | null;
  /** what becomes of the API key in the engine's environment: sent to this endpoint, held back from it, or there is none */
  api_key: "sent" | "held back" | "none";
  /** the model's contract among the undertone tools, when the engine was started with one */
  contract: { name: string; served_as: string[]; knows_model: boolean | null } | null;
  /** whether the engine rebuilds token ids itself when the endpoint returns none */
  rebuilds_ids: boolean;
  transcripts: string;
  kept_in: string;
}

/** What a chat may carry beside the message (settings.ts, `chatExtras`). */
export interface ChatExtras {
  system?: string;
  params?: Record<string, number>;
}

/** The engine could not be reached at all (as against answering with an error). */
export class EngineDown extends Error {}

/** The engine's requests, at the address it listens on. */
export function engineAt(base: string) {
  const root = base.replace(/\/+$/, "");
  async function call<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(root + path, init);
    } catch {
      throw new EngineDown("The engine is not answering.");
    }
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(body?.error ?? `The engine answered ${response.status}.`);
    return body as T;
  }
  const json = (method: string, body: unknown): RequestInit => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return {
    url: root,
    health: () => call<Health>("/health"),
    conversations: () => call<Summary[]>("/conversations"),
    conversation: (id: string) => call<Transcript[]>(`/conversations/${encodeURIComponent(id)}`),
    chat: (message: string, conversation: string | null, extras: ChatExtras = {}) =>
      call<Transcript>("/chat", json("POST", { message, ...(conversation ? { conversation } : {}), ...extras })),
    settings: () => call<EngineSettings>("/settings"),
    /** point the engine at another endpoint; answers the settings as they now are */
    setEndpoint: (endpoint: { base_url: string; model: string; token_ids: boolean }) => call<EngineSettings>("/settings", json("PUT", endpoint)),
    /** the names an endpoint knows its models by: the one in use, or one at an address being tried */
    models: (base_url?: string) => call<{ host: string; models: string[] }>("/endpoint/models", json("POST", base_url ? { base_url } : {})),
  };
}

export type EngineApi = ReturnType<typeof engineAt>;
