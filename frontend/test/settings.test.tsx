import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/app";
import type { EngineSettings, Transcript } from "../src/engine";
import { noReadings } from "../src/readings";
import { address, chatExtras, defaults, ENGINE_DEFAULT, load, save } from "../src/settings";

const SETTINGS: EngineSettings = {
  endpoint: { base_url: "http://localhost:8000/v1", model: "Qwen/Qwen3.5-4B", host: "localhost:8000", token_ids: true },
  source: "start", api_key: "none",
  contract: { name: "qwen3.5-4b", served_as: ["Qwen/Qwen3.5-4B", "qwen3.5-4b"], knows_model: true },
  rebuilds_ids: true, transcripts: "transcripts", kept_in: "transcripts/engine.json",
};

/** An engine and a readings server in memory, told apart by their ports. What was asked of each is kept. */
function world({ engine = SETTINGS as EngineSettings, readings = true } = {}) {
  const state = { engine: structuredClone(engine) };
  const asked: { url: string; method: string; body?: any }[] = [];
  const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const calls: Transcript[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    const at = new URL(url);
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    asked.push({ url, method: init?.method ?? "GET", ...(body !== undefined ? { body } : {}) });
    if (at.port === "8788") {
      if (!readings) throw new TypeError("Failed to fetch");
      if (at.pathname === "/health") return json(200, { ok: true, model: "qwen3.5-4b", revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", device: "cuda", tools: { emotions: 171, lens: "jpp", lens_ranked_by: "lift over a baseline" }, readings: 24 });
      return json(404, { error: "No such transcript." });
    }
    const e = state.engine.endpoint;
    if (at.pathname === "/health") return json(200, { ok: true, endpoint: e?.host ?? null, model: e?.model ?? null });
    if (at.pathname === "/conversations") return json(200, calls.length ? [{ id: "c1", turns: calls.length, created: "", updated: "", title: calls[0].request.messages.at(-1)!.content }] : []);
    if (at.pathname === "/settings" && init?.method === "PUT") {
      if (!body.model) return json(400, { error: "Say the name the endpoint knows the model by." });
      state.engine = { ...state.engine, endpoint: { ...body, host: new URL(body.base_url).host }, source: "page", contract: state.engine.contract && { ...state.engine.contract, knows_model: state.engine.contract.served_as.includes(body.model) } };
      return json(200, state.engine);
    }
    if (at.pathname === "/settings") return json(200, state.engine);
    if (at.pathname === "/endpoint/models") return body.base_url?.includes("9999") ? json(502, { error: "Could not reach localhost:9999: connection refused" }) : json(200, { host: new URL(body.base_url).host, models: ["Qwen/Qwen3.5-4B", "another-model"] });
    if (at.pathname === "/chat") {
      const t: Transcript = { id: `c1-${calls.length + 1}`, conversation: "c1", turn: calls.length + 1, created: "2026-11-18T09:00:00+00:00", endpoint: { host: "localhost:8000", served_model: "Qwen/Qwen3.5-4B" },
        request: { messages: [{ role: "user", content: body.message }] }, response: { content: `Reply to: ${body.message}` } };
      calls.push(t);
      return json(200, t);
    }
    return json(404, { error: "Not found." });
  });
  return { asked, state };
}

async function openSettings() {
  fireEvent.click(await screen.findByRole("button", { name: "Settings" }));
  return screen.findByRole("heading", { name: "Settings" });
}

describe("the settings page", () => {
  it("takes the conversation's place, says which endpoint the engine talks to, and gives the place back", async () => {
    world();
    render(<App readings={noReadings} />);
    await openSettings();
    expect(window.location.hash).toBe("#settings");
    expect(screen.queryByRole("textbox", { name: "Message" })).toBeNull();
    const part = screen.getByRole("region", { name: "Model endpoint" });
    expect(await within(part).findByRole("textbox", { name: "Endpoint address" })).toHaveValue("http://localhost:8000/v1");
    expect(within(part).getByRole("combobox", { name: "Model name" })).toHaveValue("Qwen/Qwen3.5-4B");
    expect(within(part).getByRole("checkbox", { name: "Ask the endpoint for the token ids it used" })).toBeChecked();
    expect(part).toHaveTextContent("Qwen/Qwen3.5-4B at localhost:8000. Given when the engine started.");
    expect(part).toHaveTextContent("A key is never typed into this page.");
    expect(part).toHaveTextContent("qwen3.5-4b, which knows this model's name.");
    expect(part).toHaveTextContent("the engine rebuilds them with the contract's tokenizer");
    // nothing has been changed, so there is nothing to save
    expect(within(part).getByRole("button", { name: "Use this endpoint" })).toBeDisabled();
    expect(screen.getByRole("region", { name: "The engine" })).toHaveTextContent("Answering.");
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.getByRole("textbox", { name: "Message" })).toBeVisible();
    expect(window.location.hash).toBe("");
  });

  it("opens on its own address, and a conversation picked from the history leaves it", async () => {
    world();
    window.history.replaceState(null, "", "#settings");
    render(<App readings={noReadings} />);
    expect(await screen.findByRole("heading", { name: "Settings" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Settings" })).toHaveAttribute("aria-current", "page");
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));
    expect(screen.queryByRole("heading", { name: "Settings" })).toBeNull();
  });

  it("finds an endpoint's models and points the engine at one", async () => {
    const { asked } = world();
    const user = userEvent.setup();
    render(<App readings={noReadings} />);
    await openSettings();
    const part = screen.getByRole("region", { name: "Model endpoint" });
    const where = await within(part).findByRole("textbox", { name: "Endpoint address" });
    await user.clear(where);
    await user.type(where, "http://localhost:9000/v1/");
    await user.click(within(part).getByRole("button", { name: "Find models" }));
    expect(await within(part).findByText("localhost:9000 answered. It serves: Qwen/Qwen3.5-4B, another-model.")).toBeVisible();
    expect(asked.at(-1)).toEqual({ url: `${ENGINE_DEFAULT}/endpoint/models`, method: "POST", body: { base_url: "http://localhost:9000/v1" } });
    const name = within(part).getByRole("combobox", { name: "Model name" });
    await user.clear(name);
    await user.type(name, "another-model");
    await user.click(within(part).getByRole("button", { name: "Use this endpoint" }));
    expect(await within(part).findByText("The engine now talks to localhost:9000 (another-model).")).toBeVisible();
    expect(asked.find((a) => a.method === "PUT")).toEqual({ url: `${ENGINE_DEFAULT}/settings`, method: "PUT", body: { base_url: "http://localhost:9000/v1", model: "another-model", token_ids: true } });
    // the page shows what the engine now says: where it came from, and that the contract does not know the name
    expect(part).toHaveTextContent("Set here, and kept in transcripts/engine.json for the engine's next start.");
    expect(part).toHaveTextContent("The qwen3.5-4b contract knows its weights as “Qwen/Qwen3.5-4B” or “qwen3.5-4b”, not as “another-model”. Readings of these conversations will be flagged.");
    await waitFor(() => expect(screen.getByRole("navigation", { name: "Conversations" })).toHaveTextContent("another-model · localhost:9000"));
    // an endpoint that cannot be reached, and an address that is not one, are said plainly
    await user.clear(where);
    await user.type(where, "http://localhost:9999/v1");
    await user.click(within(part).getByRole("button", { name: "Find models" }));
    expect(await within(part).findByRole("alert")).toHaveTextContent("Could not reach localhost:9999: connection refused");
    await user.clear(where);
    await user.type(where, "localhost");
    await user.click(within(part).getByRole("button", { name: "Use this endpoint" }));
    expect(within(part).getByRole("alert")).toHaveTextContent('The endpoint is an address like "http://localhost:8000/v1".');
  });

  it("says when the key is held back from an endpoint", async () => {
    world({ engine: { ...SETTINGS, source: "kept", api_key: "held back", contract: null, rebuilds_ids: false } });
    render(<App readings={noReadings} />);
    await openSettings();
    const part = screen.getByRole("region", { name: "Model endpoint" });
    expect(await within(part).findByText(/for another endpoint, so it is not sent to this one/)).toBeVisible();
    expect(part).toHaveTextContent("Kept from an earlier change, in transcripts/engine.json.");
    expect(part).toHaveTextContent("None. Start the engine with --contract qwen3.5-4b");
    expect(part).toHaveTextContent("a conversation is kept without them and the tools rebuild them");
  });

  it("asks for an endpoint when the engine has none, and chats once one is set", async () => {
    world({ engine: { ...SETTINGS, endpoint: null, source: null, contract: null } });
    const user = userEvent.setup();
    render(<App readings={noReadings} />);
    expect(await screen.findByRole("heading", { name: "No model endpoint yet." })).toBeVisible();
    expect(screen.getByRole("textbox", { name: "Message" })).toBeDisabled();
    expect(screen.getByRole("navigation", { name: "Conversations" })).toHaveTextContent("No model endpoint set");
    await user.click(screen.getByRole("button", { name: "Open settings" }));
    const part = await screen.findByRole("region", { name: "Model endpoint" });
    expect(await within(part).findByText(/No endpoint is set yet/)).toBeVisible();
    await user.type(within(part).getByRole("textbox", { name: "Endpoint address" }), "http://localhost:8000/v1");
    await user.click(within(part).getByRole("button", { name: "Find models" }));
    await within(part).findByText(/localhost:8000 answered/);
    expect(within(part).getByRole("combobox", { name: "Model name" })).toHaveValue("Qwen/Qwen3.5-4B");   // the first it serves
    await user.click(within(part).getByRole("button", { name: "Use this endpoint" }));
    await within(part).findByText("The engine now talks to localhost:8000 (Qwen/Qwen3.5-4B).");
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(await screen.findByRole("heading", { name: "Say something." })).toBeVisible();
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
  });

  it("sends a new conversation with the system message and the sampling settings", async () => {
    const { asked } = world();
    const user = userEvent.setup();
    render(<App readings={noReadings} />);
    await openSettings();
    const part = screen.getByRole("region", { name: "New conversations" });
    await user.type(within(part).getByRole("textbox", { name: "System message" }), "  Be brief.  ");
    await user.type(within(part).getByRole("spinbutton", { name: "Longest reply, in tokens" }), "120");
    await user.type(within(part).getByRole("spinbutton", { name: "Temperature" }), "0.7");
    await user.click(screen.getByRole("button", { name: "Done" }));
    const box = screen.getByRole("textbox", { name: "Message" });
    await user.type(box, "Hello{Enter}");
    await screen.findByText("Reply to: Hello");
    await user.type(box, "And then?{Enter}");
    await screen.findByText("Reply to: And then?");
    const sent = asked.filter((a) => a.url.endsWith("/chat")).map((a) => a.body);
    expect(sent).toEqual([
      { message: "Hello", system: "Be brief.", params: { max_tokens: 120, temperature: 0.7 } },
      { message: "And then?", conversation: "c1", params: { max_tokens: 120, temperature: 0.7 } },   // the system message opens a conversation only
    ]);
    // a number out of range is not taken, and says so; emptied, the endpoint chooses again
    await openSettings();
    const tokens = within(screen.getByRole("region", { name: "New conversations" })).getByRole("spinbutton", { name: "Longest reply, in tokens" });
    expect(tokens).toHaveValue(120);
    await user.clear(tokens);
    await user.type(tokens, "0");
    expect(tokens).toHaveAttribute("aria-invalid", "true");
    expect(load().maxTokens).toBeNull();
  });

  it("keeps the page's settings in this browser, uses them at once, and can go back to the defaults", async () => {
    const { asked } = world();
    const user = userEvent.setup();
    const view = render(<App readings={noReadings} />);
    await openSettings();
    const where = within(screen.getByRole("region", { name: "The engine" })).getByRole("textbox", { name: "Engine address" });
    await user.clear(where);
    await user.type(where, "not an address{Enter}");
    expect(where).toHaveAttribute("aria-invalid", "true");
    expect(load().engineUrl).toBe(ENGINE_DEFAULT);
    await user.clear(where);
    await user.type(where, "http://127.0.0.1:9000/{Enter}");
    await waitFor(() => expect(asked.some((a) => a.url === "http://127.0.0.1:9000/health")).toBe(true));
    expect(asked.some((a) => a.url === "http://127.0.0.1:9000/settings")).toBe(true);
    // only what differs from the defaults is written down
    expect(JSON.parse(localStorage.getItem("undertone.settings")!)).toEqual({ engineUrl: "http://127.0.0.1:9000" });
    // a page opened afresh starts from what was kept
    view.unmount();
    asked.length = 0;
    render(<App readings={noReadings} />);
    await waitFor(() => expect(asked.some((a) => a.url === "http://127.0.0.1:9000/health")).toBe(true));
    await screen.findByRole("heading", { name: "Settings" });   // (the address still says #settings: it opens there)
    await user.click(screen.getByRole("button", { name: "Back to defaults" }));
    expect(localStorage.getItem("undertone.settings")).toBeNull();
    await waitFor(() => expect(asked.some((a) => a.url === `${ENGINE_DEFAULT}/health`)).toBe(true));
  });

  it("chooses where readings come from and says what the readings server holds", async () => {
    world();
    const user = userEvent.setup();
    render(<App />);
    await openSettings();
    const part = screen.getByRole("region", { name: "Readings" });
    expect(within(part).getByRole("radio", { name: "The readings server" })).toBeChecked();
    expect(await within(part).findByText("qwen3.5-4b at revision 851bf6e806ef, on cuda.")).toBeVisible();
    expect(part).toHaveTextContent("171 emotion vectors.");
    expect(part).toHaveTextContent("jpp, its words ranked by lift over a baseline.");
    expect(part).toHaveTextContent("24 readings.");
    // samples: the page says they are made up, and a reply is then read by the sample source
    await user.click(within(part).getByRole("radio", { name: "Samples" }));
    expect(within(part).queryByRole("textbox", { name: "Readings server address" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Done" }));
    await user.type(screen.getByRole("textbox", { name: "Message" }), "Tell me about the quiet street{Enter}");
    expect(await screen.findByText(/· sample reading$/)).toBeVisible();
    // nowhere: no panel at all
    await openSettings();
    await user.click(within(screen.getByRole("region", { name: "Readings" })).getByRole("radio", { name: "Nowhere" }));
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("button", { name: "Inside" })).toBeNull();
    expect(screen.queryByText(/sample reading$/)).toBeNull();
    expect(load().readings).toBe("off");
  });

  it("says how to start the readings server when it is not answering", async () => {
    world({ readings: false });
    render(<App />);
    await openSettings();
    const part = screen.getByRole("region", { name: "Readings" });
    expect(await within(part).findByRole("alert")).toHaveTextContent("Not answering at http://127.0.0.1:8788.");
    expect(part).toHaveTextContent("undertone-server serve");
    expect(within(part).getByRole("button", { name: "Look again" })).toBeVisible();
  });
});

describe("the page's settings", () => {
  it("start from the usual addresses, or from what the page was started with", () => {
    expect(defaults({})).toEqual({ engineUrl: "http://127.0.0.1:8787", readings: "server", readingsUrl: "http://127.0.0.1:8788", system: "", maxTokens: null, temperature: null });
    expect(defaults({ VITE_ENGINE_URL: "http://127.0.0.1:9000/", VITE_READINGS: "http://gpu.local:8788/" })).toMatchObject({ engineUrl: "http://127.0.0.1:9000", readings: "server", readingsUrl: "http://gpu.local:8788" });
    expect(defaults({ VITE_READINGS: "sample" })).toMatchObject({ readings: "sample", readingsUrl: "http://127.0.0.1:8788" });
    expect([address("http://localhost:8000/v1/"), address(" https://a.example "), address("localhost:8000"), address("ftp://x"), address(""), address(undefined)]).toEqual(["http://localhost:8000/v1", "https://a.example", null, null, null, null]);
  });

  it("fall back to a default wherever what was kept cannot be used", () => {
    localStorage.setItem("undertone.settings", JSON.stringify({ engineUrl: "nonsense", readings: "everywhere", readingsUrl: 7, system: ["x"], maxTokens: -3, temperature: 9 }));
    expect(load({})).toEqual(defaults({}));
    localStorage.setItem("undertone.settings", "not json");
    expect(load({})).toEqual(defaults({}));
    save({ ...defaults({}), system: "Be brief.", maxTokens: 64.4, temperature: 0 }, {});
    expect(JSON.parse(localStorage.getItem("undertone.settings")!)).toEqual({ system: "Be brief.", maxTokens: 64.4, temperature: 0 });
    expect(load({})).toMatchObject({ system: "Be brief.", maxTokens: 64, temperature: 0 });
  });

  it("add nothing to a chat until something is set", () => {
    expect(chatExtras(defaults({}), true)).toEqual({});
    expect(chatExtras({ ...defaults({}), system: "  ", temperature: 0 }, true)).toEqual({ params: { temperature: 0 } });
    expect(chatExtras({ ...defaults({}), system: "Be brief.", maxTokens: 64 }, false)).toEqual({ params: { max_tokens: 64 } });
  });
});
