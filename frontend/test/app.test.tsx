import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/app";
import type { Summary, Transcript } from "../src/engine";
import { noReadings } from "../src/readings";

/** An engine in memory: it answers the page's four requests and keeps what it was asked. */
function fakeEngine({ down = false, chatError = null as string | null, source = "server" as "server" | "rebuilt" | null } = {}) {
  const kept: Record<string, Transcript[]> = {
    abc: [
      call("abc", 1, "What do you see?", "A quiet street."),
      { ...call("abc", 2, "And at night?", "Streetlights."), response: { content: "Streetlights.", reasoning: "It is dark then." } },
    ],
  };
  const asked: { path: string; body?: unknown }[] = [];
  function call(conversation: string, turn: number, said: string, reply: string): Transcript {
    return {
      id: `${conversation}-${turn}`, conversation, turn, created: "2026-11-18T09:00:00+00:00",
      endpoint: { host: "localhost:8000", served_model: "Qwen/Qwen3.5-4B" },
      request: { messages: [{ role: "user", content: said }] },
      response: { content: reply },
      ...(source ? { tokens: { prompt_ids: [1], completion_ids: [2], source } } : {}),
    };
  }
  const summaries = (): Summary[] =>
    Object.entries(kept).map(([id, calls]) => ({ id, turns: calls.length, created: calls[0].created, updated: calls.at(-1)!.created, title: calls[0].request.messages[0].content }));
  const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const state = { down };
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    const path = new URL(url).pathname;
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    asked.push({ path, body });
    if (state.down) throw new TypeError("Failed to fetch");
    if (path === "/health") return json(200, { ok: true, endpoint: "localhost:8000", model: "Qwen/Qwen3.5-4B" });
    if (path === "/conversations") return json(200, summaries());
    if (path.startsWith("/conversations/")) return json(200, kept[path.split("/")[2]] ?? []);
    if (path === "/chat") {
      if (chatError) return json(502, { error: chatError });
      const id = body.conversation ?? "new1";
      const t = call(id, (kept[id]?.length ?? 0) + 1, body.message, `Reply to: ${body.message}`);
      (kept[id] ??= []).push(t);
      return json(200, t);
    }
    return json(404, { error: "Not found." });
  });
  return { asked, state };
}

// (readings are switched off here: this file is the plain chat. test/inside.test.tsx covers the readings.)
describe("the chat page", () => {
  it("shows which endpoint the engine talks to and the conversations it has kept", async () => {
    fakeEngine();
    render(<App readings={noReadings} />);
    expect(await screen.findByRole("status")).toHaveTextContent("Qwen/Qwen3.5-4B · localhost:8000");
    const history = screen.getByRole("navigation", { name: "Conversations" });
    const item = await within(history).findByRole("button", { name: /What do you see\?/ });
    expect(item).toHaveTextContent("2 turns");
    // a new conversation to begin with
    expect(screen.getByRole("heading", { name: "Say something." })).toBeVisible();

    // opening one shows every call: what was said, the reply, and its reasoning folded away
    fireEvent.click(item);
    expect(item).toHaveAttribute("aria-current", "true");
    const log = screen.getByRole("log", { name: "Conversation" });
    expect(await within(log).findByText("A quiet street.")).toBeVisible();
    expect(within(log).getByText("And at night?")).toBeVisible();
    expect(within(log).getByText("Reasoning").closest("details")).not.toHaveAttribute("open");
    expect(within(log).getAllByText("token ids from the endpoint")).toHaveLength(2);
  });

  it("sends a message through the engine, shows the reply and keeps the conversation", async () => {
    const { asked } = fakeEngine({ source: "rebuilt" });
    const user = userEvent.setup();
    render(<App readings={noReadings} />);
    await screen.findByRole("status");
    const box = screen.getByRole("textbox", { name: "Message" });
    const send = screen.getByRole("button", { name: "Send" });
    expect(send).toBeDisabled();   // nothing to send yet

    await user.type(box, "Hello there{Enter}");
    const log = screen.getByRole("log", { name: "Conversation" });
    expect(await within(log).findByText("Reply to: Hello there")).toBeVisible();
    expect(within(log).getByText("Hello there")).toBeVisible();
    expect(box).toHaveValue("");
    // a new conversation was started (no id sent), and the reply says where its token ids came from
    expect(asked.find((a) => a.path === "/chat")!.body).toEqual({ message: "Hello there" });
    expect(within(log).getByText("token ids rebuilt by the engine")).toHaveClass("rebuilt");
    // it joins the history and is the one on show
    const history = screen.getByRole("navigation", { name: "Conversations" });
    await waitFor(() => expect(within(history).getByRole("button", { name: /Hello there/ })).toHaveAttribute("aria-current", "true"));

    // the next message carries the conversation on; Shift+Enter makes a new line instead of sending
    await user.type(box, "line one{Shift>}{Enter}{/Shift}line two{Enter}");
    await within(log).findByText(/Reply to: line one/);
    expect(asked.filter((a) => a.path === "/chat")[1].body).toEqual({ message: "line one\nline two", conversation: "new1" });

    // New conversation clears the page
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));
    expect(screen.getByRole("heading", { name: "Say something." })).toBeVisible();
  });

  it("says so when a call has no token ids", async () => {
    fakeEngine({ source: null });
    render(<App readings={noReadings} />);
    fireEvent.click(await screen.findByRole("button", { name: /What do you see\?/ }));
    expect(await screen.findAllByText("no token ids")).toHaveLength(2);
  });

  it("keeps the message and shows the endpoint's error when a call fails", async () => {
    fakeEngine({ chatError: "localhost:8000 answered 503: the server is unwell" });
    const user = userEvent.setup();
    render(<App readings={noReadings} />);
    await screen.findByRole("status");
    const box = screen.getByRole("textbox", { name: "Message" });
    await user.type(box, "Hello{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("localhost:8000 answered 503: the server is unwell");
    expect(box).toHaveValue("Hello");   // nothing was kept: the words are back in the box
    expect(screen.queryByText("Reply to: Hello")).toBeNull();
  });

  it("shows the conversation asked for last, whichever answers first", async () => {
    // two conversations; the first one opened answers late, after the second is already on show
    const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    const one = (id: string, said: string, reply: string): Transcript => ({
      id: `${id}-1`, conversation: id, turn: 1, created: "2026-11-18T09:00:00+00:00",
      endpoint: { host: "localhost:8000", served_model: "Qwen/Qwen3.5-4B" },
      request: { messages: [{ role: "user", content: said }] }, response: { content: reply },
    });
    let late!: () => void;
    vi.stubGlobal("fetch", async (url: string) => {
      const path = new URL(url).pathname;
      if (path === "/health") return json({ ok: true, endpoint: "localhost:8000", model: "Qwen/Qwen3.5-4B" });
      if (path === "/conversations") return json([
        { id: "slow", turns: 1, created: "", updated: "", title: "The slow one" },
        { id: "fast", turns: 1, created: "", updated: "", title: "The fast one" },
      ]);
      if (path === "/conversations/slow") await new Promise<void>((done) => (late = done));
      return json([path.endsWith("slow") ? one("slow", "Slow?", "Slow reply.") : one("fast", "Fast?", "Fast reply.")]);
    });
    render(<App readings={noReadings} />);
    fireEvent.click(await screen.findByRole("button", { name: /The slow one/ }));
    fireEvent.click(screen.getByRole("button", { name: /The fast one/ }));
    expect(await screen.findByText("Fast reply.")).toBeVisible();
    await act(async () => late());
    expect(screen.getByText("Fast reply.")).toBeVisible();
    expect(screen.queryByText("Slow reply.")).toBeNull();
  });

  it("explains how to start the engine when it is not running, and finds it on Try again", async () => {
    const { state } = fakeEngine({ down: true });
    render(<App readings={noReadings} />);
    expect(await screen.findByRole("heading", { name: "The engine is not running." })).toBeVisible();
    expect(screen.getByText(/undertone-engine serve/)).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Engine not running");
    expect(screen.getByRole("textbox", { name: "Message" })).toBeDisabled();
    state.down = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Say something." })).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Qwen/Qwen3.5-4B · localhost:8000");
  });
});
