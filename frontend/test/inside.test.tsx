import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/app";
import { ReplyText } from "../src/inside";
import type { Transcript } from "../src/engine";
import { DEPTHS, fits, httpReadings, isReading, noReadings, ReadingsDown, sampleReading, sampleReadings, sourceFor, type Reading, type ReadingSource } from "../src/readings";
import { defaults, READINGS_DEFAULT } from "../src/settings";

const call = (id: string, turn: number, said: string, reply: string): Transcript => ({
  id, conversation: "abc", turn, created: "2026-11-18T09:00:00+00:00",
  endpoint: { host: "localhost:8000", served_model: "Qwen/Qwen3.5-4B" },
  request: { messages: [{ role: "user", content: said }] },
  response: { content: reply },
});
const FIRST = call("t1", 1, "What do you see?", "A quiet street under gray rain.");
const SECOND = call("t2", 2, "And at night?", "Streetlights flicker along the empty road.");

/** A reading written by hand, as a tool would return it: "quiet" and "rain" are marked in the first reply. */
const READING: Reading = {
  schema: "undertone.reading.v1", transcript: "t1", model: "qwen3.5-4b",
  overall: {
    emotions: [
      { name: "lonely", family: "Despair and Shame", level: 91 },
      { name: "calm", family: "Peaceful Contentment", level: 78 },
    ],
    lens: { early: ["street", "road", "lane"], middle: ["empty", "still", "alone"], late: ["rain", "gray", "cold"], output: ["falls", "and", "on"] },
  },
  tokens: [
    { text: "A" }, { text: " " },
    { text: "quiet", emotions: [{ name: "calm", family: "Peaceful Contentment", strength: 6 }, { name: "lonely", family: "Despair and Shame", strength: 3 }], lens: { early: ["hush", "soft", "low"], middle: ["still", "calm", "rest"], late: ["street", "night", "room"], output: ["street", "road", "house"] } },
    { text: " " }, { text: "street" }, { text: " " }, { text: "under" }, { text: " " }, { text: "gray" }, { text: " " },
    { text: "rain.", emotions: [{ name: "lonely", family: "Despair and Shame", strength: 8 }] },
  ],
};
const handWritten: ReadingSource = { label: "measured", reading: async (t) => (t.id === "t1" ? READING : null) };

/** An engine that has one conversation of two calls. */
function withConversation() {
  const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  vi.stubGlobal("fetch", async (url: string) => {
    const path = new URL(url).pathname;
    if (path === "/health") return json({ ok: true, endpoint: "localhost:8000", model: "Qwen/Qwen3.5-4B" });
    if (path === "/conversations") return json([{ id: "abc", turns: 2, created: FIRST.created, updated: SECOND.created, title: "What do you see?" }]);
    return json([FIRST, SECOND]);
  });
}
async function openConversation(source: ReadingSource) {
  withConversation();
  const view = render(<App readings={source} />);
  fireEvent.click(await screen.findByRole("button", { name: /What do you see\?/ }));
  await screen.findByText("And at night?");
  return view;
}

describe("a reading", () => {
  it("is made up repeatably by the sample source, in the real shape, and says it is a sample", () => {
    const a = sampleReading(FIRST), b = sampleReading(FIRST);
    expect(a).toEqual(b);
    expect(sampleReading(SECOND)).not.toEqual(a);
    expect(a.sample).toBe(true);
    expect(a.schema).toBe("undertone.reading.v1");
    expect(a.transcript).toBe("t1");
    // the pieces, joined, are the reply exactly
    expect(a.tokens.map((t) => t.text).join("")).toBe(FIRST.response.content);
    expect(a.overall.emotions.length).toBeGreaterThan(3);
    expect(a.overall.emotions.every((e) => e.level >= 0 && e.level <= 100 && e.family)).toBe(true);
    expect(Object.keys(a.overall.lens)).toEqual([...DEPTHS]);
    for (const t of a.tokens) {
      for (const e of t.emotions ?? []) expect(e.strength).toBeGreaterThanOrEqual(1), expect(e.strength).toBeLessThanOrEqual(9);
      expect(/\s/.test(t.text) && t.emotions ? "space marked" : "ok").toBe("ok");
    }
  });

  it("comes from the source the page is configured with", async () => {
    // the readings server at its usual address, unless the page is told otherwise
    expect(sourceFor(defaults({})).where).toBe(READINGS_DEFAULT);
    expect(sourceFor(defaults({})).label).toBe("measured");
    expect(sourceFor(defaults({ VITE_READINGS: "sample" }))).toBe(sampleReadings);
    expect(sourceFor(defaults({ VITE_READINGS: "off" }))).toBe(noReadings);
    expect(await noReadings.reading(FIRST)).toBeNull();
    // a service: one request per transcript; anything but a reading is "none"
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      asked.push(url);
      return url.endsWith("/t1") ? new Response(JSON.stringify(READING)) : new Response("{}", { status: 404 });
    });
    const service = sourceFor(defaults({ VITE_READINGS: "http://127.0.0.1:8788/" }));
    expect(service.label).toBe("measured");
    expect(service.where).toBe("http://127.0.0.1:8788");
    expect(await service.reading(FIRST)).toEqual(READING);
    expect(await service.reading(SECOND)).toBeNull();
    expect(asked).toEqual(["http://127.0.0.1:8788/readings/t1", "http://127.0.0.1:8788/readings/t2"]);
    vi.stubGlobal("fetch", async () => {
      throw new TypeError("Failed to fetch");
    });
    // a service that cannot be reached at all is not the same as one that has no reading
    await expect(httpReadings("http://127.0.0.1:8788").reading(FIRST)).rejects.toBeInstanceOf(ReadingsDown);
  });

  it("counts as none when a service answers something that is not a reading", async () => {
    expect(isReading(READING)).toBe(true);
    expect(isReading(sampleReading(FIRST))).toBe(true);
    const broken: unknown[] = [
      null, {}, { ...READING, schema: "undertone.reading.v2" }, { ...READING, tokens: undefined },
      { ...READING, overall: { emotions: READING.overall.emotions } }, { ...READING, tokens: [{ text: 3 }] },
      { ...READING, overall: { ...READING.overall, lens: { early: [], middle: [], late: [] } } },
    ];
    for (const body of broken) {
      expect(isReading(body)).toBe(false);
      vi.stubGlobal("fetch", async () => new Response(JSON.stringify(body)));
      expect(await httpReadings("http://127.0.0.1:8788").reading(FIRST)).toBeNull();
    }
  });
});

describe("Inside the model", () => {
  it("is a panel that slides out on the right, closed to begin with", async () => {
    const view = await openConversation(handWritten);
    const app = view.container.querySelector(".chat-app")!;
    const column = view.container.querySelector("#chat-inside")!;
    const toggle = screen.getByRole("button", { name: "Inside" });
    expect(app).not.toHaveClass("inside-open");
    expect(column).toHaveAttribute("inert");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(app).toHaveClass("inside-open");
    expect(column).not.toHaveAttribute("inert");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(screen.getByRole("button", { name: "Close Inside the model" }));
    expect(app).not.toHaveClass("inside-open");
    expect(column).toHaveAttribute("inert");
  });

  it("marks the words of a reply that has a reading and shows the whole reply's reading", async () => {
    await openConversation(handWritten);
    const log = screen.getByRole("log", { name: "Conversation" });
    // the first reply has a reading: two of its words are marked, in their emotion's family colour, and it reads as before
    const quiet = await within(log).findByRole("button", { name: "quiet: calm, lonely" });
    const rain = within(log).getByRole("button", { name: "rain.: lonely" });
    expect(quiet.style.getPropertyValue("--emo")).toBe("#8fcaa0");
    expect(rain.style.getPropertyValue("--emo")).toBe("#7f9cc4");
    expect(quiet.closest("p")).toHaveTextContent("A quiet street under gray rain.");
    // the second has none: plain text, and only the first says it has a reading
    expect(within(log).getByText("Streetlights flicker along the empty road.")).toBeVisible();
    expect(within(log).getAllByText(/· reading$/)).toHaveLength(1);

    // pressing a turn opens the panel on that reply, whole
    fireEvent.click(within(log).getByRole("button", { name: "Turn 1" }));
    expect(within(log).getByRole("button", { name: "Turn 1" })).toHaveAttribute("aria-pressed", "true");
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(panel).toHaveTextContent("Turn 1 · whole reply");
    expect(panel).toHaveTextContent("measured");
    const emotions = within(within(panel).getByRole("region", { name: "Emotions" })).getAllByRole("listitem");
    expect(emotions.map((li) => li.textContent)).toEqual(["1lonelyDespair91", "2calmContentment78"]);
    // the lens: the four depths side by side, each with its words down the column, the most leaned-toward first
    const depths = [...panel.querySelectorAll(".inside-lens > li")];
    expect(depths.map((li) => li.querySelector(".inside-depth")!.textContent)).toEqual(["early", "middle", "late", "output"]);
    expect(within(panel).getAllByRole("list", { name: "early words" })[0].querySelectorAll("li:not(.idle)")).toHaveLength(3);
    expect(depths.map((li) => [...li.querySelectorAll(".inside-words li:not(.idle)")].map((w) => w.textContent))).toEqual([
      ["street", "road", "lane"], ["empty", "still", "alone"], ["rain", "gray", "cold"], ["falls", "and", "on"],
    ]);
    expect(depths[0].querySelector(".inside-words li")).toHaveClass("lead");
    expect(panel).toHaveTextContent("They are not evidence that it feels anything.");
    // the panel always has the same rows, whatever it shows: six emotions, six words a depth
    expect(panel.querySelectorAll(".inside-emotions li")).toHaveLength(6);
    for (const d of depths) expect(d.querySelectorAll(".inside-words li")).toHaveLength(6);
  });

  it("turns to a word while it is pointed at, and keeps to one that is pressed", async () => {
    await openConversation(handWritten);
    const quiet = await screen.findByRole("button", { name: "quiet: calm, lonely" });
    const rain = screen.getByRole("button", { name: "rain.: lonely" });
    fireEvent.click(screen.getByRole("button", { name: "Inside" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    // (the newest reply has no reading; pointing at a word of the first one brings its reading up)
    expect(panel).toHaveTextContent("No reading for this reply.");
    fireEvent.pointerEnter(quiet);
    expect(panel).toHaveTextContent("Turn 1 · “quiet”");
    expect(within(within(panel).getByRole("region", { name: "Emotions" })).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["1calmContentment6/9", "2lonelyDespair3/9"]);
    expect([...panel.querySelectorAll('[aria-label="early words"] li:not(.idle)')].map((w) => w.textContent)).toEqual(["hush", "soft", "low"]);
    fireEvent.pointerLeave(quiet);
    expect(panel).toHaveTextContent("No reading for this reply.");

    // pressed, it stays; a word with no lens of its own shows none; "Whole reply" lets go
    fireEvent.click(rain);
    fireEvent.pointerLeave(rain);
    expect(rain).toHaveAttribute("aria-pressed", "true");
    expect(panel).toHaveTextContent("Turn 1 · “rain.”");
    expect(panel.querySelectorAll(".inside-words li:not(.idle)")).toHaveLength(0);
    expect(panel.querySelector('[aria-label="early words"]')).toHaveTextContent(/^—$/);
    fireEvent.click(within(panel).getByRole("button", { name: "Whole reply" }));
    expect(rain).toHaveAttribute("aria-pressed", "false");
    expect(panel).toHaveTextContent("No reading for this reply.");
  });

  it("lays the conversation out turn by turn at its foot, and a turn pressed there shows that reply", async () => {
    await openConversation(sampleReadings);
    fireEvent.click(screen.getByRole("button", { name: "Inside" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    const over = await within(panel).findByRole("region", { name: "Over the turns" });
    const first = sampleReading(FIRST).overall.emotions[0], second = sampleReading(SECOND).overall.emotions[0];
    // one column a turn, each named for its reply's strongest emotion; the newest reply is the one on show
    const turns = within(over).getAllByRole("button");
    expect(turns.map((b) => b.getAttribute("aria-label"))).toEqual([`Turn 1: ${first.name}, ${first.level}`, `Turn 2: ${second.name}, ${second.level}`]);
    expect(turns[1]).toHaveAttribute("aria-current", "step");
    expect(over).toHaveTextContent(second.name);
    expect(panel).toHaveTextContent("Turn 2 · whole reply");
    // a bar as tall as the emotion stands out
    expect(turns[0].style.getPropertyValue("--share")).toBe((first.level / 100).toFixed(3));
    fireEvent.click(turns[0]);
    expect(turns[0]).toHaveAttribute("aria-current", "step");
    expect(panel).toHaveTextContent("Turn 1 · whole reply");
    expect(over).toHaveTextContent(first.name);
    // it stands at the panel's foot, after everything else
    expect(panel.lastElementChild).toContainElement(over);
  });

  it("lets any turn of the conversation be pressed, and then shows the timeline up to that turn", async () => {
    const view = await openConversation(sampleReadings);
    const app = view.container.querySelector(".chat-app")!;
    await screen.findAllByText(/· sample reading$/);
    const replies = [...view.container.querySelectorAll<HTMLElement>(".chat-turns > li")];
    expect(replies.every((li) => li.classList.contains("pickable"))).toBe(true);
    expect(app).not.toHaveClass("inside-open");

    // pressing the first turn (anywhere on it) slides the panel out on it
    fireEvent.click(replies[0].querySelector(".chat-user")!);
    expect(app).toHaveClass("inside-open");
    expect(replies[0]).toHaveClass("shown");
    expect(replies[1]).not.toHaveClass("shown");
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(panel).toHaveTextContent("Turn 1 · whole reply");
    // the timeline reads up to turn 1: it is the current one, and the turn after it stands back
    const over = within(panel).getByRole("region", { name: "Over the turns" });
    const columns = [...over.querySelectorAll("ol > li")];
    expect(columns.map((li) => li.className)).toEqual(["cur", "ahead"]);
    expect(over).toHaveTextContent("turn 1 of 2");
    // a turn that stands back can still be pressed, to move on to it; the newest shows the whole conversation
    fireEvent.click(within(over).getAllByRole("button")[1]);
    expect(columns.map((li) => li.className)).toEqual(["", "cur"]);
    expect(over).toHaveTextContent("turns 1–2");
    expect(over).not.toHaveTextContent("of 2");
    expect(replies[1]).toHaveClass("shown");
    // the button under a turn does the same as pressing the turn
    fireEvent.click(screen.getByRole("button", { name: "Turn 1" }));
    expect(panel).toHaveTextContent("Turn 1 · whole reply");
    // a press that lands on a word is the word's own: it does not also pick the turn's whole reply
    const word = replies[1].querySelector<HTMLElement>(".tok")!;
    fireEvent.click(word);
    expect(panel).toHaveTextContent(`Turn 2 · “${word.textContent!.trim()}”`);
  });

  it("leaves turns alone when readings are off: there is no panel to show them in", async () => {
    const view = await openConversation(noReadings);
    const reply = view.container.querySelector<HTMLElement>(".chat-turns > li")!;
    expect(reply).not.toHaveClass("pickable");
    expect(screen.queryByRole("button", { name: "Turn 1" })).toBeNull();
    fireEvent.click(reply);
    expect(view.container.querySelector(".chat-app")).not.toHaveClass("inside-open");
  });

  it("keeps a turn without a reading in its place on the timeline, empty", async () => {
    await openConversation(handWritten);
    fireEvent.click(screen.getByRole("button", { name: "Inside" }));
    const over = await screen.findByRole("region", { name: "Over the turns" });
    const turns = within(over).getAllByRole("button");
    expect(turns.map((b) => b.getAttribute("aria-label"))).toEqual(["Turn 1: lonely, 91", "Turn 2: no reading"]);
    expect(turns[1]).toBeDisabled();
  });

  it("labels sample readings as samples, on the reply and in the panel", async () => {
    await openConversation(sampleReadings);
    expect(await screen.findAllByText(/· sample reading$/)).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Turn 2" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(within(panel).getByText("Sample")).toHaveClass("sample");
    expect(panel).toHaveTextContent("They are not measurements of the model.");
  });

  it("is put away by Escape", async () => {
    const view = await openConversation(handWritten);
    const app = view.container.querySelector(".chat-app")!;
    fireEvent.click(screen.getByRole("button", { name: "Inside" }));
    expect(app).toHaveClass("inside-open");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(app).not.toHaveClass("inside-open");
    expect(screen.getByRole("button", { name: "Inside" })).toHaveAttribute("aria-expanded", "false");
  });

  it("says a reply is being read until its reading arrives, and asks for each reply once", async () => {
    const asked: string[] = [];
    let arrive!: (r: Reading | null) => void;
    const slow: ReadingSource = {
      label: "measured",
      reading: (t) => {
        asked.push(t.id);
        return t.id === "t1" ? new Promise((done) => (arrive = done)) : Promise.resolve(null);
      },
    };
    await openConversation(slow);
    fireEvent.click(screen.getByRole("button", { name: "Turn 1" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(panel).toHaveTextContent("Reading this reply…");
    await act(async () => arrive(READING));
    expect(panel).toHaveTextContent("Turn 1 · whole reply");
    expect(asked).toEqual(["t1", "t2"]);
  });

  it("keeps the transcript's own words when a reading's pieces are not the reply", async () => {
    const off: Reading = { ...READING, tokens: [{ text: "Something else entirely.", emotions: [{ name: "calm", family: "Peaceful Contentment", strength: 5 }] }] };
    expect(fits(READING, FIRST)).toBe(true);
    expect(fits(off, FIRST)).toBe(false);
    const view = await openConversation({ label: "measured", reading: async (t) => (t.id === "t1" ? off : null) });
    expect(await screen.findAllByText(/· reading$/)).toHaveLength(1);
    expect(screen.getByText("A quiet street under gray rain.")).toBeVisible();
    expect(screen.queryByText("Something else entirely.")).toBeNull();
    expect(view.container.querySelector(".tok")).toBeNull();
    // the whole reply's reading is still there to show
    fireEvent.click(screen.getByRole("button", { name: "Turn 1" }));
    expect(screen.getByRole("complementary", { name: "Inside the model" })).toHaveTextContent("Turn 1 · whole reply");
  });

  it("says when the readings server is not answering, and asks again when told to", async () => {
    const asked: string[] = [];
    let up = false;
    const service: ReadingSource = {
      label: "measured",
      where: "http://127.0.0.1:8788",
      reading: async (t) => {
        asked.push(t.id);
        if (!up) throw new ReadingsDown("The readings server is not answering.");
        return t.id === "t1" ? READING : null;
      },
    };
    await openConversation(service);
    fireEvent.click(screen.getByRole("button", { name: "Turn 1" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(await within(panel).findByText("The readings server is not answering.")).toBeVisible();
    expect(panel).toHaveTextContent("undertone-server serve");
    expect(panel).toHaveTextContent("Looking for it at http://127.0.0.1:8788.");
    expect(screen.queryByText(/· reading$/)).toBeNull();
    expect(asked).toEqual(["t1", "t2"]);
    // started since: asked again, both replies are read and the notice goes
    up = true;
    fireEvent.click(within(panel).getByRole("button", { name: "Try again" }));
    expect(await within(panel).findByText(/Turn 1 · whole reply/)).toBeVisible();
    expect(within(panel).queryByText("The readings server is not answering.")).toBeNull();
    expect(asked).toEqual(["t1", "t2", "t1", "t2"]);
  });

  it("says when a reply was read outside the setting the model was checked in", async () => {
    const flagged: Reading = { ...READING, qualified: false, flags: ["ids_differ_from_template", "thinking_on"] };
    await openConversation({ label: "measured", reading: async (t) => (t.id === "t1" ? flagged : null) });
    fireEvent.click(await screen.findByRole("button", { name: "Turn 1" }));
    const panel = screen.getByRole("complementary", { name: "Inside the model" });
    expect(panel).toHaveTextContent("This reply lies outside the setting the model was checked in (ids differ from template, thinking on), so read it with more care.");
    // a reading inside it, or one that does not say, carries no such line
    expect(isReading(flagged)).toBe(true);
  });

  it("keeps the pieces of one word together and leaves the space between words outside them", async () => {
    const reply = "My heart is aching, still.\nYes.";
    const mark = [{ name: "lonely", family: "Despair and Shame", strength: 4 }];
    const split: Reading = {
      ...READING,
      tokens: [{ text: "My" }, { text: " heart", emotions: mark }, { text: " is" }, { text: " a", emotions: mark }, { text: "ching", emotions: mark }, { text: "," }, { text: " still" }, { text: ".\n" }, { text: "Yes" }, { text: "." }],
    };
    const view = render(<ReplyText reading={split} active={null} onPoint={() => {}} onPin={() => {}} />);
    const marked = view.container.querySelector(".chat-reply")!;
    // the text is the reply to the character, whatever the grouping
    expect(marked.textContent).toBe(reply);
    // "a" + "ching" + "," are one word; so are "still" + ".", and "Yes" + "." after the line break
    expect([...marked.querySelectorAll(".word")].map((w) => w.textContent)).toEqual(["aching,", "still.\n", "Yes."]);
    // a marked piece is pressed without its leading space, and is named without it
    expect(screen.getByRole("button", { name: "heart: lonely" }).textContent).toBe("heart");
    expect(screen.getByRole("button", { name: "a: lonely" }).textContent).toBe("a");
  });

  it("is not there at all when readings are off", async () => {
    const view = await openConversation(noReadings);
    expect(screen.queryByRole("button", { name: "Inside" })).toBeNull();
    expect(view.container.querySelector("#chat-inside")).toBeNull();
    expect(view.container.querySelector(".tok")).toBeNull();
  });
});
