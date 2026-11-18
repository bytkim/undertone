import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { engineAt, EngineDown, type Health, type Summary, type Transcript } from "./engine";
import { InsidePanel, ReplyText } from "./inside";
import { fits, noReadings, ReadingsDown, sourceFor, type Reading, type ReadingSource } from "./readings";
import { chatExtras, load as loadSettings, save as saveSettings, type PageSettings } from "./settings";
import { SettingsPage } from "./settingspage";

/** undertone's chat page: one conversation in the middle, the conversations kept so far down the side.
 *
 * It is a face for the engine and nothing more. A message goes to the engine, which sends it to the endpoint the user
 * brought and keeps the call as a transcript; the reply shown here is that transcript. The history is the engine's
 * own list of transcripts, so a conversation started from the command line appears here too, and the other way round.
 *
 * Under each reply a small line says where the call's token ids came from (the endpoint, rebuilt by the engine, or
 * none), since that is what the undertone tools read.
 *
 * "Inside the model" (inside.tsx) is a panel that slides out on the right: the reading of one reply. It is closed
 * to begin with; "Inside" in the header, pressing a turn or pressing a marked word opens it (Escape closes it), and
 * the conversation narrows beside it instead of being covered. Readings come through a `ReadingSource` (readings.ts):
 * the readings server unless the page is told otherwise; the page only draws what a source gives it. A reply with a
 * reading has its words marked (so long as the reading's pieces are the reply, word for word), and pointing at a
 * marked word turns the open panel to it. Every turn can be pressed: the panel then shows that turn, and its timeline
 * the conversation up to it.
 *
 * "Settings", at the foot of the history, takes the conversation's place with the settings page (settingspage.tsx):
 * which model endpoint the engine talks to, what a new conversation is sent with, where readings come from, and
 * where the engine is. The page's own settings (settings.ts) are kept in this browser and take effect at once. */
export function App({ readings: given }: { /** where readings come from (default: as the settings say) */ readings?: ReadingSource } = {}) {
  const [settings, setSettings] = useState<PageSettings>(loadSettings);
  const api = useMemo(() => engineAt(settings.engineUrl), [settings.engineUrl]);
  const source = useMemo(() => given ?? sourceFor({ readings: settings.readings, readingsUrl: settings.readingsUrl }), [given, settings.readings, settings.readingsUrl]);
  const panel = source !== noReadings;   // there are readings to ask for, so there is a panel to show them in
  const [view, setView] = useState<"chat" | "settings">(() => (window.location.hash === "#settings" ? "settings" : "chat"));
  const [health, setHealth] = useState<Health | "checking" | "down">("checking");
  const [history, setHistory] = useState<Summary[]>([]);
  const [open, setOpen] = useState<string | null>(null);   // the conversation on show; null is a new one
  const [calls, setCalls] = useState<Transcript[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState<string | null>(null);   // the message on its way
  const [error, setError] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);   // the history drawer on a narrow screen
  const [inside, setInside] = useState(false);   // the Inside panel is out
  // each reply's reading by transcript id: a Reading, null where there is none, absent while it is being fetched
  const [read, setRead] = useState<Record<string, Reading | null>>({});
  const [away, setAway] = useState(false);   // the readings service is not answering
  const [again, setAgain] = useState(0);   // counts the times the service was asked again by hand
  const [chosen, setChosen] = useState<string | null>(null);   // the reply picked for the panel (default: the newest)
  const [pointed, setPointed] = useState<{ id: string; token: number } | null>(null);
  const [pinned, setPinned] = useState<{ id: string; token: number } | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, list] = await Promise.all([api.health(), api.conversations()]);
      setHealth(h);
      setHistory(list);
    } catch (e) {
      setHealth("down");
      if (!(e instanceof EngineDown)) setError((e as Error).message);
    }
  }, [api]);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  /** Show the conversation or the settings. The address says which (#settings), so it can be reloaded or linked. */
  const go = useCallback((to: "chat" | "settings") => {
    setView(to);
    setMenu(false);
    if (to === "settings") setInside(false);
    const hash = to === "settings" ? "#settings" : "";
    if (window.location.hash !== hash) window.history.replaceState(null, "", window.location.pathname + window.location.search + hash);
  }, []);
  useEffect(() => {
    const moved = () => setView(window.location.hash === "#settings" ? "settings" : "chat");
    window.addEventListener("hashchange", moved);
    return () => window.removeEventListener("hashchange", moved);
  }, []);

  /** Take new settings: keep them in this browser, and where the engine's address changed, start over with what
   * that engine has kept. (Readings from another source are asked for again: see below.) */
  const applySettings = (next: PageSettings) => {
    saveSettings(next);
    if (next.engineUrl !== settings.engineUrl) {
      opening.current++;
      setHealth("checking");
      setHistory([]);
      setOpen(null);
      setCalls([]);
      setChosen(null);
      setPointed(null);
      setPinned(null);
      setError(null);
    }
    setSettings(next);
  };

  // Every reply on show is read, once: a transcript is asked for a single time, however often the page redraws
  // while the answers come in. A service that cannot be reached is another matter: the page says so, and the
  // replies it could not ask about are asked about again when the conversation moves on or "Try again" is pressed.
  const asked = useRef(new Set<string>());
  const mounted = useRef(true);
  // readings from one source are not another's: when the source changes, what was read is let go and asked for
  // again (the effect below this one does the asking), and an answer still on its way from the old one is dropped
  const asking = useRef(source);
  useEffect(() => {
    if (asking.current === source) return;
    asking.current = source;
    asked.current = new Set();
    setRead({});
    setAway(false);
  }, [source]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!panel) return;
    for (const t of calls) {
      if (asked.current.has(t.id)) continue;
      asked.current.add(t.id);
      source.reading(t).then(
        (r) => {
          if (!mounted.current || asking.current !== source) return;
          setAway(false);
          setRead((before) => ({ ...before, [t.id]: r }));
        },
        (e) => {
          if (!mounted.current || asking.current !== source) return;
          if (e instanceof ReadingsDown) {
            asked.current.delete(t.id);
            setAway(true);
          } else setRead((before) => ({ ...before, [t.id]: null }));
        },
      );
    }
  }, [calls, source, panel, again]);
  // the panel shows the word pressed, else the word pointed at, else the reply picked, else the newest reply
  const scope = pinned ?? pointed;
  const shownId = scope?.id ?? chosen ?? calls.at(-1)?.id ?? null;
  const shownCall = calls.find((t) => t.id === shownId) ?? null;
  const shownReading = shownCall ? read[shownCall.id] : undefined;

  /** Show a turn in the panel: its reply's reading, and the conversation up to it on the timeline. */
  const pick = (id: string) => {
    setChosen(id);
    setPinned(null);
    setInside(true);
  };

  // the newest words stay in view
  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [calls, sending]);

  // Escape puts away whatever is out: the history drawer first, then the panel
  useEffect(() => {
    if (!menu && !inside) return;
    const key = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (menu) setMenu(false);
      else setInside(false);
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [menu, inside]);

  const opening = useRef(0);   // the latest conversation asked for: an earlier one that answers late is dropped
  const show = async (id: string | null) => {
    const mine = ++opening.current;
    go("chat");
    setError(null);
    setOpen(id);
    setCalls([]);
    setChosen(null);
    setPointed(null);
    setPinned(null);
    if (id) {
      try {
        const kept = await api.conversation(id);
        if (mine === opening.current) setCalls(kept);
      } catch (e) {
        if (mine === opening.current) setError((e as Error).message);
      }
    }
    box.current?.focus();
  };

  const send = async (e?: FormEvent) => {
    e?.preventDefault();
    const message = draft.trim();
    if (!message || sending !== null || health === "down" || noEndpoint) return;
    setDraft("");
    setError(null);
    setSending(message);
    try {
      const transcript = await api.chat(message, open, chatExtras(settings, open === null));
      setCalls((before) => [...before, transcript]);
      setChosen(null);   // the panel follows the newest reply
      setPinned(null);
      setOpen(transcript.conversation);
      setHistory(await api.conversations());
    } catch (err) {
      // nothing was kept: the words go back in the box to be sent again
      setDraft(message);
      setError((err as Error).message);
      if (err instanceof EngineDown) setHealth("down");
    } finally {
      setSending(null);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  };

  const down = health === "down";
  const noEndpoint = typeof health === "object" && !health.endpoint;   // the engine is up and has no model to talk to
  const inSettings = view === "settings";
  return (
    <div className={`chat-app${menu ? " menu-open" : ""}${inside ? " inside-open" : ""}${panel ? "" : " no-inside"}`}>
      <button type="button" className="chat-menu" aria-label={menu ? "Close history" : "Open history"} aria-expanded={menu} aria-controls="chat-history" onClick={() => setMenu(!menu)}>
        <span />
        <span />
      </button>
      <div
        className="chat-scrim"
        aria-hidden="true"
        onClick={() => {
          setMenu(false);
          setInside(false);
        }}
      />
      <nav id="chat-history" className="chat-rail" aria-label="Conversations">
        <p className="chat-mark">undertone</p>
        <button type="button" className="chat-new" onClick={() => void show(null)}>
          New conversation
        </button>
        <p className="chat-kind">History</p>
        {history.length === 0 ? (
          <p className="chat-none">Nothing kept yet.</p>
        ) : (
          <ol>
            {history.map((c) => (
              <li key={c.id}>
                <button type="button" aria-current={c.id === open && !inSettings ? "true" : undefined} onClick={() => void show(c.id)}>
                  <span className="chat-title">{c.title || "Untitled"}</span>
                  <span className="chat-when">
                    {c.turns === 1 ? "1 turn" : `${c.turns} turns`} · {when(c.updated)}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        )}
        <button type="button" className="chat-settings" aria-current={inSettings ? "page" : undefined} onClick={() => go(inSettings ? "chat" : "settings")}>
          Settings
        </button>
        {/* which endpoint the engine is talking to, at the foot */}
        <p className={`chat-engine${down ? " down" : noEndpoint ? " unset" : ""}`} role="status">
          <i aria-hidden="true" />
          {health === "checking" ? "Looking for the engine…" : down ? "Engine not running" : noEndpoint ? "No model endpoint set" : `${health.model} · ${health.endpoint}`}
        </p>
      </nav>

      {inSettings ? (
        <main className="chat-main settings-main">
          <header className="chat-head">
            <span className="chat-eyebrow">undertone / settings</span>
            <button type="button" className="chat-inside-toggle" onClick={() => go("chat")}>
              Done
            </button>
          </header>
          <div className="chat-log">
            <SettingsPage settings={settings} onChange={applySettings} api={api} health={health} onEndpoint={() => void refresh()} />
          </div>
        </main>
      ) : (
      <main className="chat-main">
        <header className="chat-head">
          <span className="chat-eyebrow">undertone / chat</span>
          {panel && (
            <button type="button" className="chat-inside-toggle" aria-expanded={inside} aria-controls="chat-inside" onClick={() => setInside(!inside)}>
              Inside
            </button>
          )}
        </header>
        <div className="chat-log" ref={log} role="log" aria-live="polite" aria-label="Conversation">
          {down ? (
            <div className="chat-empty">
              <h1>The engine is not running.</h1>
              <p>This page talks to the undertone engine on this machine, and the engine talks to your endpoint. Start it with:</p>
              <pre>undertone-engine serve</pre>
              <p className="chat-faint">Looking for it at {settings.engineUrl}.</p>
              <p className="chat-row">
                <button type="button" className="chat-retry" onClick={() => void refresh()}>
                  Try again
                </button>
                <button type="button" className="chat-retry" onClick={() => go("settings")}>
                  Open settings
                </button>
              </p>
            </div>
          ) : noEndpoint && calls.length === 0 ? (
            <div className="chat-empty">
              <h1>No model endpoint yet.</h1>
              <p>Say where your model is served, and the engine will send your messages there.</p>
              <button type="button" className="chat-retry" onClick={() => go("settings")}>
                Open settings
              </button>
            </div>
          ) : calls.length === 0 && sending === null ? (
            <div className="chat-empty">
              <h1>Say something.</h1>
              <p>Each message goes to your endpoint through the engine, and the call is kept as a transcript.</p>
            </div>
          ) : (
            <ol className="chat-turns">
              {calls.map((t) => (
                // a turn is pressed to show it in the panel (a press that selects text, or lands on a word or a
                // control of its own, is left alone); "Turn n" under it is the same thing as a button
                <li
                  key={t.id}
                  className={[panel ? "pickable" : "", inside && t.id === shownId ? "shown" : ""].join(" ").trim() || undefined}
                  onClick={(e) => {
                    if (!panel || (e.target as HTMLElement).closest("button, summary, a") || window.getSelection()?.toString()) return;
                    pick(t.id);
                  }}
                >
                  <p className="chat-user">{t.request.messages.at(-1)?.content}</p>
                  {t.response.reasoning && (
                    <details className="chat-reasoning">
                      <summary>Reasoning</summary>
                      <p>{t.response.reasoning}</p>
                    </details>
                  )}
                  {/* the reading's pieces stand in for the reply only when they are the reply, word for word */}
                  {read[t.id] && fits(read[t.id]!, t) ? (
                    <ReplyText
                      reading={read[t.id]!}
                      active={scope?.id === t.id ? scope.token : null}
                      onPoint={(token) => setPointed(token === null ? null : { id: t.id, token })}
                      onPin={(token) => {
                        setPinned(pinned?.id === t.id && pinned.token === token ? null : { id: t.id, token });
                        setInside(true);
                      }}
                    />
                  ) : (
                    <p className="chat-reply">{t.response.content}</p>
                  )}
                  <p className="chat-meta">
                    {panel ? (
                      <button type="button" className="chat-turn" aria-pressed={inside && t.id === shownId} onClick={() => pick(t.id)}>
                        Turn {t.turn}
                      </button>
                    ) : (
                      `Turn ${t.turn}`
                    )}
                    {" · "}
                    <span className={`ids ${t.tokens?.source ?? "none"}`}>{IDS[t.tokens?.source ?? "none"]}</span>
                    {read[t.id] && ` · ${read[t.id]!.sample ? "sample reading" : "reading"}`}
                  </p>
                </li>
              ))}
              {sending !== null && (
                <li>
                  <p className="chat-user">{sending}</p>
                  <p className="chat-reply waiting" aria-label="Waiting for the reply">
                    <i />
                    <i />
                    <i />
                  </p>
                </li>
              )}
            </ol>
          )}
        </div>
        {error && (
          <p className="chat-error" role="alert">
            {error}
          </p>
        )}
        <form className="chat-composer" onSubmit={send}>
          <textarea
            ref={box}
            rows={1}
            value={draft}
            placeholder={down ? "Start the engine to chat." : noEndpoint ? "Set a model endpoint to chat." : "Message"}
            aria-label="Message"
            disabled={down || noEndpoint}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
          />
          <button type="submit" aria-label="Send" disabled={!draft.trim() || sending !== null || down || noEndpoint}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M8 13V3.5M3.75 7.5 8 3.25l4.25 4.25" />
            </svg>
          </button>
        </form>
      </main>
      )}
      {panel && (
        <div id="chat-inside" className="chat-inside" inert={!inside}>
          <InsidePanel
            onClose={() => setInside(false)}
            turns={down ? [] : calls.map((t) => ({ id: t.id, turn: t.turn, reading: read[t.id] }))}
            current={shownId}
            onPick={pick}
            reading={down || !shownCall ? undefined : shownReading === undefined ? "pending" : shownReading}
            away={away && !down ? { where: source.where ?? "", onRetry: () => setAgain((n) => n + 1) } : undefined}
            turn={shownCall?.turn ?? null}
            token={scope && scope.id === shownId ? scope.token : null}
            pinned={pinned !== null}
            onWhole={() => setPinned(null)}
            source={source.label}
          />
        </div>
      )}
    </div>
  );
}

/** Where a call's token ids came from, in a few words. */
const IDS = { server: "token ids from the endpoint", rebuilt: "token ids rebuilt by the engine", none: "no token ids" } as const;

/** A time for the history list: the hour today, the date before that. */
function when(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : d.toLocaleDateString([], { month: "short", day: "numeric" });
}
