import { useCallback, useEffect, useId, useState, type FormEvent, type ReactNode } from "react";
import { EngineDown, type EngineApi, type EngineSettings, type Health } from "./engine";
import { readingsHealth, type ReadingsHealth } from "./readings";
import { address, defaults, type PageSettings, type ReadingsFrom } from "./settings";

/** Settings: what the page talks to, and what a conversation is sent with.
 *
 * Four parts, each saying what is so as well as letting it be changed:
 *  - the model endpoint the engine talks to (the engine's own setting: asked for and changed through the engine,
 *    which keeps it for its next start);
 *  - what a new conversation is sent with (a system message, a longest reply, a temperature);
 *  - where readings come from (the readings server, made-up samples, or nowhere), and what that server holds;
 *  - where the engine itself is.
 *
 * The last three are the page's own and are kept in this browser; they take effect as they are changed. An API key
 * is in none of them: it stays in the engine's environment, and this page only says what becomes of it. */
export function SettingsPage({
  settings,
  onChange,
  api,
  health,
  onEndpoint,
}: {
  settings: PageSettings;
  onChange: (next: PageSettings) => void;
  api: EngineApi;
  health: Health | "checking" | "down";
  /** the engine's endpoint was changed here: look at the engine again */
  onEndpoint: () => void;
}) {
  const set = <K extends keyof PageSettings>(key: K, value: PageSettings[K]) => onChange({ ...settings, [key]: value });
  const down = health === "down";
  return (
    <div className="settings">
      <h1>Settings</h1>
      <p className="settings-lede">What this page talks to, and what a conversation is sent with.</p>

      <EndpointPart api={api} down={down} onEndpoint={onEndpoint} />

      <Part title="New conversations" about="What a message sent from this page carries. Left empty, each is the endpoint's own choice.">
        <Field label="System message" hint="Sent first in each new conversation. A conversation already begun keeps the one it started with.">
          {(ids) => <textarea {...ids} rows={3} value={settings.system} placeholder="None" onChange={(e) => set("system", e.target.value)} />}
        </Field>
        <div className="settings-pair">
          <NumberField label="Longest reply, in tokens" value={settings.maxTokens} least={1} most={1_000_000} whole onCommit={(v) => set("maxTokens", v)} />
          <NumberField label="Temperature" value={settings.temperature} least={0} most={2} step={0.1} onCommit={(v) => set("temperature", v)} />
        </div>
      </Part>

      <ReadingsPart settings={settings} onChange={onChange} />

      <Part title="The engine" about="The engine keeps your conversations and talks to the model endpoint. This page reaches both through it.">
        <AddressField label="Engine address" value={settings.engineUrl} onCommit={(v) => set("engineUrl", v)} />
        {health === "checking" ? (
          <Status tone="wait">Looking for the engine…</Status>
        ) : down ? (
          <>
            <Status tone="bad">Not answering at {settings.engineUrl}. Start it with:</Status>
            <pre className="settings-command">undertone-engine serve</pre>
          </>
        ) : (
          <Status tone="ok">Answering.</Status>
        )}
      </Part>

      <p className="settings-foot">
        <span>Everything but the model endpoint is kept in this browser.</span>
        <button type="button" className="settings-quiet" onClick={() => onChange(defaults())}>
          Back to defaults
        </button>
      </p>
    </div>
  );
}

/** The model endpoint: which one the engine talks to, what the engine knows of the model, and a way to change it. */
function EndpointPart({ api, down, onEndpoint }: { api: EngineApi; down: boolean; onEndpoint: () => void }) {
  const [kept, setKept] = useState<EngineSettings | null>(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [tokenIds, setTokenIds] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [note, setNote] = useState<{ tone: "ok" | "bad"; text: string } | null>(null);
  const [busy, setBusy] = useState<"look" | "save" | null>(null);
  const list = useId();

  const show = useCallback((s: EngineSettings) => {
    setKept(s);
    setBaseUrl(s.endpoint?.base_url ?? "");
    setModel(s.endpoint?.model ?? "");
    setTokenIds(s.endpoint?.token_ids ?? false);
  }, []);
  useEffect(() => {
    let mine = true;
    setKept(null);
    setNote(null);
    setModels([]);
    if (down) return;
    api.settings().then(
      (s) => mine && show(s),
      (e) => mine && !(e instanceof EngineDown) && setNote({ tone: "bad", text: (e as Error).message }),
    );
    return () => {
      mine = false;
    };
  }, [api, down, show]);

  const typed = address(baseUrl);
  const changed = !!kept && (typed !== (kept.endpoint ? address(kept.endpoint.base_url) : null) || model.trim() !== (kept.endpoint?.model ?? "") || tokenIds !== (kept.endpoint?.token_ids ?? false));
  const look = async () => {
    if (!typed) return setNote({ tone: "bad", text: 'The endpoint is an address like "http://localhost:8000/v1".' });
    setBusy("look");
    setNote(null);
    try {
      const found = await api.models(typed);
      setModels(found.models);
      if (!model.trim() && found.models.length) setModel(found.models[0]);
      setNote({ tone: "ok", text: found.models.length ? `${found.host} answered. It serves: ${found.models.join(", ")}.` : `${found.host} answered, and lists no models.` });
    } catch (e) {
      setModels([]);
      setNote({ tone: "bad", text: (e as Error).message });
    } finally {
      setBusy(null);
    }
  };
  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (!typed) return setNote({ tone: "bad", text: 'The endpoint is an address like "http://localhost:8000/v1".' });
    if (!model.trim()) return setNote({ tone: "bad", text: "Say the name the endpoint knows the model by." });
    setBusy("save");
    setNote(null);
    try {
      const now = await api.setEndpoint({ base_url: typed, model: model.trim(), token_ids: tokenIds });
      show(now);
      setNote({ tone: "ok", text: `The engine now talks to ${now.endpoint!.host} (${now.endpoint!.model}).` });
      onEndpoint();
    } catch (err) {
      setNote({ tone: "bad", text: (err as Error).message });
    } finally {
      setBusy(null);
    }
  };

  return (
    <Part title="Model endpoint" about="The OpenAI-compatible server your model runs on. The engine sends your messages there; this page never talks to it itself.">
      {down ? (
        <Status tone="bad">The engine is not answering, so its endpoint cannot be shown. Its address is at the foot of this page.</Status>
      ) : !kept ? (
        note ? <Status tone={note.tone}>{note.text}</Status> : <Status tone="wait">Asking the engine…</Status>
      ) : (
        <form className="settings-form" onSubmit={save} noValidate>
          {!kept.endpoint && <Status tone="warn">No endpoint is set yet. Say where your model is served to start chatting.</Status>}
          <Field label="Endpoint address">
            {(ids) => <input {...ids} type="url" inputMode="url" spellCheck={false} autoComplete="off" value={baseUrl} placeholder="http://localhost:8000/v1" onChange={(e) => setBaseUrl(e.target.value)} />}
          </Field>
          <Field label="Model name" hint="The name the endpoint knows the model by. “Find models” asks the endpoint which it has.">
            {(ids) => <input {...ids} type="text" spellCheck={false} autoComplete="off" list={list} value={model} onChange={(e) => setModel(e.target.value)} />}
          </Field>
          <datalist id={list}>
            {models.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          <Choice kind="checkbox" name="Ask the endpoint for the token ids it used" about="vLLM can give them. An endpoint that cannot is asked once and not again." checked={tokenIds} onChange={setTokenIds} />
          <div className="settings-actions">
            <button type="button" className="settings-button" disabled={busy !== null} onClick={() => void look()}>
              {busy === "look" ? "Asking…" : "Find models"}
            </button>
            <button type="submit" className="settings-button primary" disabled={busy !== null || !changed}>
              {busy === "save" ? "Saving…" : "Use this endpoint"}
            </button>
          </div>
          {note && <Status tone={note.tone}>{note.text}</Status>}

          <dl className="settings-facts">
            {kept.endpoint && (
              <Fact name="In use">
                {kept.endpoint.model} at {kept.endpoint.host}. {SOURCE[kept.source ?? "page"](kept.kept_in)}
              </Fact>
            )}
            <Fact name="API key" tone={kept.api_key === "held back" ? "warn" : undefined}>
              {KEY[kept.api_key]}
            </Fact>
            <Fact name="Model contract" tone={kept.contract?.knows_model === false ? "warn" : undefined}>
              {!kept.contract
                ? "None. Start the engine with --contract qwen3.5-4b and each conversation names the model the tools read."
                : kept.contract.knows_model === false
                  ? `The ${kept.contract.name} contract knows its weights as ${kept.contract.served_as.map((n) => `“${n}”`).join(" or ")}, not as “${kept.endpoint?.model}”. Readings of these conversations will be flagged.`
                  : `${kept.contract.name}${kept.contract.knows_model ? ", which knows this model's name." : "."}`}
            </Fact>
            <Fact name="Token ids">
              {kept.rebuilds_ids
                ? "When the endpoint returns none, the engine rebuilds them with the contract's tokenizer."
                : "When the endpoint returns none, a conversation is kept without them and the tools rebuild them."}
            </Fact>
            <Fact name="Conversations">Kept as files in {kept.transcripts}.</Fact>
          </dl>
        </form>
      )}
    </Part>
  );
}

/** Where the endpoint in use came from, in a sentence. */
const SOURCE: Record<"start" | "kept" | "page", (file: string) => string> = {
  start: () => "Given when the engine started. A change made here is kept for its next start, unless it is started with an endpoint again.",
  kept: (file) => `Kept from an earlier change, in ${file}.`,
  page: (file) => `Set here, and kept in ${file} for the engine's next start.`,
};

/** What becomes of the API key, in a sentence. */
const KEY: Record<EngineSettings["api_key"], string> = {
  sent: "Set in the engine's environment, and sent to this endpoint.",
  "held back": "Set in the engine's environment for another endpoint, so it is not sent to this one. To use a key here, start the engine with this endpoint.",
  none: "None. If the endpoint needs one, set UNDERTONE_API_KEY where the engine starts. A key is never typed into this page.",
};

/** Where readings come from, and what the readings server holds when they come from it. */
function ReadingsPart({ settings, onChange }: { settings: PageSettings; onChange: (next: PageSettings) => void }) {
  const [server, setServer] = useState<ReadingsHealth | null | "checking">("checking");
  const [again, setAgain] = useState(0);
  const group = useId();
  useEffect(() => {
    if (settings.readings !== "server") return;
    let mine = true;
    setServer("checking");
    void readingsHealth(settings.readingsUrl).then((h) => mine && setServer(h));
    return () => {
      mine = false;
    };
  }, [settings.readings, settings.readingsUrl, again]);
  const choice = (value: ReadingsFrom, name: string, about: string) => (
    <Choice kind="radio" group={group} name={name} about={about} checked={settings.readings === value} onChange={() => onChange({ ...settings, readings: value })} />
  );
  return (
    <Part title="Readings" about="What the Inside panel shows for each reply.">
      <fieldset className="settings-choices">
        <legend>Readings come from</legend>
        {choice("server", "The readings server", "Measured: it reads each reply again with the model, on a GPU.")}
        {choice("sample", "Samples", "Made up from the reply's own words, to see the panel without a GPU. Not measurements.")}
        {choice("off", "Nowhere", "The page is the plain chat, with no panel.")}
      </fieldset>
      {settings.readings === "server" && (
        <>
          <AddressField label="Readings server address" value={settings.readingsUrl} onCommit={(v) => onChange({ ...settings, readingsUrl: v })} />
          {server === "checking" ? (
            <Status tone="wait">Looking for the readings server…</Status>
          ) : server === null ? (
            <>
              <Status tone="bad">Not answering at {settings.readingsUrl}. Start it on the machine with the GPU:</Status>
              <pre className="settings-command">undertone-server serve</pre>
              <button type="button" className="settings-button" onClick={() => setAgain((n) => n + 1)}>
                Look again
              </button>
            </>
          ) : (
            <>
              <Status tone="ok">Answering.</Status>
              <dl className="settings-facts">
                <Fact name="Model">
                  {server.model} at revision {server.revision.slice(0, 12)}, on {server.device}.
                </Fact>
                <Fact name="Emotions" tone={server.tools.emotions === null ? "warn" : undefined}>
                  {server.tools.emotions === null ? "No vector set for this model: readings have no emotions." : `${server.tools.emotions} emotion vectors.`}
                </Fact>
                <Fact name="Lens" tone={server.tools.lens === null ? "warn" : undefined}>
                  {server.tools.lens === null ? "No lens: readings have no words." : `${server.tools.lens}, its words ranked by ${server.tools.lens_ranked_by}.`}
                </Fact>
                <Fact name="Kept">{server.readings === 1 ? "1 reading." : `${server.readings} readings.`}</Fact>
              </dl>
            </>
          )}
        </>
      )}
    </Part>
  );
}

function Part({ title, about, children }: { title: string; about: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className="settings-part" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      <p className="settings-about">{about}</p>
      {children}
    </section>
  );
}

function Status({ tone, children }: { tone: "ok" | "bad" | "warn" | "wait"; children: ReactNode }) {
  return (
    <p className={`settings-status ${tone}`} role={tone === "bad" ? "alert" : "status"}>
      <i aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

function Fact({ name, tone, children }: { name: string; tone?: "warn"; children: ReactNode }) {
  return (
    <div className={tone}>
      <dt>{name}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/** A labelled field. Its hint, or what is wrong with it, describes it without becoming part of its name. */
function Field({ label, hint, bad, children }: { label: string; hint?: string; bad?: string | false; children: (ids: { id: string; "aria-describedby"?: string }) => ReactNode }) {
  const id = useId();
  const about = hint || bad ? `${id}-about` : undefined;
  return (
    <div className="settings-field">
      <label htmlFor={id}>{label}</label>
      {children({ id, "aria-describedby": about })}
      {bad ? (
        <small id={about} className="bad">
          {bad}
        </small>
      ) : (
        hint && <small id={about}>{hint}</small>
      )}
    </div>
  );
}

/** A box to tick or one of several to pick, with a line saying what it means. */
function Choice({ kind, group, name, about, checked, onChange }: { kind: "checkbox" | "radio"; group?: string; name: string; about: string; checked: boolean; onChange: (checked: boolean) => void }) {
  const id = useId();
  return (
    <div className="settings-choice">
      <input id={id} type={kind} name={group} checked={checked} aria-describedby={`${id}-about`} onChange={(e) => onChange(e.target.checked)} />
      <div>
        <label htmlFor={id}>{name}</label>
        <small id={`${id}-about`}>{about}</small>
      </div>
    </div>
  );
}

/** An address that takes effect when it is left (or Enter is pressed), and only if it is one. */
function AddressField({ label, value, onCommit }: { label: string; value: string; onCommit: (value: string) => void }) {
  const [draft, setDraft] = useState(value);
  const [bad, setBad] = useState(false);
  useEffect(() => {
    setDraft(value);
    setBad(false);
  }, [value]);
  const commit = () => {
    const typed = address(draft);
    setBad(!typed);
    if (typed && typed !== value) onCommit(typed);
    else if (typed) setDraft(typed);
  };
  return (
    <Field label={label} bad={bad && "An address like http://127.0.0.1:8787."}>
      {(ids) => (
        <input
          {...ids}
          type="url"
          inputMode="url"
          spellCheck={false}
          autoComplete="off"
          value={draft}
          aria-invalid={bad}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), commit())}
        />
      )}
    </Field>
  );
}

/** A number that may be left empty (the endpoint then chooses). What is typed takes effect once it is a number in
 * range; anything else leaves the setting as it was. */
function NumberField({ label, value, least, most, step, whole, onCommit }: { label: string; value: number | null; least: number; most: number; step?: number; whole?: boolean; onCommit: (value: number | null) => void }) {
  const [draft, setDraft] = useState(value === null ? "" : String(value));
  useEffect(() => {
    setDraft((before) => (before !== "" && Number(before) === value ? before : value === null ? "" : String(value)));
  }, [value]);
  const n = draft.trim() === "" ? null : Number(draft);
  const bad = n !== null && (!Number.isFinite(n) || n < least || n > most || (!!whole && !Number.isInteger(n)));
  return (
    <Field label={label} bad={bad && (whole ? `A whole number from ${least}.` : `A number from ${least} to ${most}.`)}>
      {(ids) => (
        <input
          {...ids}
          type="number"
          inputMode={whole ? "numeric" : "decimal"}
          min={least}
          max={most}
          step={step ?? 1}
          value={draft}
          placeholder="The endpoint's own"
          aria-invalid={bad}
          onChange={(e) => {
            const text = e.target.value;
            setDraft(text);
            const typed = text.trim() === "" ? null : Number(text);
            if (typed === null) onCommit(null);
            else if (Number.isFinite(typed) && typed >= least && typed <= most && (!whole || Number.isInteger(typed))) onCommit(typed);
          }}
        />
      )}
    </Field>
  );
}
