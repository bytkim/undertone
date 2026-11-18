import type { CSSProperties, ReactNode } from "react";
import { familyColor, familyShort } from "./families";
import { DEPTHS, type LensWords, type Reading } from "./readings";

/** How many emotion rows the panel always has, whichever scope is on show: the panel never changes size. */
const ROWS = 6;
/** Words listed under each depth, one to a line (always this many lines, so the panel keeps its size). */
const WORDS = 6;
/** The pieces of one word are kept on one line up to this many characters; anything longer may break. */
const UNBROKEN = 32;

/** A reply's text with its reading laid over it: a piece where an emotion stands out is underlined in that emotion's
 * family colour, heavier the further it stands out, and can be pointed at or pressed. Everything else is plain text.
 * Pointing shows that piece in the panel for as long as the pointer or the focus stays; pressing keeps it there.
 *
 * The pieces are the model's tokens, and a word is often several of them ("a" and "ching"). A line may break at the
 * white space a piece begins with, and nowhere inside a word: that space is drawn as plain text outside the piece,
 * and the pieces of one word are kept together. */
export function ReplyText({
  reading,
  active,
  onPoint,
  onPin,
}: {
  reading: Reading;
  /** the piece on show in the panel, if it is one of this reply's */
  active: number | null;
  onPoint: (index: number | null) => void;
  onPin: (index: number) => void;
}) {
  const out: ReactNode[] = [];
  let word: ReactNode[] = [];
  let letters = 0;
  const endWord = () => {
    if (word.length > 1 && letters <= UNBROKEN) out.push(<span key={`w${out.length}`} className="word">{word}</span>);
    else out.push(...word);
    word = [];
    letters = 0;
  };
  reading.tokens.forEach((t, k) => {
    const space = /^\s*/.exec(t.text)![0];
    const text = t.text.slice(space.length);
    if (space) {
      endWord();
      out.push(space);
    }
    if (!text) return;
    const first = t.emotions?.[0];
    word.push(
      first ? (
        <button
          key={k}
          type="button"
          className={`tok${active === k ? " on" : ""}`}
          style={{ "--emo": familyColor(first.family), "--s": (first.strength / 9).toFixed(2) } as CSSProperties}
          aria-label={`${text.trim()}: ${t.emotions!.map((e) => e.name).join(", ")}`}
          aria-pressed={active === k}
          onPointerEnter={() => onPoint(k)}
          onPointerLeave={() => onPoint(null)}
          onFocus={() => onPoint(k)}
          onBlur={() => onPoint(null)}
          onClick={() => onPin(k)}
        >
          {text}
        </button>
      ) : (
        <span key={k}>{text}</span>
      ),
    );
    letters += text.length;
    if (/\s$/.test(text)) endWord();
  });
  endWord();
  return <p className="chat-reply marked">{out}</p>;
}

/** Inside the model: the reading of one reply, beside the conversation.
 *
 * It has one layout and two scopes. With nothing pointed at it shows the whole reply: the emotions that stand out
 * most, each with its level, and the words the model leans toward at each depth. With a word pointed at (or pressed)
 * it shows that word instead: the emotions on it and the words at each depth there. The rows are always the same
 * rows, so moving between a word and the whole reply never shifts anything on the page.
 *
 * At its foot, always in view, is the conversation turn by turn (`OverTurns`): pressing a turn shows that reply.
 *
 * It draws whatever reading it is given and knows nothing of where it came from, beyond the label it is told to
 * show. When the readings service cannot be reached it says so, with how to start it and a way to ask again. The page slides it out on the right and away again (`onClose`). */
export function InsidePanel({
  reading: given,
  turn,
  token,
  pinned,
  onWhole,
  onClose,
  source,
  turns,
  current,
  onPick,
  away,
}: {
  /** every turn of the conversation, for the timeline at the foot */
  turns: TurnReading[];
  /** the turn the panel is showing */
  current: string | null;
  /** show another turn's reply */
  onPick: (id: string) => void;
  /** slide the panel away */
  onClose: () => void;
  /** the reading on show; "pending" while it is being asked for; null when the reply has none; undefined before
   * there is any reply */
  reading: Reading | "pending" | null | undefined;
  turn: number | null;
  /** the piece in scope, or null for the whole reply */
  token: number | null;
  /** the piece was pressed, so it stays until "Whole reply" */
  pinned: boolean;
  onWhole: () => void;
  /** where the readings come from, in a word */
  source: string;
  /** set when the readings service cannot be reached: where it was looked for, and how to ask again */
  away?: { where: string; onRetry: () => void };
}) {
  const reading = given === "pending" ? undefined : given;
  const piece = reading && token !== null ? reading.tokens[token] : null;
  const lens: LensWords | null = piece ? (piece.lens ?? null) : (reading?.overall.lens ?? null);
  const rows = piece
    ? (piece.emotions ?? []).map((e) => ({ name: e.name, family: e.family, share: e.strength / 9, value: `${e.strength}/9` }))
    : (reading?.overall.emotions ?? []).map((e) => ({ name: e.name, family: e.family, share: e.level / 100, value: String(e.level) }));
  return (
    <aside className="inside" aria-label="Inside the model">
      <header className="inside-head">
        <h2>Inside the model</h2>
        {reading && <span className={`inside-tag${reading.sample ? " sample" : ""}`}>{reading.sample ? "Sample" : source}</span>}
        <button type="button" className="inside-close" aria-label="Close Inside the model" onClick={onClose}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M4 4l8 8M12 4l-8 8" />
          </svg>
        </button>
      </header>
      {away && given === "pending" ? (
        <div className="inside-away" role="status">
          <p>The readings server is not answering.</p>
          <p>It runs the model on a GPU and reads each reply the engine has kept. Start it with:</p>
          <pre>undertone-server serve</pre>
          {away.where && <p className="inside-faint">Looking for it at {away.where}.</p>}
          <button type="button" className="chat-retry" onClick={away.onRetry}>
            Try again
          </button>
        </div>
      ) : given === "pending" ? (
        <p className="inside-none">Reading this reply…</p>
      ) : reading === undefined ? (
        <p className="inside-none">Readings appear here once the model has replied.</p>
      ) : reading === null ? (
        <p className="inside-none">No reading for this reply.</p>
      ) : (
        <>
          <p className="inside-scope">
            <span>
              Turn {turn} · {piece ? <b>“{piece.text.trim()}”</b> : "whole reply"}
            </span>
            <button type="button" className="inside-whole" onClick={onWhole} disabled={!pinned}>
              Whole reply
            </button>
          </p>

          <section aria-label="Emotions">
            <h3>
              Emotions <span>{piece ? "on this word · strength 1 to 9" : "over the reply · level 0 to 100"}</span>
            </h3>
            <ol className="inside-emotions">
              {Array.from({ length: ROWS }, (_, k) => {
                const row = rows[k];
                return row ? (
                  <li key={k} style={{ "--emo": familyColor(row.family), "--share": row.share.toFixed(3) } as CSSProperties}>
                    <span className="inside-rank">{k + 1}</span>
                    <span className="inside-name">
                      {row.name}
                      <small>{familyShort(row.family)}</small>
                    </span>
                    <span className="inside-meter" aria-hidden="true">
                      <i />
                    </span>
                    <span className="inside-value">{row.value}</span>
                  </li>
                ) : (
                  <li key={k} className="idle" aria-hidden="true">
                    <span className="inside-rank">{k + 1}</span>
                    <span className="inside-name">—</span>
                    <span className="inside-meter">
                      <i />
                    </span>
                    <span className="inside-value" />
                  </li>
                );
              })}
            </ol>
          </section>

          <section aria-label="Lens">
            <h3>
              J lens <span>{piece ? "words leaned toward at this word" : "words leaned toward over the reply"}</span>
            </h3>
            {/* the four depths side by side, from the model's early layers to its output; under each, its words
                down the column, the most leaned-toward first */}
            <ol className="inside-lens">
              {DEPTHS.map((depth) => {
                const words = lens ? lens[depth].slice(0, WORDS) : [];
                return (
                  <li key={depth}>
                    <span className="inside-depth">{depth}</span>
                    <ol className="inside-words" aria-label={`${depth} words`}>
                      {Array.from({ length: WORDS }, (_, k) =>
                        words[k] !== undefined ? (
                          <li key={k} className={k === 0 ? "lead" : undefined}>
                            {words[k]}
                          </li>
                        ) : (
                          <li key={k} className="idle" aria-hidden="true">
                            {k === 0 ? "—" : ""}
                          </li>
                        ),
                      )}
                    </ol>
                  </li>
                );
              })}
            </ol>
          </section>
        </>
      )}
      {reading && (
        <p className="inside-note">
          {reading.sample
            ? "Sample readings: made up from the reply’s own words so this panel has something to draw. They are not measurements of the model."
            : "Readings describe associations inside the model. They are not evidence that it feels anything."}
          {reading.qualified === false && (
            <span className="inside-flagged">
              {" "}
              This reply lies outside the setting the model was checked in
              {reading.flags?.length ? ` (${reading.flags.map((f) => f.replaceAll("_", " ")).join(", ")})` : ""}, so read it with more care.
            </span>
          )}
        </p>
      )}
      {/* the foot of the panel, always in view (what is above it scrolls under): the conversation turn by turn */}
      <div className="inside-foot">
        <OverTurns turns={turns} current={current} onPick={onPick} />
      </div>
    </aside>
  );
}

/** One turn of the conversation, for the timeline: its reply's reading, where it has one. */
export interface TurnReading {
  id: string;
  turn: number;
  reading: Reading | null | undefined;
}

/** Over the turns: the conversation as a timeline at the foot of the panel. One column a turn, each a bar in the
 * colour of that reply's strongest emotion, as tall as its level; the turn the panel is showing is lit and named.
 * It shows the conversation up to that turn: the turns after it stand back, faint, and can still be pressed to move
 * on to them. Pressing a turn shows that reply. A turn with no reading keeps its place, empty. The columns share the
 * width, so the line holds from two turns to many (the numbers thin out as they get close). */
export function OverTurns({ turns, current, onPick }: { turns: TurnReading[]; current: string | null; onPick: (id: string) => void }) {
  const read = turns.filter((t) => t.reading);
  if (read.length === 0) return null;
  const at = turns.findIndex((t) => t.id === current);
  const now = turns[at]?.reading?.overall.emotions[0] ?? null;
  const every = [1, 2, 5, 10, 20].find((k) => turns.length / k <= 12) ?? 50;
  return (
    <section className="inside-over" aria-label="Over the turns">
      <h3>
        <span className="inside-upto">
          Over the turns
          {at >= 0 && <small>{at === 0 ? `turn ${turns[0].turn}` : `turns ${turns[0].turn}–${turns[at].turn}`}{at < turns.length - 1 ? ` of ${turns.length}` : ""}</small>}
        </span>
        <span>{now ? <b style={{ color: familyColor(now.family) }}>{now.name}</b> : "strongest emotion each turn"}</span>
      </h3>
      <ol style={{ gridTemplateColumns: `repeat(${turns.length}, minmax(0, 1fr))` }}>
        {turns.map((t, k) => {
          const top = t.reading?.overall.emotions[0];
          return (
            <li key={t.id} className={t.id === current ? "cur" : at >= 0 && k > at ? "ahead" : undefined}>
              <button
                type="button"
                disabled={!top}
                aria-current={t.id === current ? "step" : undefined}
                aria-label={top ? `Turn ${t.turn}: ${top.name}, ${top.level}` : `Turn ${t.turn}: no reading`}
                style={top ? ({ "--emo": familyColor(top.family), "--share": (top.level / 100).toFixed(3) } as CSSProperties) : undefined}
                onClick={() => onPick(t.id)}
              >
                <i />
              </button>
              <span className="inside-turn">{t.id === current || t.turn % every === 0 || t.turn === 1 ? t.turn : ""}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
