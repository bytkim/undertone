/** Readings: what the undertone tools find inside the model for one reply, and the boundary they arrive through.
 *
 * A reading belongs to one transcript (one call). It says two things:
 *  - the reply as a whole: how far each emotion stands out, and the words the model leans toward at each depth;
 *  - the reply piece by piece: for the pieces where something stands out, which emotions, and the words at each depth
 *    there.
 *
 * The page does not know how a reading is made. It asks a `ReadingSource` for the reading of a transcript and draws
 * what comes back, or nothing when there is none.
 *
 * Real readings come from the readings server (core/server, `undertone-server serve`), which runs the tools on the
 * transcripts the engine keeps. That is the source the page uses unless its settings say otherwise.
 *
 * `sampleReadings` makes up a reading from the reply's own words, for looking at the page without a GPU. It is not
 * a measurement of anything, every reading it makes says so (`sample: true`), and the page labels it wherever it
 * is shown.
 */
import type { Transcript } from "./engine";
import type { PageSettings } from "./settings";

/** The four depths a lens reads at, from the model's early layers to its output. */
export const DEPTHS = ["early", "middle", "late", "output"] as const;
export type Depth = (typeof DEPTHS)[number];

/** The words the model leans toward at each depth, most leaned-toward first. */
export type LensWords = Record<Depth, string[]>;

/** One emotion on the whole reply: how far it stands out, 0 to 100. 0 is how neutral, matter-of-fact text reads on
 * that emotion; 100 is how stories written to convey it read. */
export interface EmotionLevel {
  name: string;
  family: string;
  level: number;
}

/** One emotion on one piece of the reply: how far it stands out there, 1 (just above what neutral text almost
 * never reaches) to 9 (as far as stories written to convey it). */
export interface EmotionMark {
  name: string;
  family: string;
  strength: number;
}

/** One piece of the reply's text. The pieces, joined, are the reply exactly. A piece where nothing stands out
 * carries only its text. */
export interface TokenReading {
  text: string;
  /** the emotions that stand out here, strongest first */
  emotions?: EmotionMark[];
  /** the words the model leans toward at this piece, at each depth */
  lens?: LensWords;
}

export interface Reading {
  schema: "undertone.reading.v1";
  /** the transcript (the call) this reading is of */
  transcript: string;
  /** the model's contract the reading was made with, for example "qwen3.5-4b" */
  model: string;
  /** made up to show the page without a GPU, not measured */
  sample?: boolean;
  /** false when the transcript lies outside the setting the model was checked in; `flags` says in what way */
  qualified?: boolean;
  flags?: string[];
  overall: {
    /** the emotions that stand out most over the whole reply, strongest first */
    emotions: EmotionLevel[];
    lens: LensWords;
  };
  tokens: TokenReading[];
}

/** Where readings come from. `null` means there is no reading for that transcript. A source that asks a service
 * rejects with `ReadingsDown` when the service cannot be reached at all, so the page can say so and ask again. */
export interface ReadingSource {
  /** a word for the page to show beside a reading: where it came from */
  label: string;
  /** the address readings are asked for at, when they come from a service */
  where?: string;
  reading(transcript: Transcript): Promise<Reading | null>;
}

/** The readings service could not be reached at all (as against answering that it has no reading). */
export class ReadingsDown extends Error {}

/** No readings at all: the page is the plain chat. */
export const noReadings: ReadingSource = { label: "off", reading: async () => null };

/** Whether something a service answered is a reading the page can draw: the right schema, and every part the page
 * reads present. Anything else is treated as no reading, so a service that answers badly cannot break the page. */
export function isReading(body: unknown): body is Reading {
  const r = body as Reading | null;
  const words = (lens: LensWords | undefined) => !!lens && DEPTHS.every((d) => Array.isArray(lens[d]));
  return (
    !!r &&
    r.schema === "undertone.reading.v1" &&
    Array.isArray(r.overall?.emotions) &&
    words(r.overall.lens) &&
    Array.isArray(r.tokens) &&
    r.tokens.every((t) => typeof t?.text === "string" && (t.emotions === undefined || Array.isArray(t.emotions)) && (t.lens === undefined || words(t.lens)))
  );
}

/** Whether a reading's pieces join back into the reply exactly. Only then can they stand in for the reply's text;
 * where they do not, the page shows the transcript's own words, unmarked, and never the reading's. */
export function fits(reading: Reading, transcript: Transcript): boolean {
  return reading.tokens.map((t) => t.text).join("") === transcript.response.content;
}

/** Readings from the readings server over HTTP: `GET {base}/readings/{transcript id}` answers a Reading, made on
 * the spot the first time it is asked for (which takes a few seconds), or an error when the transcript cannot be
 * read. Anything but a reading counts as none. */
export function httpReadings(base: string): ReadingSource {
  const where = base.replace(/\/$/, "");
  return {
    label: "measured",
    where,
    async reading(transcript) {
      let response: Response;
      try {
        response = await fetch(`${where}/readings/${encodeURIComponent(transcript.id)}`);
      } catch {
        throw new ReadingsDown("The readings server is not answering.");
      }
      if (!response.ok) return null;
      const body: unknown = await response.json().catch(() => null);
      return isReading(body) ? body : null;
    },
  };
}

// ── sample readings ───────────────────────────────────────────────────────────────────────────────────────────────

/** A handful of emotions with their families, enough to draw every colour. */
const SAMPLE_EMOTIONS: [string, string][] = [
  ["hopeful", "Compassionate Gratitude"], ["grateful", "Compassionate Gratitude"], ["calm", "Peaceful Contentment"],
  ["content", "Peaceful Contentment"], ["cheerful", "Exuberant Joy"], ["eager", "Exuberant Joy"],
  ["proud", "Competitive Pride"], ["amused", "Playful Amusement"], ["bored", "Depleted Disengagement"],
  ["weary", "Depleted Disengagement"], ["vigilant", "Vigilant Suspicion"], ["suspicious", "Vigilant Suspicion"],
  ["annoyed", "Hostile Anger"], ["defiant", "Hostile Anger"], ["anxious", "Fear and Overwhelm"],
  ["uneasy", "Fear and Overwhelm"], ["nostalgic", "Despair and Shame"], ["lonely", "Despair and Shame"],
];
const SAMPLE_WORDS = ["next", "then", "still", "toward", "light", "quiet", "open", "again", "inside", "almost", "here", "after"];

/** A small repeatable number generator, so the same transcript always gets the same sample. */
function seeded(text: string): () => number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
}

/** A made-up reading for a transcript: repeatable, shaped exactly like a real one, and marked as a sample. */
export function sampleReading(transcript: Transcript): Reading {
  const random = seeded(transcript.id + transcript.response.content);
  const pick = <T,>(list: T[]) => list[Math.floor(random() * list.length)];
  const pieces = transcript.response.content.match(/\s+|[^\s]+/g) ?? [];
  const own = pieces.filter((p) => /[\p{L}]{4,}/u.test(p)).map((p) => p.toLowerCase().replace(/[^\p{L}'-]/gu, ""));
  // eight different words a depth: mostly the reply's own, the rest from a small stock
  const words = (): string[] => {
    const out = new Set<string>();
    for (let tries = 0; out.size < 8 && tries < 60; tries++) out.add(own.length && random() < 0.6 ? pick(own) : pick(SAMPLE_WORDS));
    return [...out];
  };
  const lens = (): LensWords => Object.fromEntries(DEPTHS.map((d) => [d, words()])) as LensWords;
  // three or four emotions run through the reply; about one word in five carries one or two of them
  const cast = Array.from({ length: 4 }, () => pick(SAMPLE_EMOTIONS));
  const tokens: TokenReading[] = pieces.map((text) => {
    if (!/[\p{L}]{3,}/u.test(text) || random() > 0.22) return { text };
    const first = pick(cast), second = random() < 0.35 ? pick(cast) : null;
    const strength = 2 + Math.floor(random() * 8);
    const emotions: EmotionMark[] = [{ name: first[0], family: first[1], strength }];
    if (second && second[0] !== first[0]) emotions.push({ name: second[0], family: second[1], strength: Math.max(1, strength - 1 - Math.floor(random() * 3)) });
    return { text, emotions, lens: lens() };
  });
  // the strongest stands somewhere between 68 and 98 (so replies differ on the timeline), the rest fall away under it
  const top = 68 + random() * 30;
  const seen = new Map<string, EmotionLevel>();
  for (const [name, family] of [...cast, ...Array.from({ length: 6 }, () => pick(SAMPLE_EMOTIONS))]) {
    if (!seen.has(name)) seen.set(name, { name, family, level: Math.max(1, Math.round(top - seen.size * (4 + random() * 5))) });
  }
  return {
    schema: "undertone.reading.v1",
    transcript: transcript.id,
    model: transcript.model ?? "sample",
    sample: true,
    overall: { emotions: [...seen.values()].slice(0, 8), lens: lens() },
    tokens,
  };
}

/** Made-up readings, to show the page without a GPU. Not measurements. */
export const sampleReadings: ReadingSource = { label: "sample", reading: async (t) => sampleReading(t) };

/** The source the page's settings name: the readings server at its address, made-up samples, or none. */
export function sourceFor(settings: Pick<PageSettings, "readings" | "readingsUrl">): ReadingSource {
  if (settings.readings === "sample") return sampleReadings;
  if (settings.readings === "off") return noReadings;
  return httpReadings(settings.readingsUrl);
}

/** What the readings server says of itself (`GET /health`): the model it holds and the tools it reads with. */
export interface ReadingsHealth {
  ok: boolean;
  model: string;
  revision: string;
  device: string;
  tools: { emotions: number | null; lens: string | null; lens_ranked_by: string | null };
  /** how many readings it has kept */
  readings: number;
}

/** The readings server's account of itself, or null when it is not answering there. */
export async function readingsHealth(base: string): Promise<ReadingsHealth | null> {
  try {
    const response = await fetch(`${base.replace(/\/+$/, "")}/health`);
    const body = (await response.json()) as ReadingsHealth | null;
    return response.ok && body?.ok && typeof body.model === "string" && !!body.tools ? body : null;
  } catch {
    return null;
  }
}
