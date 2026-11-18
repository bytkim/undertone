/** The ten emotion families as undertone draws them: one colour and one short name each. A reading names each
 * emotion's family, so the page needs nothing else to colour it. */
export const FAMILY_COLORS: Record<string, string> = {
  "Exuberant Joy": "#e6c56a",
  "Peaceful Contentment": "#8fcaa0",
  "Compassionate Gratitude": "#e3a3b5",
  "Competitive Pride": "#d79a5c",
  "Playful Amusement": "#c5d67a",
  "Depleted Disengagement": "#9aa39d",
  "Vigilant Suspicion": "#8fb4e3",
  "Hostile Anger": "#e0806c",
  "Fear and Overwhelm": "#b9a0dc",
  "Despair and Shame": "#7f9cc4",
};

export const FAMILY_SHORT: Record<string, string> = {
  "Exuberant Joy": "Joy",
  "Peaceful Contentment": "Contentment",
  "Compassionate Gratitude": "Gratitude",
  "Competitive Pride": "Pride",
  "Playful Amusement": "Amusement",
  "Depleted Disengagement": "Disengagement",
  "Vigilant Suspicion": "Suspicion",
  "Hostile Anger": "Anger",
  "Fear and Overwhelm": "Fear",
  "Despair and Shame": "Despair",
};

/** A family's colour; the page's ink for a family it does not know. */
export const familyColor = (family: string): string => FAMILY_COLORS[family] ?? "var(--ink2)";
export const familyShort = (family: string): string => FAMILY_SHORT[family] ?? family;
