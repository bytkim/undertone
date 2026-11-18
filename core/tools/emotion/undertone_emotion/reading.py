"""Read a capture with a set of emotion vectors: which emotions stand out, over the reply and token by token.

For each captured token, at each block of the band: subtract the stories' average, take out the neutral directions,
take the cosine with every emotion's vector; then average over the band. That is a raw cosine, and raw cosines mean
little alone, so each is placed between two reference points kept in the vector set:

    share  0 is how neutral, matter-of-fact text reads on that emotion's vector; 1 is how the stories written to
           convey that emotion read on it. It can go below 0 and above 1.
    level  the share of a whole reply as a whole number from 0 to 100
    marks  token by token, the emotions whose share reaches a floor that neutral text almost never reaches, with a
           strength from 1 (at the floor) to 9 (at a share of 1)

These are associations inside the model, measured against stories written to convey each emotion. They are not
evidence that the model feels anything.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from undertone_capture import CaptureError, items as capture_items, read as read_capture

from .build import BuildError, VectorSet, _rows, smooth

SCHEMA = "undertone.emotion.v1"
RUN_SCHEMA = "undertone.emotion.run.v1"
#: how many further emotions a marked token may carry beside its strongest
ALSO = 2


class ReadingError(ValueError):
    """A capture that cannot be read with these vectors, or a folder that does not match."""


def read(vs: VectorSet, record: Mapping[str, Any], blocks: Any, top: int = 8) -> dict[str, Any]:
    """One capture as emotions: the `top` with the greatest share over the whole reply, and each token's marks."""
    if record.get("mean"):
        raise ReadingError("this capture holds an average over tokens, not the tokens: capture again without --mean")
    if record["model"]["name"] != vs.record["model"]["name"] or record["model"]["revision"] != vs.record["model"]["revision"]:
        raise ReadingError(f"this capture was made with {record['model']['name']} at another revision than the vectors")
    try:
        rows = _rows(dict(record), vs.blocks)
    except BuildError as error:
        raise ReadingError(str(error)) from None
    device = vs.vectors.device
    x = blocks[rows].to(device).float().transpose(0, 1)   # [tokens, B, d]
    names, family = vs.emotions, vs.record["families"]

    # the whole reply: the average over its tokens, from how a neutral assistant message reads to how the stories do
    whole = vs.cosines(x.mean(0, keepdim=True))
    share = vs.shares(whole, "message")[0]
    best = share.topk(min(top, len(names))).indices.tolist()
    overall = [{"name": names[j], "family": family.get(names[j]), "cosine": _round(whole[0, j]), "share": _round(share[j]), "level": _level(float(share[j]))} for j in best]

    # token by token, from how a neutral token reads, smoothed with its neighbours
    st = smooth(vs.shares(vs.cosines(x), "token"))
    floor, step = vs.record["marks"]["floor"], vs.record["marks"]["step"]
    strongest = st.topk(min(1 + ALSO, len(names)), dim=-1)
    tokens = []
    for t, values, indices in zip(record["tokens"], strongest.values.tolist(), strongest.indices.tolist()):
        marks = [{"name": names[j], "family": family.get(names[j]), "share": _round(v), "strength": min(9, 1 + int((v - floor) / step))} for v, j in zip(values, indices) if v >= floor]
        tokens.append({"position": t["position"], "id": t["id"], "text": t["text"], "emotions": marks})
    return {
        "schema": SCHEMA,
        "transcript": record.get("transcript"),
        "conversation": record.get("conversation"),
        "turn": record.get("turn"),
        "model": record["model"],
        "vectors": {"sha256": vs.record["sha256"], "emotions": len(names), "blocks": vs.blocks},
        "capture": {"sha256": record["tensor"].get("sha256"), "select": record.get("select"), "skip": record.get("skip")},
        "flags": list(record.get("flags", [])),
        "qualified": record.get("qualified"),
        "marks": {"floor": floor, "step": step},
        "overall": overall,
        "tokens": tokens,
    }


def run(vs: VectorSet, captures: str | Path, out: str | Path, top: int = 8, log: Any = None) -> dict[str, Any]:
    """Read every capture in a folder and keep each as `<transcript id>.json` in `out`, with one `emotion.json`
    for the run. Running again with the same vectors carries on; other vectors or settings are refused."""
    out = Path(out)
    settings = {"model": vs.record["model"], "vectors": vs.record["sha256"], "top": top}
    before = json.loads((out / "emotion.json").read_text()) if (out / "emotion.json").is_file() else None
    if before is not None and before.get("settings") != settings:
        raise ReadingError(f"{out} already holds readings made with other vectors or settings. Use another folder.")
    out.mkdir(parents=True, exist_ok=True)
    started, done, kept, left = _now(), [], [], []
    for record in capture_items(captures):
        tid = record["transcript"]
        if (out / f"{tid}.json").is_file():
            kept.append(tid)
            continue
        try:
            _, blocks = read_capture(captures, tid)   # the file is checked against its hash first
            reading = read(vs, record, blocks, top=top)
        except (ReadingError, CaptureError) as error:
            left.append({"transcript": tid, "reason": str(error)})
            continue
        (out / f"{tid}.json").write_text(json.dumps(reading, ensure_ascii=False, indent=1) + "\n")
        done.append(tid)
        if log:
            marked = sum(1 for t in reading["tokens"] if t["emotions"])
            log(f"{tid}  turn {reading['turn']}  {', '.join(e['name'] + ' ' + str(e['level']) for e in reading['overall'][:4])}  |  {marked} of {len(reading['tokens'])} tokens marked")
    result = {"schema": RUN_SCHEMA, "settings": settings, "started": started, "finished": _now(), "read": done, "already_there": kept, "left_out": left}
    (out / "emotion.json").write_text(json.dumps({**result, "read": sorted(set((before or {}).get("read", [])) | set(done) | set(kept))}, ensure_ascii=False, indent=1) + "\n")
    return result


def readings(out: str | Path) -> list[dict[str, Any]]:
    """Every reading in a folder."""
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("*.json")) if p.name != "emotion.json"]


def _level(share: float) -> int:
    return int(round(max(0.0, min(1.0, share)) * 100))


def _round(x: Any) -> float:
    return float(f"{float(x):.4g}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
