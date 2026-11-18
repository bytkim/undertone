"""The arithmetic of emotion vectors. Pure functions over captured block outputs: no files, no model.

An emotion's vector at one block is what the stories of that emotion have in common there, beyond what all the
stories have in common, with the directions that ordinary emotionless text varies along taken out:

    1. average each story's block outputs over its tokens            (done at capture)
    2. average the stories of each emotion                            emotion_means
    3. subtract the average over all emotions                         center
    4. take out the leading directions of neutral text, per block     neutral_basis, project_out

Reading is the same steps on a new text: subtract the same average, take out the same directions, and take the
cosine with each emotion's vector, averaged equally over a band of blocks (band_similarity).

Shapes: S stories, E emotions, B blocks, d the model's width, T neutral tokens. Decompositions run in float64.
"""
from typing import Any, Sequence

#: neutral directions are taken out up to this share of the neutral text's variance, block by block
VARIANCE = 0.5


def emotion_means(story_means: Any, labels: Sequence[str], rows: Sequence[int] | None = None) -> tuple[list[str], Any]:
    """story_means [S, B, d] and one emotion per story -> (emotions in first-seen order, their means [E, B, d]).
    `rows` uses only those stories."""
    import torch

    rows = range(len(labels)) if rows is None else rows
    emotions = list(dict.fromkeys(labels[i] for i in rows))
    by: dict[str, list[int]] = {e: [] for e in emotions}
    for i in rows:
        by[labels[i]].append(i)
    return emotions, torch.stack([story_means[by[e]].double().mean(0) for e in emotions])


def center(means: Any) -> tuple[Any, Any]:
    """Subtract the average over emotions: (centered [E, B, d], the average [B, d])."""
    grand = means.mean(0)
    return means - grand, grand


class Gram:
    """The neutral text of one block, gathered a few tokens at a time: enough to find its leading directions
    without holding every token in memory."""

    def __init__(self, width: int):
        import torch

        self.n = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.outer = torch.zeros(width, width, dtype=torch.float64)

    def add(self, x: Any) -> None:
        """x [tokens, d]"""
        x = x.double()
        self.n += x.shape[0]
        self.sum += x.sum(0)
        self.outer += x.T @ x

    def basis(self, variance: float = VARIANCE) -> tuple[Any, float]:
        """The leading directions [k, d] (orthonormal rows) whose share of the variance first reaches `variance`,
        and the share they reach. The same answer as a principal component analysis of the centred tokens."""
        import torch

        if not 0 < variance <= 1:
            raise ValueError("variance is a share: above 0, at most 1")
        if self.n < 2:
            raise ValueError("neutral text is needed to find its directions")
        mean = self.sum / self.n
        cov = self.outer - self.n * torch.outer(mean, mean)
        values, vectors = torch.linalg.eigh(cov)
        values, vectors = values.flip(0).clamp_min(0), vectors.flip(1).T
        if not bool(torch.isfinite(values).all()) or float(values.sum()) <= 0:
            raise ValueError("the neutral text has no variance to speak of")
        share = values.cumsum(0) / values.sum()
        k = int((share < variance).sum()) + 1
        return vectors[:k].contiguous(), float(share[k - 1])


def neutral_basis(neutral: Any, variance: float = VARIANCE) -> tuple[list[Any], list[float]]:
    """neutral [T, B, d] -> for each block, its leading directions and the share of variance they reach."""
    out = []
    for b in range(neutral.shape[1]):
        gram = Gram(neutral.shape[2])
        gram.add(neutral[:, b])
        out.append(gram.basis(variance))
    return [o[0] for o in out], [o[1] for o in out]


def project_out(x: Any, basis: Sequence[Any]) -> Any:
    """x [N, B, d]: take out each block's neutral directions."""
    out = x.double().clone()
    for b, v in enumerate(basis):
        v = v.double()   # (a basis read back from a file is float32)
        out[:, b] = out[:, b] - (out[:, b] @ v.T) @ v
    return out


def extract(story_means: Any, labels: Sequence[str], basis: Sequence[Any], rows: Sequence[int] | None = None) -> dict[str, Any]:
    """The whole recipe on stories already averaged over their tokens: emotions, their vectors [E, B, d] and the
    average that was subtracted [B, d]."""
    emotions, means = emotion_means(story_means, labels, rows)
    centered, grand = center(means)
    return {"emotions": emotions, "vectors": project_out(centered, basis).float(), "grand_mean": grand.float()}


def _unit(x: Any) -> Any:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def similarity(x: Any, vectors: Any, grand_mean: Any, basis: Sequence[Any]) -> Any:
    """x [N, B, d] -> the cosine with each emotion's vector at each block [N, B, E]: centre by the stories'
    average, take out the neutral directions, then the cosine."""
    import torch

    out = []
    for b in range(vectors.shape[1]):
        h = x[:, b].double() - grand_mean[b].double()
        v = basis[b].double()
        h = h - (h @ v.T) @ v
        out.append(_unit(h) @ _unit(vectors[:, b].double()).T)
    return torch.stack(out, dim=1)


def band_similarity(x: Any, vectors: Any, grand_mean: Any, basis: Sequence[Any]) -> Any:
    """x [N, B, d] -> [N, E]: the cosines of `similarity`, averaged equally over the blocks given."""
    return similarity(x, vectors, grand_mean, basis).mean(1)


# ── how good are they: descriptive checks, none of them a pass mark ─────────────────────────────────────────────

def heldout(story_means: Any, labels: Sequence[str], topics: Sequence[str], train_topics: set[str], basis: Sequence[Any], top: int = 5) -> dict[str, Any]:
    """Build vectors from the stories of some situations and ask them to name the emotion of the stories of the
    rest: the share named first, and within the first `top`, at each block and with the band's average."""
    import torch

    train = [i for i, t in enumerate(topics) if t in train_topics]
    test = [i for i, t in enumerate(topics) if t not in train_topics]
    if not train or not test:
        raise ValueError("held-out checking needs stories on both sides")
    fit = extract(story_means, labels, basis, rows=train)
    index = {e: n for n, e in enumerate(fit["emotions"])}
    top = min(top, len(index))
    truth = torch.tensor([index[labels[i]] for i in test])
    sims = similarity(story_means[test], fit["vectors"], fit["grand_mean"], basis)   # [N, B, E]

    def score(s: Any) -> tuple[float, float]:
        ranks = s.topk(top, dim=-1).indices
        return float((ranks[:, 0] == truth).double().mean()), float((ranks == truth[:, None]).any(-1).double().mean())

    by_block = [score(sims[:, b]) for b in range(sims.shape[1])]
    band = score(sims.mean(1))
    return {"train_topics": sorted(train_topics), "stories_tested": len(test), "emotions": len(index), "chance_top1": 1 / len(index), f"chance_top{top}": top / len(index),
            "top1_by_block": [s[0] for s in by_block], f"top{top}_by_block": [s[1] for s in by_block], "band_top1": band[0], f"band_top{top}": band[1]}


def split_half(story_means: Any, labels: Sequence[str], topics: Sequence[str], basis: Sequence[Any]) -> dict[str, Any]:
    """Vectors from every other situation against vectors from the rest: the cosine between the two, emotion by
    emotion, and its median at each block."""
    order = sorted(set(topics))
    first = set(order[0::2])
    a = extract(story_means, labels, basis, rows=[i for i, t in enumerate(topics) if t in first])
    b = extract(story_means, labels, basis, rows=[i for i, t in enumerate(topics) if t not in first])
    where = [b["emotions"].index(e) for e in a["emotions"]]
    cos = (_unit(a["vectors"].double()) * _unit(b["vectors"][where].double())).sum(-1)   # [E, B]
    return {"median_by_block": cos.median(0).values.tolist(), "lowest": {a["emotions"][int(i)]: float(cos.mean(1)[i]) for i in cos.mean(1).argsort()[:5]}}


def group_agreement(story_means: Any, labels: Sequence[str], groups: Sequence[str], basis: Sequence[Any], full: dict[str, Any]) -> dict[str, Any]:
    """Vectors from each group's stories alone (one author, say) against the vectors from all of them: the median
    cosine at each block."""
    out = {}
    for g in sorted(set(groups)):
        rows = [i for i, v in enumerate(groups) if v == g]
        fit = extract(story_means, labels, basis, rows=rows)
        shared = [e for e in fit["emotions"] if e in full["emotions"]]
        a = fit["vectors"][[fit["emotions"].index(e) for e in shared]].double()
        b = full["vectors"][[full["emotions"].index(e) for e in shared]].double()
        out[g] = {"stories": len(rows), "median_cosine_to_all_by_block": (_unit(a) * _unit(b)).sum(-1).median(0).values.tolist()}
    return out
