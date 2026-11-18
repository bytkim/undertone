"""The arithmetic, on numbers with a known answer planted in them."""
import pytest

from undertone_emotion import vectors as V

torch = pytest.importorskip("torch")

D, B = 24, 3


def planted(seed=0, stories_each=40, noise=0.1):
    """Three emotions, each a direction of its own; a shared offset; and two loud "neutral" directions that carry
    no emotion at all. Stories are spread over four situations."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    directions = {"calm": basis[0], "afraid": basis[1], "proud": basis[2]}
    loud = basis[3:5]
    offset = 5 * basis[5]
    rows, labels, topics = [], [], []
    for name, direction in directions.items():
        for k in range(stories_each):
            x = offset + direction + (torch.randn(2, generator=g) * 4) @ loud + noise * torch.randn(D, generator=g)
            rows.append(torch.stack([x * (b + 1) for b in range(B)]))   # every block the same shape of thing, scaled
            labels.append(name)
            topics.append(f"t{k % 4}")
    neutral = torch.stack([torch.stack([(offset + (torch.randn(2, generator=g) * 4) @ loud + 0.1 * torch.randn(D, generator=g)) * (b + 1) for b in range(B)]) for _ in range(400)])
    return torch.stack(rows), labels, topics, neutral, directions, loud


def test_the_vectors_are_the_planted_directions_with_the_neutral_ones_taken_out():
    X, labels, topics, neutral, directions, loud = planted()
    basis, reached = V.neutral_basis(neutral, variance=0.9)
    # the two loud neutral directions are found (they carry nearly all of the neutral variance), at every block
    for v, share in zip(basis, reached):
        assert v.shape[0] == 2 and share >= 0.9
        assert torch.allclose(v @ v.T, torch.eye(2, dtype=v.dtype), atol=1e-8)
        assert torch.allclose((v.float() @ loud.T).square().sum(1), torch.ones(2), atol=0.02)   # the same plane as the planted two
    # asked for half the variance, one direction is enough, and only one is taken
    assert [v.shape[0] for v in V.neutral_basis(neutral, variance=0.4)[0]] == [1, 1, 1]
    fit = V.extract(X, labels, basis)
    assert fit["emotions"] == ["calm", "afraid", "proud"] and list(fit["vectors"].shape) == [3, B, D]
    centred = {e: d - torch.stack(list(directions.values())).mean(0) for e, d in directions.items()}
    for n, e in enumerate(fit["emotions"]):
        for b in range(B):
            cos = torch.nn.functional.cosine_similarity(fit["vectors"][n, b], centred[e], dim=0)
            assert float(cos) > 0.9, (e, b, float(cos))
            assert float((fit["vectors"][n, b] @ basis[b].float().T).abs().max()) < 1e-4   # nothing left along a neutral direction


def test_the_gathered_directions_are_the_principal_components():
    _, _, _, neutral, _, _ = planted()
    gram = V.Gram(D)
    for chunk in neutral[:, 1].split(37):   # gathered a few tokens at a time
        gram.add(chunk)
    ours, reached = gram.basis(0.9)
    x = neutral[:, 1].double()
    x = x - x.mean(0)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    share = s.square().cumsum(0) / s.square().sum()
    k = int((share < 0.9).sum()) + 1
    assert ours.shape[0] == k and reached == pytest.approx(float(share[k - 1]))
    # the same span: projecting onto one basis and then the other loses nothing
    assert torch.allclose(ours.T @ ours, vh[:k].T @ vh[:k], atol=1e-8)
    with pytest.raises(ValueError):
        V.Gram(D).basis()


def test_reading_names_the_right_emotion_and_the_checks_say_so():
    X, labels, topics, neutral, _, _ = planted()
    basis, _ = V.neutral_basis(neutral, variance=0.9)
    fit = V.extract(X, labels, basis)
    sims = V.band_similarity(X, fit["vectors"], fit["grand_mean"], basis)
    assert list(sims.shape) == [len(labels), 3]
    assert [fit["emotions"][int(j)] for j in sims.argmax(-1)] == labels
    assert torch.allclose(sims, V.similarity(X, fit["vectors"], fit["grand_mean"], basis).mean(1))
    held = V.heldout(X, labels, topics, {"t0", "t1", "t2"}, basis, top=2)
    assert held["band_top1"] == 1.0 and held["stories_tested"] == 30 and held["chance_top1"] == pytest.approx(1 / 3) and len(held["top1_by_block"]) == B
    assert min(V.split_half(X, labels, topics, basis)["median_by_block"]) > 0.9
    groups = ["a" if k % 2 else "b" for k in range(len(labels))]
    agreement = V.group_agreement(X, labels, groups, basis, fit)
    assert set(agreement) == {"a", "b"} and min(agreement["a"]["median_cosine_to_all_by_block"]) > 0.9
    # with no neutral directions taken out, the loud ones swamp the emotion: the step earns its place
    nothing = [torch.zeros(0, D, dtype=torch.float64) for _ in range(B)]
    raw = V.extract(X, labels, nothing)
    raw_sims = V.band_similarity(X, raw["vectors"], raw["grand_mean"], nothing)
    assert (sims.max(-1).values.mean() > raw_sims.max(-1).values.mean())
