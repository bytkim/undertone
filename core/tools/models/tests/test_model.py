"""`run`, `head` and the self-test, on a small stand-in with the same layout as the real model (skipped without torch).
This proves the hooks, the head's arithmetic and the self-test's logic. It proves nothing about Qwen itself: that is
what `undertone-models check qwen3.5-4b` is for, on a machine with the weights."""
import pytest

from undertone_models import ContractError, check
from undertone_models.standin import standin_model

torch = pytest.importorskip("torch")

@pytest.fixture
def model(small, qwen_adapter, tok):
    return qwen_adapter.Model(small(), standin_model(), tok, "cpu")


def test_run_answers_every_blocks_output_at_the_positions_asked(model):
    ids = [3, 9, 27, 4, 11, 40]
    every = model.run(ids)
    assert list(every.shape) == [4, 6, 16]
    some = model.run(ids, [1, 4])
    assert torch.equal(some, every[:, [1, 4]])
    # the blocks really are the model's: the first is the embedding through block 0, the next builds on it
    decoder = model._hf.model.language_model
    first = decoder.layers[0](decoder.embed_tokens(torch.tensor([ids])))[0][0]
    assert torch.allclose(every[0], first) and not torch.allclose(every[1], every[0])
    # the model's own logits come back only when asked for, and nothing is left hooked
    blocks, logits = model.run(ids, [1, 4], logits_at=[4])
    assert torch.equal(blocks, some) and list(logits.shape) == [1, 512]
    assert all(len(b._forward_hooks) == 0 for b in decoder.layers)


def test_head_is_the_models_own_last_step(model):
    ids = [3, 9, 27, 4, 11, 40]
    blocks, native = model.run(ids, logits_at=[0, 3, 5])
    final = blocks[-1][[0, 3, 5]]
    ours = model.head(final)
    assert ours.dtype == torch.float32
    assert torch.allclose(ours, native, atol=1e-5) and torch.allclose(ours, model.native_head(final), atol=1e-5)
    # an earlier block read through the head is a different answer (that gap is what a lens is for)
    assert not torch.allclose(model.head(blocks[0][[0, 3, 5]]), native, atol=1e-3)


def test_the_last_step_on_its_own_reads_the_same_as_the_model(model, qwen, qwen_adapter):
    blocks = model.run([3, 9, 27, 4, 11, 40])
    head = model.as_head()
    assert head.vocab_size == 512 and head.tokenizer is model.tokenizer
    assert torch.equal(head.head(blocks[-1]), model.head(blocks[-1]))
    # built from the two tensors alone, as load_head does from the files
    decoder = model._hf.model.language_model
    alone = qwen_adapter.Head(model.contract, decoder.norm.weight.detach().clone(), decoder.embed_tokens.weight.detach().clone(), 1e-6, model.tokenizer, "cpu")
    assert torch.equal(alone.head(blocks[1]), model.head(blocks[1]))
    with pytest.raises(ContractError, match="last step"):
        qwen_adapter.Head(qwen, decoder.norm.weight, decoder.embed_tokens.weight, 1e-6, model.tokenizer, "cpu")


def test_a_model_of_another_shape_or_with_untied_embeddings_is_refused(small, qwen_adapter, tok, qwen):
    with pytest.raises(ContractError, match="4 blocks of width 16"):
        qwen_adapter.Model(qwen, standin_model(), tok, "cpu")
    with pytest.raises(ContractError, match="not tied"):
        qwen_adapter.Model(small(), standin_model(tied=False), tok, "cpu")


def test_the_self_test_passes_on_a_model_that_keeps_the_contract(model):
    report = check("qwen3.5-4b").check(model)   # (the stand-in's cut-down contract has no folder of its own)
    assert report["passed"] and report["model"] == "qwen3.5-4b"
    assert [c["passed"] for c in report["checks"]] == [True] * 5
    # the report says what is installed here, and how that differs from what the model was qualified with
    here = torch.__version__.split("+")[0]
    assert report["runtime"]["packages"]["torch"] == here
    assert ("torch" in report["differs_from_qualified"]) == (here != "2.14.0")


def test_the_self_test_fails_a_model_that_does_not_read_the_same_twice(small, qwen_adapter, tok):
    noisy = qwen_adapter.Model(small(), standin_model(noisy=True), tok, "cpu")
    report = check("qwen3.5-4b").check(noisy)
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert not report["passed"] and "reading twice gives the same block outputs" in failed
