"""A self-test: does the model in hand behave as the contract says?

It reads one short fixed exchange and checks, against the tolerances in `contract.json`:

  1. the pinned template and the tokenizer agree on the token ids (`render` raises if not);
  2. every block answers, with the contract's shape, and nothing is infinite or missing;
  3. at three positions, `head` on the last block agrees with the model's own norm and head in float32, and its
     distribution is close to the model's native logits (total variation);
  4. reading the same text twice gives exactly the same block outputs.

Passing says the install matches the setting the model was qualified in. It says nothing about any serving
endpoint, about emotions, or about a lens.
"""
import importlib.metadata
import platform
from typing import Any

#: one fixed exchange, long enough to read at three positions
SAMPLE = {
    "request": {"messages": [{"role": "user", "content": "Describe a harbour early in the morning, in three sentences."}]},
    "response": {
        "content": "The water lies flat and grey between the moored boats, and the first gulls are already walking the quay. "
        "A single engine starts somewhere beyond the breakwater and fades toward open sea. "
        "By the time the light reaches the cranes, the fish market has its shutters up and the day has begun."
    },
}


def check(model: Any) -> dict[str, Any]:
    """Run the self-test on a loaded model. Answers a report; `passed` is the verdict."""
    import torch

    contract = model.contract
    limits = contract.checks
    rendering = model.render(SAMPLE)   # (1): raises when template and tokenizer disagree
    reply = list(rendering.reply.positions)
    at = [reply[0], reply[len(reply) // 2], reply[-1]]
    blocks, native = model.run(rendering.ids, reply, logits_at=at)
    again = model.run(rendering.ids, reply)

    final = blocks[-1][[reply.index(p) for p in at]]
    ours = model.head(final)
    tv = float((ours.softmax(-1) - native.float().softmax(-1)).abs().sum(-1).max() / 2)
    head_gap = float((ours - model.native_head(final)).abs().max())
    repeat_gap = float((blocks - again).abs().max())
    shape = list(blocks.shape)

    checks = [
        {"name": "template and tokenizer agree", "passed": True, "value": len(rendering.ids), "unit": "tokens"},
        {"name": "every block answers in the contract's shape", "passed": shape == [contract.blocks, len(reply), contract.width] and bool(torch.isfinite(blocks.float()).all()), "value": shape},
        {"name": "head agrees with the model's own norm and head in float32", "passed": head_gap <= limits["fp32_head_atol"], "value": head_gap, "limit": limits["fp32_head_atol"]},
        {"name": "head is close to the model's native logits (total variation)", "passed": tv <= limits["full_vocab_tv_max"], "value": tv, "limit": limits["full_vocab_tv_max"]},
        {"name": "reading twice gives the same block outputs", "passed": repeat_gap <= limits["repeat_residual_max_abs"], "value": repeat_gap, "limit": limits["repeat_residual_max_abs"]},
    ]
    runtime = _runtime()
    qualified = contract.data["qualified"]["runtime"]["packages"]
    return {
        "model": contract.name,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "runtime": runtime,
        # not a failure: the checks above are the test. It says how far this install is from the one qualified.
        "differs_from_qualified": {p: {"here": runtime["packages"].get(p), "qualified": v} for p, v in qualified.items() if runtime["packages"].get(p) != v},
    }


def _runtime() -> dict[str, Any]:
    import torch

    packages = {}
    for name in ("torch", "transformers", "safetensors", "accelerate", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
