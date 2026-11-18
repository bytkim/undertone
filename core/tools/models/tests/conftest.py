"""The tests need no network, no GPU and no model: they use the package's stand-ins (undertone_models.standin) and
a small contract cut down from the real one."""
import copy
import json

import pytest

from undertone_models import adapter, contract, read
from undertone_models.standin import FakeTokenizer

@pytest.fixture
def qwen():
    return contract("qwen3.5-4b")


@pytest.fixture
def qwen_adapter(qwen):
    return adapter(qwen)


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def small(tmp_path, qwen):
    """The real contract cut down to 4 blocks of width 16, with whatever else a test changes."""

    def make(**changes):
        data = copy.deepcopy(qwen.data)
        data["shape"].update(blocks=4, width=16)
        data["depths"] = {"early": [0], "middle": [1], "late": [2], "output": [3]}
        for key, value in changes.items():
            node = data
            *path, last = key.split(".")
            for part in path:
                node = node[part]
            node[last] = value
        folder = tmp_path / f"small-{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        (folder / "contract.json").write_text(json.dumps(data))
        return read(folder / "contract.json")

    return make
