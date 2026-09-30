"""The README's Quickstart code runs as written."""
import pathlib
import re

import numpy as np
import torch

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_quickstart_runs():
    section = README.read_text().split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    assert len(blocks) >= 3
    torch.manual_seed(0)
    np.random.seed(0)
    namespace = {}
    for code in blocks:
        if "push_to_hub" in code:      # needs a Hub account; the round trip is in test_tides.py
            continue
        exec(compile(code, str(README), "exec"), namespace)
