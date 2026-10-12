import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class _Stub:
    """Placeholder for classes from packages that aren't installed."""
    def __init__(self, *a, **k):
        self._items = []
    def append(self, x): self._items.append(x)
    def extend(self, xs): self._items.extend(xs)
    def __setitem__(self, k, v):
        if not hasattr(self, "_map"): self._map = {}
        self._map[k] = v
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (_Stub,), {})


def children(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if hasattr(obj, "__dict__"):
        return list(vars(obj).items())
    return []


def find_candidates(obj, path="obj", depth=0, seen=None):
    """Yield (path, kind, value) for tables and lists of samples."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))

    if isinstance(obj, pd.DataFrame):
        yield path, "table", obj
        return
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        yield path, "table", pd.DataFrame(obj)
        return
    if isinstance(obj, (list, tuple)) and len(obj) > 1 and all(
        isinstance(s, (list, tuple, np.ndarray)) for s in obj[:50]
    ):
        yield path, "samples", obj
        return
    for k, v in children(obj):
        yield from find_candidates(v, f"{path}.{k}", depth + 1, seen)


def main(pkl_path, out_csv=None):
    with open(pkl_path, "rb") as f:
        obj = StubUnpickler(f).load()
    print("Top-level type:", type(obj).__name__)

    found = list(find_candidates(obj))
    if not found:
        print("Nothing table-like found. Contents:")
        print({k: type(v).__name__ for k, v in children(obj)})
        return

    # pick the biggest candidate
    def size(item):
        _, kind, val = item
        return val.size if kind == "table" else len(val)

    path, kind, val = max(found, key=size)

    if kind == "table":
        df = val
    else:
        # one sample per row, tokens in order across columns
        df = pd.DataFrame([list(s) for s in val])
        df.index.name = "sample"

    out_csv = Path(out_csv or Path(pkl_path).with_suffix(".csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"{path} ({kind}) shape {df.shape} -> {out_csv}")


if __name__ == "__main__":
    main(*sys.argv[1:])