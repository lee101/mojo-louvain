"""Benchmark the public Louvain operation against python-louvain."""

from __future__ import annotations

import importlib.util
import math
import platform
import sys
import time
from pathlib import Path

import networkx as nx

import community


def upstream():
    root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "community"
    spec = importlib.util.spec_from_file_location("bench_upstream", root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REFERENCE = upstream()


def best_time(fn, repeat=3):
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        result = min(result, time.perf_counter() - start)
    return result


def graph(groups, size, p_in, p_out, seed):
    return nx.planted_partition_graph(groups, size, p_in, p_out, seed=seed)


CASES = [
    ("Louvain, 8 communities / 2,000 nodes", graph(8, 250, 0.060, 0.0015, 7)),
    ("Louvain, 12 communities / 4,800 nodes", graph(12, 400, 0.035, 0.0008, 9)),
]


def main():
    print(f"Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print("| case | mojo-louvain | python-louvain | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, item in CASES:
        community.best_partition(item, random_state=0)
        ours = best_time(lambda: community.best_partition(item, random_state=0))
        theirs = best_time(lambda: REFERENCE.best_partition(item, random_state=0))
        ratio = theirs / ours
        result = "faster" if ours < theirs else "slower"
        print(f"| {name} | {ours * 1e3:.1f} ms | {theirs * 1e3:.1f} ms | {ratio:.2f}x | {result} |")


if __name__ == "__main__":
    main()
