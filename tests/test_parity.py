"""Behavioural parity tests against the installed python-louvain 0.16."""

from __future__ import annotations

import array
import importlib.util
import inspect
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import community
from community.community_louvain import _csr, _ffi_array
from community.community_status import Status


def _upstream():
    prefix = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "community"
    spec = importlib.util.spec_from_file_location(
        "upstream_community", prefix / "__init__.py", submodule_search_locations=[str(prefix)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


UPSTREAM = _upstream()


def equivalent(left, right):
    nodes = list(left)
    return all((left[a] == left[b]) == (right[a] == right[b]) for a in nodes for b in nodes)


def test_public_signatures_match_upstream():
    for name in ["best_partition", "generate_dendrogram", "induced_graph", "modularity", "load_binary"]:
        assert inspect.signature(getattr(community, name)) == inspect.signature(getattr(UPSTREAM, name))


def test_weighted_modularity_matches_upstream():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 2.5), ("b", "c", 1.25), ("a", "a", 3.0), ("c", "d", 4.0)])
    partition = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert community.modularity(partition, graph) == pytest.approx(UPSTREAM.modularity(partition, graph))


def test_induced_graph_matches_upstream():
    graph = nx.cycle_graph(8)
    nx.set_edge_attributes(graph, {edge: i + 0.5 for i, edge in enumerate(graph.edges())}, "weight")
    partition = {node: node // 2 for node in graph}
    ours = community.induced_graph(partition, graph)
    theirs = UPSTREAM.induced_graph(partition, graph)
    assert nx.is_isomorphic(ours, theirs, edge_match=lambda a, b: a["weight"] == pytest.approx(b["weight"]))


def test_ring_of_cliques_partition_matches_upstream():
    graph = nx.ring_of_cliques(5, 8)
    ours = community.best_partition(graph, random_state=0)
    theirs = UPSTREAM.best_partition(graph, random_state=0)
    assert equivalent(ours, theirs)
    assert community.modularity(ours, graph) == pytest.approx(UPSTREAM.modularity(theirs, graph), abs=1e-12)


def test_planted_partition_matches_upstream():
    graph = nx.planted_partition_graph(4, 15, 0.55, 0.015, seed=4)
    ours = community.best_partition(graph, random_state=0)
    theirs = UPSTREAM.best_partition(graph, random_state=0)
    assert equivalent(ours, theirs)
    assert community.modularity(ours, graph) == pytest.approx(UPSTREAM.modularity(theirs, graph), abs=1e-12)


def test_simple_graph_csr_uses_contiguous_adjacency_rows():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (0, 0, 5.0)])
    nodes, offsets, neighbors, weights, degrees, loops = _csr(graph, "weight")
    assert nodes == [0, 1, 2]
    assert np.array_equal(offsets, [0, 2, 4, 5])
    assert np.array_equal(neighbors, [1, 0, 0, 2, 1])
    assert np.array_equal(weights, [2.0, 5.0, 2.0, 3.0, 3.0])
    assert np.array_equal(degrees, [12.0, 5.0, 3.0])
    assert np.array_equal(loops, [5.0, 0.0, 0.0])


def test_karate_quality_is_near_upstream():
    graph = nx.karate_club_graph()
    ours = community.best_partition(graph, random_state=0)
    theirs = UPSTREAM.best_partition(graph, random_state=0)
    assert len(set(ours.values())) == len(set(theirs.values()))
    assert community.modularity(ours, graph) >= UPSTREAM.modularity(theirs, graph) - 0.01


def test_dendrogram_composes_to_best_partition():
    graph = nx.ring_of_cliques(4, 7)
    dendo = community.generate_dendrogram(graph, random_state=3)
    partition = community.partition_at_level(dendo, len(dendo) - 1)
    assert equivalent(partition, community.best_partition(graph, random_state=3))
    assert len(dendo) >= 1


def test_initial_partition_is_supported():
    graph = nx.ring_of_cliques(4, 6)
    initial = {node: node // 6 for node in graph}
    ours = community.best_partition(graph, partition=initial, random_state=0)
    theirs = UPSTREAM.best_partition(graph, partition=initial, random_state=0)
    assert community.modularity(ours, graph) == pytest.approx(UPSTREAM.modularity(theirs, graph), abs=1e-12)


def test_empty_and_directed_graph_behaviour_matches_upstream():
    graph = nx.empty_graph(4)
    assert community.best_partition(graph, random_state=0) == UPSTREAM.best_partition(graph, random_state=0)
    with pytest.raises(TypeError):
        community.best_partition(nx.DiGraph([(0, 1)]))


def test_zero_weight_graph_matches_upstream_failure():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.0)
    with pytest.raises(ZeroDivisionError):
        community.best_partition(graph, random_state=0)


def test_binary_loader_matches_upstream(tmp_path):
    path = tmp_path / "graph.bin"
    with path.open("wb") as stream:
        array.array("I", [3]).tofile(stream)
        array.array("I", [1, 3, 4]).tofile(stream)
        array.array("I", [1, 0, 2, 1]).tofile(stream)
    assert nx.is_isomorphic(community.load_binary(path), UPSTREAM.load_binary(path))


def test_status_copy_and_init_match_upstream():
    graph = nx.path_graph(5)
    ours = Status()
    theirs = UPSTREAM.community_louvain.Status()
    ours.init(graph, "weight")
    theirs.init(graph, "weight")
    assert ours.node2com == theirs.node2com
    assert ours.degrees == theirs.degrees
    assert ours.copy().internals == theirs.internals


def test_random_state_contract_matches_upstream():
    ours = community.check_random_state(np.int64(8))
    theirs = UPSTREAM.community_louvain.check_random_state(np.int64(8))
    assert ours.randint(100000) == theirs.randint(100000)


def test_weight_resolution_and_randomize_are_supported():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 2.0), (2, 3, 3.0), (3, 0, 2.0)])
    with pytest.warns(DeprecationWarning):
        partition = community.best_partition(graph, weight="weight", resolution=0.8, randomize=False)
    assert set(partition) == set(graph)
    assert np.isfinite(community.modularity(partition, graph, weight="weight"))


def test_ffi_buffer_contract_rejects_wrong_dtype_shape_and_layout():
    with pytest.raises(RuntimeError, match="invalid values"):
        _ffi_array(np.zeros(3, dtype=np.float32), np.dtype(np.float64), 3, "values")
    with pytest.raises(RuntimeError, match="invalid values"):
        _ffi_array(np.zeros(4, dtype=np.float64)[::2], np.dtype(np.float64), 2, "values")
