"""Louvain community detection, with the local-moving phase in Mojo."""

from __future__ import annotations

import numbers
import warnings
from collections import defaultdict
from typing import Hashable

import networkx as nx
import numpy as np

from ._lib import lib
from .community_status import Status

__PASS_MAX = 100
__MIN = 1e-7


def check_random_state(seed):
    """Turn `seed` into the NumPy random-state form used by python-louvain."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy random state")


def _seed(random_state) -> int:
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**63 - 1))
    return int(random_state.randint(0, 2**63 - 1))


def _validate_graph(graph):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
    if graph.number_of_nodes() == 0:
        raise ValueError("Bad graph type, empty graph")


def _edge_weight(data, weight):
    value = data.get(weight, 1) if weight else 1
    if value < 0:
        raise ValueError("Bad node degree (-1.0). Louvain requires non-negative weights")
    return float(value)


def _csr(graph, weight):
    nodes = list(graph)
    index = {node: i for i, node in enumerate(nodes)}
    if graph.is_multigraph():
        rows = [defaultdict(float) for _ in nodes]
        edges = graph.edges(data=True, keys=True)
        for u, v, _, data in edges:
            w = _edge_weight(data, weight)
            ui, vi = index[u], index[v]
            rows[ui][vi] += w
            if ui != vi:
                rows[vi][ui] += w
        offsets = np.empty(len(nodes) + 1, dtype=np.int64)
        offsets[0] = 0
        indices = []
        values = []
        degrees = np.empty(len(nodes), dtype=np.float64)
        loops = np.zeros(len(nodes), dtype=np.float64)
        for i, row in enumerate(rows):
            for j, value in row.items():
                indices.append(j)
                values.append(value)
                if i == j:
                    loops[i] = value
            degrees[i] = sum(row.values()) + loops[i]
            offsets[i + 1] = len(indices)
        return nodes, offsets, np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float64), degrees, loops

    index_rows = [[] for _ in nodes]
    weight_rows = [[] for _ in nodes]
    for u, v, data in graph.edges(data=True):
        w = _edge_weight(data, weight)
        ui, vi = index[u], index[v]
        index_rows[ui].append(vi)
        weight_rows[ui].append(w)
        if ui != vi:
            index_rows[vi].append(ui)
            weight_rows[vi].append(w)
    offsets = np.empty(len(nodes) + 1, dtype=np.int64)
    offsets[0] = 0
    indices = []
    values = []
    degrees = np.empty(len(nodes), dtype=np.float64)
    loops = np.zeros(len(nodes), dtype=np.float64)
    for i, row in enumerate(index_rows):
        row_weights = weight_rows[i]
        indices.extend(row)
        values.extend(row_weights)
        for j, value in zip(row, row_weights):
            if i == j:
                loops[i] = value
        degrees[i] = sum(row_weights) + loops[i]
        offsets[i + 1] = len(indices)
    return nodes, offsets, np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float64), degrees, loops


def _ffi_array(array, dtype, size, name):
    """Validate a caller-owned NumPy buffer before exposing its address to C."""
    if array.dtype != dtype or array.ndim != 1 or array.size != size:
        raise RuntimeError(f"invalid {name} buffer for the native kernel")
    if not array.flags.c_contiguous or not array.flags.aligned or array.ctypes.data == 0:
        raise RuntimeError(f"invalid {name} buffer for the native kernel")
    return array


def _one_level(graph, weight, resolution, random_state):
    nodes, offsets, neighbors, weights, degrees, loops = _csr(graph, weight)
    n = len(nodes)
    communities = np.empty(n, dtype=np.int64)
    totals = np.empty(n, dtype=np.float64)
    internals = np.empty(n, dtype=np.float64)
    marks = np.empty(n, dtype=np.int64)
    candidate = np.empty(n, dtype=np.int64)
    neigh_weights = np.empty(n, dtype=np.float64)
    links = float(graph.size(weight=weight))
    # The C ABI receives raw addresses.  Keep every array strongly referenced in
    # this frame and validate its exact dtype, shape, and layout before the call.
    offsets = _ffi_array(offsets, np.dtype(np.int64), n + 1, "offsets")
    neighbors = _ffi_array(neighbors, np.dtype(np.int64), int(offsets[-1]), "neighbors")
    weights = _ffi_array(weights, np.dtype(np.float64), neighbors.size, "weights")
    degrees = _ffi_array(degrees, np.dtype(np.float64), n, "degrees")
    loops = _ffi_array(loops, np.dtype(np.float64), n, "loops")
    communities = _ffi_array(communities, np.dtype(np.int64), n, "communities")
    totals = _ffi_array(totals, np.dtype(np.float64), n, "totals")
    internals = _ffi_array(internals, np.dtype(np.float64), n, "internals")
    marks = _ffi_array(marks, np.dtype(np.int64), n, "marks")
    candidate = _ffi_array(candidate, np.dtype(np.int64), n, "candidate")
    neigh_weights = _ffi_array(neigh_weights, np.dtype(np.float64), n, "neigh_weights")
    score = lib().mlj_one_level(
        offsets.ctypes.data, neighbors.ctypes.data, weights.ctypes.data,
        degrees.ctypes.data, loops.ctypes.data, communities.ctypes.data,
        totals.ctypes.data, internals.ctypes.data, marks.ctypes.data,
        candidate.ctypes.data, neigh_weights.ctypes.data, n, links,
        float(resolution), _seed(random_state), __PASS_MAX, __MIN,
    )
    return {node: int(communities[i]) for i, node in enumerate(nodes)}, float(score)


def _renumber(partition):
    values = {value: i for i, value in enumerate(dict.fromkeys(partition.values()))}
    return {node: values[community] for node, community in partition.items()}


def partition_at_level(dendrogram, level):
    """Return the node-to-community partition at dendrogram `level`."""
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        partition = {node: dendrogram[index][community] for node, community in partition.items()}
    return partition


def induced_graph(partition, graph, weight="weight"):
    """Produce the weighted graph whose nodes are the supplied communities."""
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())
    for u, v, data in graph.edges(data=True):
        w = _edge_weight(data, weight)
        cu, cv = partition[u], partition[v]
        previous = ret.get_edge_data(cu, cv, {weight: 0}).get(weight, 0)
        ret.add_edge(cu, cv, **{weight: previous + w})
    return ret


def modularity(partition, graph, weight="weight"):
    """Compute the weighted Newman-Girvan modularity of `partition`."""
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")
    inc = defaultdict(float)
    deg = defaultdict(float)
    for node in graph:
        com = partition[node]
        deg[com] += graph.degree(node, weight=weight)
        for neighbor, data in graph[node].items():
            if partition[neighbor] == com:
                if graph.is_multigraph():
                    edge_weight = sum(_edge_weight(attrs, weight) for attrs in data.values())
                else:
                    edge_weight = _edge_weight(data, weight)
                inc[com] += 2.0 * edge_weight if neighbor == node else edge_weight
    result = 0.0
    for com in set(partition.values()):
        result += inc[com] / (2.0 * links) - (deg[com] / (2.0 * links)) ** 2
    return result


def generate_dendrogram(graph, part_init=None, weight="weight", resolution=1.0,
                         randomize=None, random_state=None):
    """Find a Louvain hierarchy; mirrors python-louvain's public signature."""
    _validate_graph(graph)
    if randomize is not None:
        warnings.warn(
            "The `randomize` parameter will be deprecated in future versions. "
            "Use `random_state` instead.", DeprecationWarning,
        )
        if randomize and random_state is not None:
            raise ValueError("`randomize` and `random_state` cannot be used at the same time")
        random_state = 0 if randomize is False else random_state
    state = check_random_state(random_state)
    if graph.number_of_edges() == 0:
        return [{node: i for i, node in enumerate(graph)}]
    # A graph containing only zero-weight edges reaches a division by zero in
    # upstream's modularity initialization.  Do not call the C kernel with
    # uninitialized output buffers in that case.
    if graph.size(weight=weight) == 0:
        raise ZeroDivisionError("float division by zero")
    current_graph = graph.copy()
    if part_init is None:
        status, current_modularity = _one_level(current_graph, weight, resolution, state)
    else:
        if set(part_init) != set(graph):
            raise KeyError("part_init must contain every graph node")
        initial_graph = induced_graph(part_init, current_graph, weight)
        moved, current_modularity = _one_level(initial_graph, weight, resolution, state)
        status = {node: moved[part_init[node]] for node in current_graph}
    status = _renumber(status)
    dendrogram = [status]
    while True:
        current_graph = induced_graph(status, current_graph, weight)
        next_status, next_modularity = _one_level(current_graph, weight, resolution, state)
        next_status = _renumber(next_status)
        if next_modularity - current_modularity < __MIN:
            break
        status = next_status
        dendrogram.append(status)
        current_modularity = next_modularity
    return dendrogram


def best_partition(graph, partition=None, weight="weight", resolution=1.0,
                   randomize=None, random_state=None):
    """Compute a node-to-community mapping with the Louvain method."""
    dendo = generate_dendrogram(graph, partition, weight, resolution, randomize, random_state)
    return partition_at_level(dendo, len(dendo) - 1)


def load_binary(data):
    """Load the binary graph format used by python-louvain's CLI."""
    import array

    with open(data, "rb") as stream:
        reader = array.array("I")
        reader.fromfile(stream, 1)
        num_nodes = reader.pop()
        reader = array.array("I")
        reader.fromfile(stream, num_nodes)
        cumulative = reader.tolist()
        num_links = reader.pop()
        reader = array.array("I")
        reader.fromfile(stream, num_links)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    start = 0
    for node, stop in enumerate(cumulative):
        graph.add_edges_from((node, int(neighbor)) for neighbor in reader[start:stop])
        start = stop
    return graph
