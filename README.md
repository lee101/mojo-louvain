# mojo-louvain

`mojo-louvain` is a standalone Mojo port of the compute-heavy local-moving
phase of [python-louvain](https://github.com/taynaud/python-louvain). It keeps
the familiar `community` Python package and NetworkX graph interface while
moving repeated neighbour-community scans and modularity updates into a native
Mojo shared library.

## Covered API

The public python-louvain 0.16 API is covered:

| module | contents |
| --- | --- |
| `community` | `best_partition`, `generate_dendrogram`, `partition_at_level`, `induced_graph`, `modularity`, `load_binary`, `check_random_state` |
| `community.community_status` | `Status` compatibility container |

`best_partition` and `generate_dendrogram` support weighted undirected
NetworkX `Graph` and `MultiGraph` inputs, `resolution`, `random_state`,
deprecated `randomize`, and an initial `partition`. Community labels may
differ from upstream because both implementations legitimately explore ties
in a different randomized order. The test suite exercises every documented
public function and the option combinations above against python-louvain.

Not covered: the upstream command-line entry point and private helper
functions. Directed graphs are rejected, matching upstream. This project is a
Linux/x86_64 Mojo build; it is not a drop-in binary wheel for other platforms.

## Install and use

```bash
pixi install
pixi run build
```

`pixi run test` and `pixi run bench` use the same environment. The Python
wrapper also builds `dist/libmojo-louvain.so` automatically if it is absent or
older than `src/capi.mojo`.

```bash
pixi run python - <<'PY'
import networkx as nx
import community

graph = nx.ring_of_cliques(4, 8)
partition = community.best_partition(graph, random_state=0)
print(len(set(partition.values())))  # 4
print(round(community.modularity(partition, graph), 3))  # 0.716
PY
```

Existing python-louvain imports remain familiar:

```python
import community as community_louvain

partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
```

## How it works

```
python/community/       NetworkX API, hierarchy coarsening, ctypes wrapper
          |             CSR conversion once per hierarchy level
src/capi.mojo           C ABI and native Louvain local-move passes
          |
dist/libmojo-louvain.so shared library
```

Each hierarchy level is a compact CSR graph: `int64` row offsets and neighbour
indices, with `float64` weights, degrees, self-loop weights, and community
totals. CSR arrays are preallocated NumPy buffers filled directly from the
NetworkX adjacency maps. The wrapper passes their addresses as `Int` values
through ctypes, without copying at the FFI boundary. Mojo rebuilds typed
mutable pointers inside the C ABI boundary and uses caller-owned scratch
buffers for community marks and neighbour weights; there are no allocations
or ownership transfers in the kernel. Contiguous initialization and modularity
reduction use float64 SIMD with scalar remainder handling. Python keeps the
flexible node labels and constructs the induced graph between levels.

## Tests

The test suite imports the installed `python-louvain 0.16` package under a
separate module name and asserts API-signature, numerical, and behavioural
parity. It also covers self-loop modularity, binary graph loading, dendrogram
composition, random-state behaviour, and the internal compatibility status
object.

```bash
pixi run build && pixi run test
```

## Benchmark

Measured with `pixi run bench` on Linux 6.8.0-136-generic, x86_64, glibc
2.39. Each figure is the fastest of three full public `best_partition` calls;
the comparison uses python-louvain 0.16 on the identical NetworkX graph.

| case | mojo-louvain | python-louvain | ratio | result |
| --- | ---: | ---: | ---: | --- |
| Louvain, 8 communities / 2,000 nodes | 69.0 ms | 344.6 ms | 4.99x | faster |
| Louvain, 12 communities / 4,800 nodes | 159.0 ms | 1122.3 ms | 7.06x | faster |

No GPU path is included: CSR construction and the local-moving neighbour scans
are irregular, bandwidth-bound operations with insufficient arithmetic
intensity to recover GPU launch and transfer costs. The local-moving pass is
also order-dependent, so it is kept serial; its independent initialization is
too small to recover thread-launch overhead at the measured graph sizes.

Reproduce the table with:

```bash
pixi run bench
```

## License

MIT
