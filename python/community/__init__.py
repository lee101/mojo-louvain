"""Drop-in covered subset of :mod:`python-louvain`."""

from .community_louvain import (
    best_partition,
    check_random_state,
    generate_dendrogram,
    induced_graph,
    load_binary,
    modularity,
    partition_at_level,
)

__version__ = "0.1.0"

__all__ = [
    "best_partition", "check_random_state", "generate_dendrogram", "induced_graph",
    "load_binary", "modularity", "partition_at_level",
]
