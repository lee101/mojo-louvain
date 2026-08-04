"""Compatibility status container used by python-louvain integrations."""


class Status:
    """Mutable Louvain bookkeeping, matching python-louvain's internal type."""

    def __init__(self):
        self.node2com = {}
        self.total_weight = 0
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}

    def __str__(self):
        return (
            "node2com : " + str(self.node2com) + " degrees : " + str(self.degrees)
            + " internals : " + str(self.internals) + " total_weight : " + str(self.total_weight)
        )

    def copy(self):
        other = Status()
        other.node2com = self.node2com.copy()
        other.internals = self.internals.copy()
        other.degrees = self.degrees.copy()
        other.gdegrees = self.gdegrees.copy()
        other.loops = self.loops.copy()
        other.total_weight = self.total_weight
        return other

    def init(self, graph, weight, part=None):
        self.__init__()
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for count, node in enumerate(graph):
                degree = float(graph.degree(node, weight=weight))
                if degree < 0:
                    raise ValueError(f"Bad node degree ({degree})")
                self.node2com[node] = count
                self.degrees[count] = degree
                self.gdegrees[node] = degree
                edge = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge.get(weight, 1))
                self.internals[count] = self.loops[node]
            return
        for node in graph:
            community = part[node]
            degree = float(graph.degree(node, weight=weight))
            self.node2com[node] = community
            self.degrees[community] = self.degrees.get(community, 0.0) + degree
            self.gdegrees[node] = degree
            inside = 0.0
            for neighbor, data in graph[node].items():
                edge_weight = data.get(weight, 1)
                if edge_weight <= 0:
                    raise ValueError(f"Bad graph type ({type(graph)})")
                if part[neighbor] == community:
                    inside += float(edge_weight) if neighbor == node else float(edge_weight) / 2.0
            self.internals[community] = self.internals.get(community, 0.0) + inside
