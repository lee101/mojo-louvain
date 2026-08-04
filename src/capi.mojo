"""C ABI for the Louvain local-moving kernel.

The graph is CSR: offsets[n + 1], neighbors[2m], and weights[2m].  The
caller owns graph and scratch buffers; the kernel never allocates.
"""

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ip(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def fp(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


def modularity(internals: F64Ptr, totals: F64Ptr, n: Int, links: Float64, resolution: Float64) -> Float64:
    if links == 0.0:
        return 0.0
    var result = 0.0
    for c in range(n):
        if totals[c] > 0.0:
            var q = totals[c] / (2.0 * links)
            result += resolution * internals[c] / links - q * q
    return result


def next_random(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1442695040888963407


@export("mlj_one_level")
def mlj_one_level(
    offsets_addr: Int,
    neighbors_addr: Int,
    weights_addr: Int,
    degrees_addr: Int,
    loops_addr: Int,
    communities_addr: Int,
    totals_addr: Int,
    internals_addr: Int,
    marks_addr: Int,
    candidate_addr: Int,
    neigh_weights_addr: Int,
    n: Int,
    links: Float64,
    resolution: Float64,
    seed: Int,
    max_pass: Int,
    min_gain: Float64,
) abi("C") -> Float64:
    """Optimize singleton communities and return the resulting modularity."""
    if n == 0 or links == 0.0:
        return 0.0
    var offsets = ip(offsets_addr)
    var neighbors = ip(neighbors_addr)
    var weights = fp(weights_addr)
    var degrees = fp(degrees_addr)
    var loops = fp(loops_addr)
    var communities = ip(communities_addr)
    var totals = fp(totals_addr)
    var internals = fp(internals_addr)
    var marks = ip(marks_addr)
    var candidates = ip(candidate_addr)
    var neigh_weights = fp(neigh_weights_addr)

    for i in range(n):
        communities[i] = Int64(i)
        totals[i] = degrees[i]
        internals[i] = loops[i]
        marks[i] = -1

    var old_modularity = modularity(internals, totals, n, links, resolution)
    var state = UInt64(seed) + 1442695040888963407
    var phase = 0
    while phase < max_pass:
        for i in range(n):
            candidates[i] = Int64(i)
        var i = n - 1
        while i > 0:
            state = next_random(state)
            var j = Int(state >> 33) % (i + 1)
            var tmp = candidates[i]
            candidates[i] = candidates[j]
            candidates[j] = tmp
            i -= 1

        var moves = 0
        for order_index in range(n):
            var node = candidates[order_index]
            var old_community = communities[node]
            var stamp = Int64(phase * n) + node
            var count = 0
            var e = offsets[node]
            while e < offsets[node + 1]:
                var neighbor = neighbors[e]
                if neighbor != node:
                    var c = communities[neighbor]
                    if marks[c] != stamp:
                        marks[c] = stamp
                        candidates[count] = c
                        neigh_weights[c] = 0.0
                        count += 1
                    neigh_weights[c] += weights[e]
                e += 1

            var old_weight = 0.0
            if marks[old_community] == stamp:
                old_weight = neigh_weights[old_community]
            var degree = degrees[node]
            totals[old_community] -= degree
            internals[old_community] -= old_weight + loops[node]

            var best_community = old_community
            var best_weight = old_weight
            var best_gain = 0.0
            var remove_cost = -old_weight + resolution * totals[old_community] * degree / (2.0 * links)
            for k in range(count):
                var c = candidates[k]
                var dnc = neigh_weights[c]
                var gain = remove_cost + dnc - resolution * totals[c] * degree / (2.0 * links)
                if gain > best_gain:
                    best_gain = gain
                    best_community = c
                    best_weight = dnc

            communities[node] = best_community
            totals[best_community] += degree
            internals[best_community] += best_weight + loops[node]
            if best_community != old_community:
                moves += 1

        var new_modularity = modularity(internals, totals, n, links, resolution)
        if moves == 0 or new_modularity - old_modularity < min_gain:
            return new_modularity
        old_modularity = new_modularity
        phase += 1
    return old_modularity
