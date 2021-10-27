import numpy as np
cimport cython

from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int value_in_array(int val, long[::1] arr, int size) nogil:
    """Check if value in array."""
    cdef int i
    for i in range(size):
        if arr[i] == val:
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _shortest_path_undirected(long[::1] parents, long source, long target):
    """Find shortest undirected path between two nodes.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.
    source :    int
                Index of source node.
    target :    int
                Index of target node.

    Returns
    -------
    path :      (M, ) array
                Array of node indices making up the path between source and
                target.
    False
                Returns False if there is no path between source and target.

    """
    source_path = np.empty(len(parents), dtype='long')
    source_path[:] = -1
    target_path = np.empty(len(parents), dtype='long')
    target_path[:] = -1

    cdef long[::1] source_path_view = source_path
    cdef long[::1] target_path_view = target_path

    cdef long idx1, idx2, p_source, p_target, i
    cdef Py_ssize_t N = len(parents)

    # Walk from source to root
    # (stop if we hit `target` on the way)
    p_source = source
    idx1 = 0
    while p_source >= 0:
        source_path_view[idx1] = p_source
        idx1 += 1
        p_source = parents[p_source]

        if p_source == target:
            source_path_view[idx1] = p_source
            break

    # One final increment because we are re-using idx1
    # as length for source_path_view
    idx1 += 1

    # Return now if we met `target` on the way
    if p_source == target:
        return source_path[:idx1]

    # Cut of the tail end of source_path
    # (speeds up contains queries)
    source_path = source_path[:idx1]

    # Walk from target to root and check if on the way:
    # 1. We meet source
    # 2. We meet any node in source's path
    p_target = target
    idx2 = 0
    while p_target >= 0:
        target_path_view[idx2] = p_target
        idx2 += 1
        p_target = parents[p_target]

        if p_target == source:
            break
        if value_in_array(p_target, source_path_view, idx1):
            break

    # Cut of the tail end of target_path
    target_path = target_path[:idx2]

    if p_target == source:
        return target_path
    elif value_in_array(p_target, source_path_view, idx1):
        # Combine the two paths up to where they met
        for i in range(idx1):
            if source_path_view[i] == p_target:
                break
        return np.append(source_path[:i], target_path[::-1])

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def _shortest_path_directed(long[::1] parents, long source, long target):
    """Find shortest directed path between two nodes.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.
    source :    int
                Index of source node.
    target :    int
                Index of target node.

    Returns
    -------
    path :      (M, ) array
                Array of node indices making up the path between source and
                target.
    False
                Returns False if there is no path between source and target.

    """
    source_path = np.empty(len(parents), dtype='long')
    source_path[:] = -1

    cdef long[::1] source_path_view = source_path

    cdef long idx, p_source
    cdef Py_ssize_t N = len(parents)

    # Walk from source to root
    # (stop if we hit `target` on the way)
    p_source = source
    idx = 0
    while p_source >= 0:
        source_path_view[idx] = p_source
        idx += 1
        p_source = parents[p_source]

        if p_source == target:
            source_path_view[idx] = p_source
            break

    # Return now if we met `target` on the way
    if p_source == target:
        return source_path[:idx + 1]

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def _path_to_root(long[::1] parents, long node):
    """Return path from given node to root.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.
    node :      int
                Index of start node.

    Returns
    -------
    path :      (M, ) array
                Array of node indices making up the path between node and
                the root.
    """
    path = np.empty(len(parents), dtype='long')
    path[:] = -1

    cdef long[::1] path_view = path

    cdef long idx
    cdef Py_ssize_t N = len(parents)

    # Walk from node to root
    idx = 0
    while node >= 0:
        path_view[idx] = node
        idx += 1
        node = parents[node]

    return path[:idx]


@cython.boundscheck(False)
@cython.wraparound(False)
def _all_paths_to_root(long[::1] parents):
    """Return paths from all nodes to root.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.

    Returns
    -------
    paths :     (N, N) array
                Array of node indices making up the path between node and
                the root. Each row is a path to root. Unused cells will be -1.

    """
    paths = np.empty((len(parents), len(parents)), dtype='long')
    paths[:, :] = -1

    cdef long[:, ::1] paths_view = paths
    cdef long[::1] parents_view = parents

    cdef long idx1, idx2, node
    cdef Py_ssize_t N = len(parents)

    # Walk from each node to the root
    for idx1 in prange(N, nogil=True):
      node = idx1
      for idx2 in range(N):
        paths_view[idx1, idx2] = node
        node = parents_view[node]

        if node < 0:
          break

    return paths


@cython.boundscheck(False)
@cython.wraparound(False)
def _geodesic_matrix(long[::1] parents, weights=None):
    """Return geodesic distance between all nodes.

    Parameters
    ----------
    parents :   (N, ) int32 (long) array
                Indices (!) of parent node for each node.
    weights :   (N, ) float32 (double) array, optional
                Weights for each child -> parent connection. If ``None`` each
                step will increment distance by 1.

    Returns
    -------
    dists :     (N, N) float 32 array
                Geodesic matrix.

    """
    cdef long idx1, idx2, node, l1, l2, node1, node2, use_weights
    cdef double d
    cdef Py_ssize_t N = len(parents)

    # Create the results matrix
    dists = np.zeros((len(parents), len(parents)), dtype='double')
    # Set to -1 (for disconnected pieces)
    dists[:] = - 1

    if isinstance(weights, type(None)):
        use_weights = 0
    else:
        use_weights = 1
        weights = np.asarray(weights)
        weights[np.isnan(weights)] = 0
        weights = weights.astype('double')

    cdef double[:, ::1] dists_view = dists
    cdef long[::1] parents_view = parents
    cdef double[::1] weights_view = weights

    # Find the leafs
    nodes = np.arange(len(parents))
    leafs = nodes[~np.isin(nodes, parents)]
    cdef long[::1] leafs_view = leafs
    cdef Py_ssize_t N3 = len(leafs)

    # Walk from each node to the root
    for idx1 in range(N):
        node = idx1
        d = 0
        while node >= 0 and dists_view[idx1, node] < 0:
            dists_view[idx1, node] = d
            dists_view[node, idx1] = d

            # Track distance travelled
            if use_weights:
                d += weights_view[node]
            else:
                d += 1

            node = parents_view[node]

    # Above, we calculated the "forward" distances but we're still missing
    # the distances between nodes on separate branches. Our approach now is:
    # (1) Go over all pairs of leafs, (2) find the first common branch point and
    # (3) use their respective distances to that branch point to fill in the
    # missing values in the matrix. We'll be using several stop conditions to
    # avoid doing the same work twice!
    # One important note:
    # We can't use `prange` here because filling the matrix with threads
    # messes with our stop conditions!
    for idx1 in range(N3):
        l1 = leafs_view[idx1]
        for idx2 in range(N3):
            l2 = leafs_view[idx2]
            # Skip if we already visited this (i.e. we already did l2->l1)
            if dists_view[l1, l2] >= 0:
                continue
            # Now find the first common branch point of the two leafs
            # (avoid overshooting the root)
            node = l2
            while parents_view[node] >= 0 and dists_view[l1, node] < 0:
                node = parents_view[node]

            # If the dist is still <0 then these two leafs never converge
            if dists_view[l1, node] < 0:
                continue

            # Now walk towards the common branch point for both leafs and
            # sum up the respective distances to the root nodes
            node1 = l1
            node2 = l2
            while node1 != node and node1 >= 0:
                # Stop early if we meet a node pair we've already visited
                if dists_view[node1, node2] >= 0:
                    break
                while node2 != node and node2 >= 0:
                    # Stop early if we meet a node pair we've already visited
                    if dists_view[node1, node2] >= 0:
                      break
                    d = dists_view[node1, node] + dists_view[node2, node]
                    dists_view[node1, node2] = d
                    dists_view[node2, node1] = d

                    # Move one down on branch 2
                    node2 = parents_view[node2]
                # Move one down on branch 1
                node1 = parents_view[node1]
                # Reset position on branch 2
                node2 = l2

    return dists


@cython.boundscheck(False)
@cython.wraparound(False)
def _dist_to_root(long[::1] parents, long node, weights=None):
    """Return path length from given node to root.

    Parameters
    ----------
    parents :   (N, ) int32 (long) array
                Indices (!) of parent node for each node.
    node :      int
                Index of start node.
    weights :   (N, ) float32 (double) array, optional
                Weights for each child -> parent connection. If ``None`` each
                step will increment distance by 1.

    Returns
    -------
    dist :      float32
                Distance to root.

    """
    cdef long use_weights
    cdef double dist
    cdef Py_ssize_t N = len(parents)
    cdef long[::1] parents_view = parents

    if isinstance(weights, type(None)):
        use_weights = 0
    else:
        use_weights = 1
        weights = np.asarray(weights).astype('double')
    cdef double[::1] weights_view = weights

    # Walk from node to root
    dist = 0
    while node >= 0:
        if use_weights:
            dist += weights_view[node]
        else:
            dist += 1
        node = parents_view[node]

    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
def _all_dists_to_root(long[::1] parents, sources=None, weights=None):
    """Return path length from given nodes to root.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.
    sources :   (N, ) array, optional
                Array of node indices (!) from which to calculate distances. If
                ``None`` will calculate distances to root for all nodes.
    weights :   (N, ) float32 array, optional
                Weights for each child -> parent connection. If ``None`` each
                step will increment distance by 1. For roots the distance
                should be 0!

    Returns
    -------
    dists :     (N, ) float32 (double) array
                Distances to root.

    """
    if isinstance(sources, type(None)):
        sources = np.arange(len(parents), dtype='long')

    cdef long i, node, use_weights
    cdef Py_ssize_t N = len(sources)
    dists = np.zeros(len(sources), dtype='double')
    cdef double[::1] dists_view = dists
    cdef long[::1] parents_view = parents
    cdef long[::1] sources_view = sources

    if isinstance(weights, type(None)):
        use_weights = 0
        dists[:] = -1   # this offset the fact that even root gets +1 later on
    else:
        use_weights = 1
        weights = np.asarray(weights).astype('double')
    cdef double[::1] weights_view = weights

    # Note to self:
    # There might be mileage in using a single thread (i.e. not prange) but
    # checking whether a node's parent has already been traversed in which case
    # we can simply sum up the distances.
    for i in prange(N, nogil=True):
        # Walk from node to root
        node = sources_view[i]
        while node >= 0:
            if use_weights:
                dists_view[i] += weights_view[node]
            else:
                dists_view[i] += 1
            node = parents_view[node]

    return dists


@cython.boundscheck(False)
@cython.wraparound(False)
def _node_indices(long[::1] A, long[::1] B):
    """For each node ID in A find its index in B.

    Typically `A` will be parent IDs and `B` will be node IDs.
    Negative IDs (= parents of root nodes) will be passed through.

    Note that there is no check whether all IDs in A actually exist in B. If
    an ID in A does not exist in B it gets a negative index (i.e. like roots).
    """
    cdef long i, k
    cdef Py_ssize_t lenA = len(A)
    cdef Py_ssize_t lenB = len(B)

    indices = np.empty(len(A), dtype='long')
    cdef long[::1] indices_view = indices
    cdef long[::1] A_view = A
    cdef long[::1] B_view = B


    for i in prange(lenA, nogil=True):
        if A_view[i] < 0:
            indices_view[i] = -1
            continue

        for k in range(lenB):
            if A_view[i] == B_view[k]:
                indices_view[i] = k
                break

    return indices


@cython.boundscheck(False)
@cython.wraparound(False)
def _generate_segments(long[::1] parents, weights=None):
    """Generate linear segments while maximizing segment lengths.

    Parameters
    ----------
    parents :   (N, ) array
                Indices (!) of parent node for each node.
    weights :   (N, ) float32 array, optional
                Weights for each child -> parent connection.
                If provided, use those to maximize the segments
                lengths.

    Returns
    -------
    list
                List of arrays of node indices (!).

    """
    cdef long idx, node, i

    seg = np.zeros(len(parents), dtype='long')
    cdef long[::1] seg_view = seg
    seen = np.zeros(len(parents), dtype='long')
    cdef long[::1] seen_view = seen
    cdef long[::1] parents_view = parents

    nodes = np.arange(len(parents))
    leafs = nodes[~np.isin(nodes, parents)]
    cdef long[::1] leafs_view = leafs
    cdef Py_ssize_t N = len(leafs)

    # Sort leafs so that we start with the most distal leafs
    dists = _all_dists_to_root(parents, weights=weights)
    leafs = leafs[np.argsort(dists[leafs])][::-1]

    # Walk from each node to the root
    all_segs = []
    for idx in range(N):
        # Reset segment and counter
        seg_view[:] = -1
        i = 0
        # Start with this leaf node
        node = leafs_view[idx]
        # Iterate until we hit root
        while node >= 0:
            # Add this node to the segment
            seg_view[i] = node

            # Increment counter
            i += 1

            # If the node has already been seen
            if seen_view[node]:
                break

            # Add this node to the seen nodes
            seen_view[node] = 1

            # Pick the next node
            node = parents_view[node]

        # Track this segment
        all_segs.append(nodes[seg[:i]])

    # Sort segments by length
    all_segs = sorted(all_segs,
                      key=lambda x: dists[x[0]] - dists[x[len(x) - 1]],
                      reverse=True)

    return all_segs
