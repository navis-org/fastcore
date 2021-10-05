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

    # One final increment because we are re-using idx1
    # as length for source_path_view
    idx += 1

    # Return now if we met `target` on the way
    if p_source == target:
        return source_path[:idx]

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def _node_indices(long[::1] A, long[::1] B):
    """For each node ID in A find the index in B.

    Typically `A` will be parent IDs and `B` will be parent IDs.
    Negative IDs (i.e. root parents) will be passed through.

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
