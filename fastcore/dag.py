"""Module containing functions for directed acyclic graphs (DAG).

DAGs are characterized by every node having exactly one parent (except for the
root node which has no parents). This property makes some problems (such as
graph traversal) much easier than for general graphs.

"""

import numpy as np

from ._dag import (_node_indices, _shortest_path_undirected,
                   _shortest_path_directed, _geodesic_matrix)


__all__ = ['geodesic_matrix', 'shortest_path']


def shortest_path(node_ids, parent_ids, source, target, directed=False):
    """Find shortest path between two nodes.

    This implementation is ~40x faster than iGraph's `get_shortest_paths` and
    ~180x faster than `networkx.shortest_path` (both of which need to generalize
    to non-DAGs - which we don't).

    Parameters
    ----------
    node_ids :   (N, ) array
                 Array of int32 node IDs.
    parent_ids : (N, ) array
                 Array of int32 parent IDs for each node. Root nodes' parents
                 must be -1.
    source :     int
                 ID of source node.
    target :     int
                 ID of target node.
    directed :   bool
                 If False, will only traverse the graph in child -> parent
                 direction. This effectively means that source has to be
                 distal to target for a path to exist!

    Returns
    -------
    path :      (M, ) array
                Array of node IDs making up the path between source and target.
                The first and the last entry will be `source` and `target`,
                respectively.

    Raise
    -----
    PathError
                If no path between source and target.

    """
    # Some initial sanity checks
    node_ids = np.asanyarray(node_ids)
    parent_ids = np.asanyarray(parent_ids)
    assert node_ids.shape == parent_ids.shape
    assert node_ids.ndim == 1 and parent_ids.ndim == 1

    # Make sure we have the correct data types
    node_ids = node_ids.astype('long', order='C', copy=False)
    parent_ids = parent_ids.astype('long', order='C', copy=False)

    # Convert parent IDs into indices
    parent_ix = _node_indices(parent_ids, node_ids)

    source = int(source)
    target = int(target)
    if source not in node_ids:
        raise ValueError(f'Source "{source}" not in node IDs.')
    if target not in node_ids:
        raise ValueError(f'Target "{target}" not in node IDs.')

    # Translate source and target IDs into indices
    source_ix, target_ix = _node_indices(np.array([source, target],
                                                  dtype='long'),
                                         node_ids)

    # Get the actual path
    if directed:
        path_ix = _shortest_path_directed(parent_ix, source_ix, target_ix)
    else:
        path_ix = _shortest_path_undirected(parent_ix, source_ix, target_ix)

    if not isinstance(path_ix, np.ndarray):
        raise PathError(f'No path between nodes "{source}" and "{target}"')

    # Translate indices back into IDs
    path = node_ids[path_ix]

    return path


def geodesic_matrix(node_ids, parent_ids, weights=None):
    """Calculate all-by-all geodesic distances.

    This implementation is up to 100x faster the implementation in navis (which
    uses scipy's `csgraph`).

    Parameters
    ----------
    node_ids :   (N, ) int32 (long) array
                 Array of int32 node IDs.
    parent_ids : (N, ) int (long) array
                 Array of parent IDs for each node. Root nodes' parents
                 must be -1.
    weights :    (N, ) float32 array, optional
                 Array of distances for each child -> parent connection.
                 If ``None`` all node to node distances are set to 1.

    Returns
    -------
    matrix :    (N, N) float32 (double) array
                All-by-all geodesic distances.

    """
    # Some initial sanity checks
    node_ids = np.asanyarray(node_ids)
    parent_ids = np.asanyarray(parent_ids)
    assert node_ids.shape == parent_ids.shape
    assert node_ids.ndim == 1 and parent_ids.ndim == 1

    # Make sure we have the correct data types
    node_ids = node_ids.astype('long', order='C', copy=False)
    parent_ids = parent_ids.astype('long', order='C', copy=False)

    # Convert parent IDs into indices
    parent_ix = _node_indices(parent_ids, node_ids)

    # Get the actual path
    dists = _geodesic_matrix(parent_ix, weights=weights)

    return dists


class PathError(BaseException):
    """Error indicating that there is no path between source and target."""

    pass
