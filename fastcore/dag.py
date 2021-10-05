"""Module containing functions for directed acyclic graphs (DAG).

DAGs are characterized by every node having exactly one parent (except for the
root node which has no parents). This property makes some problems (such as
graph traversal) much easier than for general graphs.

"""

import numpy as np

from ._dag import _node_indices, _shortest_path_undirected, _shortest_path_directed


def shortest_path(node_ids, parent_ids, source, target, directed=False):
    """Find shortest path between two nodes.

    This implementation is ~40x faster than igraph's `get_shortest_paths` and
    ~180x faster than `networkx.shortest_path` (both of which generalize to
    non-DAGs).

    Parameters
    ----------
    node_ids :  (N, ) array
                Array of node IDs.
    parents :   (N, ) array
                Array of parent IDs for each node. Root nodes' parents must be -1.
    source :    int
                ID of source node.
    target :    int
                ID of target node.
    directed :  bool
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
    # Make sure we have the correct data types
    node_ids = node_ids.astype('long', order='C', copy=False)
    parent_ids = parent_ids.astype('long', order='C', copy=False)

    # Convert parent IDs into indices
    parent_ix = _node_indices(parent_ids, node_ids)

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


class PathError(BaseException):
    """Error indicating that there is no path between source and target."""

    pass
