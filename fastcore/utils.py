import numpy as np


def node_indices(A, B):
    """For each node ID in A find its index in B.

    Typically `A` will be parent IDs and `B` will be node IDs.
    Negative IDs (= parents of root nodes) will be passed through.

    Note that there is no check whether all IDs in A actually exist in B. If
    an ID in A does not exist in B it gets a negative index (i.e. like roots).
    """
    ix_dict = dict(zip(B, np.arange(len(B))))

    return np.array([ix_dict.get(p, -1) for p in A])
