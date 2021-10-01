import numpy as np

from ._vertex_cython import _vertex_similarity


__all__ = ['vertex_similarity', "_vertex_similarity_numpy"]


def vertex_similarity(mat, C1=0.5, C2=1):
    """Calculate vertex similarity between two vectors.

    Parameters
    ----------
    mat :    (N, M) array
             Array of M observations for N neurons. Typically an adjacency
             matrix. Currently, Will be automatically converted to int32.
    C1,C2 :  float
             Tuning parameters.

    Returns
    -------
    (N, N)   array
             Pairwise vertex similarity.

    """
    mat = np.asarray(mat, dtype=np.int32)
    return np.array(
        _vertex_similarity(
            mat,
            C1=np.float64(C1),
            C2=np.float64(C2)
        )
    )


def _vertex_similarity_numpy(mat, C1=0.5, C2=1):
    """Calculate vertex similarity between two vectors.

    This is the reference implementation using pure Python + numpy.
    """
    sims = np.empty(shape=(mat.shape[0], mat.shape[0]), dtype=np.float32)
    for i in range(len(mat)):
        vecA = mat[i]
        for k in range(len(mat)):
            vecB = mat[k]
            # np.minimum is much faster than
            # np.min(np.vstack(vecA, vecB), axis=1) here
            this_max = np.maximum(vecA, vecB)
            this_min = np.minimum(vecA, vecB)

            # Implement: f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
            vs = this_min - C1 * this_max * np.exp(- C2 * this_min)

            # Sum over all partners
            sims[i][k] = vs.sum()
    return sims
