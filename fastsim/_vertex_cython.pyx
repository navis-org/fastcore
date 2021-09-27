import numpy as np

cimport cython
from libc.math cimport exp
from cython.parallel import prange


cdef double vsim(double mx, double mn, double C1, double C2) nogil:
    return mn - C1 * mx * exp(- C2 * mn)


@cython.boundscheck(False)
@cython.wraparound(False)
def _vertex_similarity(int[:, ::1] mat, double C1=0.5, double C2=1):
    """Calculate vertex similarity between two vectors."""
    cdef double mx, mn, val
    cdef int idx1, idx2, obs_idx
    cdef Py_ssize_t N = mat.shape[0]
    cdef Py_ssize_t M = mat.shape[1]

    result = np.zeros((N, N), dtype=float)
    cdef double[:, ::1] result_view = result

    for idx1 in prange(N, nogil=True):
        for idx2 in range(idx1, N):
            val = 0
            if idx1 == idx2:
                for obs_idx in range(M):
                    mx = mat[idx1, obs_idx]
                    result_view[idx1, idx1] += vsim(mx, mx, C1, C2)

                continue

            for obs_idx in range(M):
                mx = mat[idx1, obs_idx]
                mn = mat[idx2, obs_idx]
                if mn > mx:
                    mn, mx = mx, mn

                val += vsim(mx, mn, C1, C2)

            result_view[idx1, idx2] += val
            result_view[idx2, idx1] += val

    return result
