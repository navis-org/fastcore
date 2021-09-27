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
    cdef int i, k, j
    cdef Py_ssize_t N = mat.shape[0]
    cdef Py_ssize_t M = mat.shape[1]

    result = np.zeros((N, N), dtype=float)
    cdef double[:, ::1] result_view = result

    for i in prange(N, nogil=True):
        for k in range(i, N):
            val = 0
            for j in range(M):
                if mat[i, j] > mat[k, j]:
                    mx, mn = mat[i, j], mat[k, j]
                else:
                    mx, mn = mat[k, j], mat[i, j]

                val += vsim(mx, mn, C1, C2)

            result_view[i, k] += val
            if i != k:
                result_view[k, i] += val

    return result
