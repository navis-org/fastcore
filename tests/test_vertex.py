from fastsim.vertex import vertex_similarity, _vertex_similarity_numpy
import numpy as np


def test_same_results():
    rand = np.random.RandomState(47)
    data = np.abs(rand.normal(scale=10, size=(200, 200))).astype(np.int32)
    ref_impl = _vertex_similarity_numpy(data)
    cython_impl = vertex_similarity(data)

    assert np.allclose(ref_impl, cython_impl)
