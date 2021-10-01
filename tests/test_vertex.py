import numpy as np
import pytest
from fastcore.vertex import vertex_similarity, _vertex_similarity_numpy


@pytest.mark.parametrize(
    ["fn"], [[vertex_similarity], [_vertex_similarity_numpy]]
)
def test_same_results(fn):
    rand = np.random.RandomState(47)
    data = np.abs(rand.normal(scale=10, size=(200, 200))).astype(np.int32)
    ref_impl = _vertex_similarity_numpy(data)
    test_impl = fn(data)

    assert np.allclose(ref_impl, test_impl)


@pytest.mark.parametrize(
    ["fn"], [[vertex_similarity], [_vertex_similarity_numpy]]
)
def test_benchmark(fn, benchmark):
    rand = np.random.RandomState(47)
    data = np.abs(rand.normal(scale=10, size=(500, 500))).astype(np.int32)
    data = benchmark(fn, data)
