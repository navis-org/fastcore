import numpy as np
from fastcore import dag

# A test neuron with:
# - 20 nodes
# - 2 roots (= 2 disconnected pieces)
# - 3 branch points
nodes = np.arange(20)
parents = np.array([1, 2, 3, 4, 5, 6, 7, 8, -1, 10, 11, 12, 4, 14, 2,
                    16, 17, 18, -1, 16])


def test_geodesic_matrix():
    # Simple matrix
    m = dag.geodesic_matrix(nodes, parents)

    # Make sure diagonal is 0
    assert np.all(np.diag(m) == 0)

    # Make sure there are some -1 values (remember the mock neuron is fragmented)
    assert m.min() == -1

    # Some manual checks
    known_dist = [(0, 8, 8), (9, 8, 8), (19, 18, 3), (0, 19, -1)]
    for s, t, d in known_dist:
        assert m[s, t] == d
        assert m[t, s] == d

    # Now make sure this also works with weights
    weights = np.ones(len(parents))
    m2 = dag.geodesic_matrix(nodes, parents, weights=weights)

    assert np.all(np.diag(m2) == 0)
    assert m2.min() == -1
    assert np.all(m == m2)

    weights2 = np.ones(len(parents)) * 2
    m3 = dag.geodesic_matrix(nodes, parents, weights=weights2)

    assert np.all(np.diag(m3) == 0)
    assert m3.min() == -1
    for s, t, d in known_dist:
        assert m3[s, t] == d * 2 if d >= 0 else d
        assert m3[t, s] == d * 2 if d >= 0 else d
