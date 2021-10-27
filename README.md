[![Tests](https://github.com/navis-org/fastcore/actions/workflows/tests.yaml/badge.svg)](https://github.com/navis-org/fastcore/actions/workflows/tests.yaml) [![CI](https://github.com/navis-org/fastcore/actions/workflows/ci.yaml/badge.svg)](https://github.com/navis-org/fastcore/actions/workflows/ci.yaml)

# navis-fastcore
Fast core functions for [`navis`](https://github.com/navis-org/navis)
re-implemented in Cython.

The idea is that `navis` will use `fastcore` if installed and fall back to
the pure-Python / numpy implementation if not.

Currently implemented:
- vertex similarity (Jarrell et al., 2012)
- shortest path from source to target (~40x faster than iGraph)
- geodesic distance matrix (up to 100x faster than scipy)

See further down for details.

## Installation
I'm still figuring out the best way for building and packaging pre-compiled
binaries (i.e. wheels). For now, you will need to compile it yourself during
setup. This requires a C-compiler to be present (see
[here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) for
a very brief explanation).

```bash
$ pip3 install git+git://github.com/navis-org/fastcore@main
```

## Examples

```python
>>> import numpy as np
>>> import navis
>>> import fastcore
>>> # Grab an example skeleton
>>> n = navis.example_neurons(1)
>>> # Time navis' scipy-based function for all-by-all geodesic distances
>>> %time m1 = navis.geodesic_matrix(n, weights=None)
CPU times: user 4.58 s, sys: 153 ms, total: 4.73 s
Wall time: 4.73 s
>>> # Time the analogous function in fastcore
>>> %time m2 = fastcore.geodesic_matrix(n.nodes.node_id.values, n.nodes.parent_id.values)
CPU times: user 2.17 s, sys: 173 ms, total: 2.35 s
Wall time: 258 ms
>>> # Make sure results are the same
>>> np.all(m1 == m2)
True
```

### Troubleshooting

#### Compiler does not support openmp

We need `openmp` for threaded processing. Without it, this is not much faster
than the pure numpy implementations. If your compiler does not support
`openmp`, you will get an error along the lines of `-fopenmp  not supported`.
This happens e.g. on OSX if you use the clang bundled with XCode. In my case,
I was able to work around it by installing `llvm` with homebrew and then
adding a couple flags to my `~/.bash_profile` to make sure the homebrew llvm
is actually used:

```
export PATH="/usr/local/opt/llvm/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
```

### Develop

To compile the extensions in place:

```bash
$ python setup.py build_ext --inplace
```
