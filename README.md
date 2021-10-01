# navis-fastcore [WIP]
Fast core functions for `navis` re-implemented in Cython.

The idea is that `navis` will use `fastcore` if installed and fall back to
the pure-Python / numpy functions if not.

Currently implemented:
- vertex similarity (Jarrell et al., 2012)

## Installation
I'm still figuring out the best way for building and packaging pre-compiled
binaries (i.e. wheels). For now, you will need to compile it yourself during
setup. This requires a C-compiler to be present (see
[here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html)) for
a very brief explanation.

```bash
pip3 install git+git://github.com/navis-org/fastcore@main
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
