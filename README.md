# navis-fastcore [WIP]
Fast core functions for navis re-implemented in Cython.

Currently implemented:
- vertex similarity (Jarrell et al., 2012)

## Installation
I'm still figuring out the best way for building and packaging. For now,
you will need to do it yourself:

1. Clone this repository
2. Install `cython`: `pip3 install cython`
3. CD into the repo
4. Run `python setup.py build_ext --inplace`

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
