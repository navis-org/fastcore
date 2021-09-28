from setuptools import Extension, setup
from Cython.Build import build_ext

ext_modules = [
    Extension(
        "fastsim._vertex_cython",
        ["fastsim/_vertex_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='fastsim',
    cmdclass={'build_ext': build_ext},
    install_requires=["numpy"],
    ext_modules=ext_modules,
)
