from setuptools import Extension, setup
from Cython.Build import build_ext

ext_modules = [
    Extension(
        "fastcore._vertex_cython",
        ["fastcore/_vertex_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='fastcore',
    cmdclass={'build_ext': build_ext},
    install_requires=["numpy"],
    ext_modules=ext_modules,
)
