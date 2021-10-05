import platform

from setuptools import Extension, setup
from Cython.Build import build_ext


if platform.system() == "Windows":
    compile_args = ["/openmp"]
elif platform.system() == "Darwin":
    compile_args = ["-Xpreprocessor", "-fopenmp"]
else:
    compile_args = ["-fopenmp", "-std=c++11"]


ext_modules = [
    Extension(
        "fastcore._vertex",
        ["fastcore/_vertex.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=compile_args,
    ),
    Extension(
        "fastcore._dag",
        ["fastcore/_dag.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=compile_args,
    )
]

setup(
    name='fastcore',
    cmdclass={'build_ext': build_ext},
    install_requires=["numpy"],
    ext_modules=ext_modules,
)
