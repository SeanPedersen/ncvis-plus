"""Build the C++ core as the `ncvis_plus._core` extension (metadata lives in pyproject.toml)."""

import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform == "darwin":  # Apple clang needs libomp (brew install libomp)
    compile_args = ["-O3", "-std=c++17", "-ffast-math", "-Xpreprocessor", "-fopenmp"]
    link_args = ["-lomp"]
elif sys.platform == "win32":
    compile_args = ["/O2", "/fp:fast", "/openmp", "/std:c++17"]
    link_args = []
else:
    compile_args = ["-O3", "-std=c++17", "-ffast-math", "-fopenmp"]
    link_args = ["-fopenmp"]

core = Extension(
    "ncvis_plus._core",
    sources=["src/ncvis_plus/_core.pyx", "csrc/ncvis.cpp"],
    include_dirs=[numpy.get_include(), "csrc", "lib/hnswlib", "lib/pcg-cpp/include"],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(ext_modules=cythonize([core], language_level=3))
