"""Build Cython extension for bitboard hot paths.

Usage:
    python3 robottler/bitboard/setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "robottler.bitboard._fast",
    ["robottler/bitboard/_fast.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
    ),
)
