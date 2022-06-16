from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="error_cython", sources=["error_cython.pyx"])
setup(ext_modules=cythonize(ext))