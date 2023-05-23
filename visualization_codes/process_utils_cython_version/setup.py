from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize('image_process_utils_cython.pyx',compiler_directives={'language_level': 3}))