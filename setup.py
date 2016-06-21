from setuptools import setup, find_packages
import sparse_identification as sp

setup(name='sparse_identification',
      version=sp.__version__,
      packages=find_packages(),
      author="Jean-Christophe Loiseau",
      author_email='loiseau.jc@gmail.com',
      description='Python library designed for sparse identification of nonlinear dynamical system',
      classifiers=["Programming Language :: Python",
                   "Topic :: Fluid dynamics"], requires=['numpy', 'scipy', 'cvxopt']
      )
