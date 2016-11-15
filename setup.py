from setuptools import setup, find_packages
import os

VERSION = "0.0.1"

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as f:
        content = f.read()
    return content.split("\n")


setup(name='sparse_identification',
      version=VERSION,
      packages=find_packages(),
      author="Jean-Christophe Loiseau",
      author_email='loiseau.jc@gmail.com',
      description='Python library designed for sparse identification of nonlinear dynamical system',
      classifiers=["Programming Language :: Python",
                   "Topic :: Fluid dynamics"],
      install_requires=read_requirements()
      )
