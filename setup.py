from setuptools import setup, find_packages

VERSION = "0.0.1"

def read_requirements():
    with open("requirements.txt", "r") as f:
        content = f.readlines()
    return [s.split("\n")[0] for s in content]


setup(name='sparse_identification',
      version=VERSION,
      packages=find_packages(),
      author="Jean-Christophe Loiseau",
      author_email='loiseau.jc@gmail.com',
      description='Python library designed for sparse identification of nonlinear dynamical system',
      classifiers=["Programming Language :: Python",
                   "Topic :: Fluid dynamics"],
      requires=read_requirements()
      )
