import os
from setuptools import setup, find_packages


def load_requirements(f):
    return list(
        filter(None,
               [l.split("#", 1)[0].strip() for l in open(os.path.join(os.getcwd(), f)).readlines()]))

# README from README.md
try:
    from pypandoc import convert

    def read_md(f):
        return convert(f, 'rst')
except ImportError:
    convert = None
    print("warning : Could not find module named pypandoc.")

    def read_md(f):
        return open(f, 'r').read()

README = os.path.join(os.path.dirname(__file__), 'README.md')

setup(
    name="mylib",
    description="mylib",
    long_description=read_md(README),
    version=0.0.1,
    install_requires=load_requirements("./requirements.txt"),
    packages=find_packages(exclude=('nobtebook'))
)
