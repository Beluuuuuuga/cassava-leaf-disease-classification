import os
from setuptools import setup, find_packages


def load_requirements(f):
    return list(
        filter(None,
               [l.split("#", 1)[0].strip() for l in open(os.path.join(os.getcwd(), f)).readlines()]))


README = os.path.join(os.path.dirname(__file__), 'README.md')

setup(
    name="cassava",
    description="cassava",
    long_description=read_md(README),
    version=VERSION,
    install_requires=load_requirements("./requirements.txt"),
    packages=find_packages(exclude=('nobtebook'))
)
