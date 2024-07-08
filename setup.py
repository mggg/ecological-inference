import codecs
import os
import re

import setuptools
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "pyei", "__init__.py")


def get_requirements():
    # TODO: The gh-action to upload to PyPI doesn't include the requirements.txt
    # so we're just copying them here as a backup. This is super brittle!
    try:
        with codecs.open(REQUIREMENTS_FILE) as buff:
            return buff.read().splitlines()
    except:
        return """pymc >= 5.10.0
arviz
scikit-learn
matplotlib
pandas
seaborn
graphviz
numpy
jax
numpyro
jaxlib
numba
netCDF4""".splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))


setup(
    name="pyei",
    version=get_version(),
    description="",
    author="Metric Geometry and Gerrymandering Group",
    url="https://github.com/mggg/ecological-inference",
    packages=find_packages(),
    install_requires=get_requirements(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
)
