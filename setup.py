import sys
from setuptools import setup, find_packages


# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
################################################################################
# SETUP
################################################################################


setup(
    name="biobench",
    version="0.1",
    author="Harry Moore",
    author_email="harry@angstrom-ai.com",
    description="A package for benchmarking organic machine learning forcefields",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.10",
    requires=["pubchemprops"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "biobench = biobench.biobench:main",
        ]
    },
)
