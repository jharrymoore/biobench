import sys
from setuptools import setup, find_packages
import versioneer


# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
################################################################################
# SETUP
################################################################################


setup(
    name="biobench",
    author="Harry Moore",
    author_email="jhm72@cam.ac.uk",
    description="A package for benchmarking organic machine learning forcefields",
    license="MIT",
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    python_requires=">3.8",  # Python version restrictions
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "biobench = biobench.entrypoint:main",
        ]
    },
)
