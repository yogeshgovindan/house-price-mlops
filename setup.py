from setuptools import find_packages, setup

setup(
    name="house_price",
    version="0.0.1",
    author="Yogesh",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
