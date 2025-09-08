from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(include=["src", "src.*"]),  # Finds both src/
    include_package_data=True,
    package_dir={
        "": ".",  # Map root-level packages (src) correctly
        "src": "src",  # Map src package
    },
    version='0.1.1',
    description='This project is a starting Pack for MLOps project. It is not perfect so feel free to make some modifications on it.',
    author='DataScientest',
    license='MIT',
)
