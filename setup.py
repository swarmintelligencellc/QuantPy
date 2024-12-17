from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='QuantPy',
    version='0.1',
    author='Joseph Smidt',
    author_email='josephsmidt@gmail.com',
    packages=['quantpy'],
    url='https://github.com/jsmidt/QuantPy',
    license='LICENSE',
    description='A framework for quantitative finance In python',
    long_description=open('README.md').read(),
    install_requires=[
        "pandas >= 0.10.0",
        "matplotlib >= 1.1.0",
    ],
)
