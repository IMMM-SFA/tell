from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='tell',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/tell.git',
    license='BSD2-Clause',
    author='Casey McGrath; Chris R. Vernon',
    author_email='casey.mcgrath@pnnl.gov; chris.vernon@pnnl.gov',
    description='A model to predict total electricity loads',
    long_description=readme(),
    python_requires='>=3.6.*, <4',
    install_requires=get_requirements()
)
