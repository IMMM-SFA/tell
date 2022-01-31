import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('tell/__init__.py').read(), re.M).group(1)


setup(
    name='tell',
    version=version,
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/tell.git',
    license='BSD2-Clause',
    author='Casey McGrath; Casey Burleyson; Chris R. Vernon; Aowabin Rahman',
    author_email='casey.mcgrath@pnnl.gov',
    description='A model to predict total electricity loads',
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.6.*, <4',
    include_package_data=True,
    install_requires=[
        'geopandas >= 0.10.2',
        'numpy >= 1.19.2',
        'openpyxl >= 3.0.9',
        'pandas >= 1.1.2',
        'klib >= 0.0.89',
        'joblib >= 1.0.1',
        'matplotlib >= 3.3.3',
        'seaborn >= 0.11.1',
        'holidays >= 0.11.1',
        'requests >= 2.27.1',
        'scikit-learn >= 0.24.1',
        'scipy >= 1.4.1',
    ],
    extras_require={
        'dev': [
            'build>=0.5.1',
            'nbsphinx>=0.8.6',
            'setuptools>=57.0.0',
            'sphinx>=4.0.2',
            'sphinx-panels>=0.6.0',
            'sphinx-rtd-theme>=0.5.2',
            'twine>=3.4.1',
            'pytest>=6.2.4',
            'pytest-cov>=2.12.1'
        ]
    }
)
