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
    python_requires='>=3.9, <3.11',
    include_package_data=True,
    install_requires=[
        'setuptools>=49.2.1',
        'attrs>=21.4.0',
        'certifi>=2021.10.8',
        'charset-normalizer>=2.0.12',
        'click>=8.0.3',
        'click-plugins>=1.1.1',
        'cligj>=0.7.2',
        'convertdate>=2.4.0',
        'cycler>=0.11.0',
        'Fiona>=1.8.21',
        'fonttools>=4.29.1',
        'geopandas>=0.10.2',
        'hijri-converter>=2.2.3',
        'holidays>=0.13',
        'idna>=3.3',
        'joblib>=1.1.0',
        'kiwisolver>=1.3.2',
        'korean-lunar-calendar>=0.2.1',
        'matplotlib>=3.5.1',
        'munch>=2.5.0',
        'numpy>=1.21.5,<2',
        'openpyxl>=3.0.9',
        'packaging>=21.3',
        'pandas>=1.3.5',
        'Pillow>=9.0.1',
        'PyMeeus>=0.5.11',
        'pyparsing>=3.0.7',
        'pyproj>=3.2.1',
        'python-dateutil>=2.8.2',
        'pytz>=2021.3',
        'PyYAML>=6.0',
        'requests>=2.27.1',
        'Rtree>=0.9.7',
        'scikit-learn==1.0.2',
        'scipy>=1.7.3',
        'Shapely>=1.8.0',
        'six>=1.16.0',
        'threadpoolctl>=3.1.0',
        'urllib3>=1.26.8',
        'fastparquet>=0.8.3'
    ],
    extras_require={
        'dev': [
            'build>=0.5.1',
            'nbsphinx>=0.8.6',
            'recommonmark>=0.7.1',
            'setuptools>=57.0.0',
            'sphinx>=4.0.2',
            'sphinx-panels>=0.6.0',
            'sphinx-rtd-theme>=1.0.0',
            'twine>=3.4.1',
            'pytest>=6.2.4',
            'pytest-cov>=2.12.1'
        ]
    }
)
