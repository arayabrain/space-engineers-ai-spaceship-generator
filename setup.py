from setuptools import setup

import configparser
import os

config = configparser.ConfigParser()
curr_dir = os.getcwd()
config.read(os.path.join(curr_dir, 'configs.ini'))

USE_TORCH = config['LIBRARY'].getboolean('use_torch')

base_dependencies = [
    "bs4 >= 0.0.1",
    "dash >= 2.4.0",
    "dash-bootstrap-components >= 1.2.1",
    "dash-core-components >= 2.0.0",
    "dash-html-components >= 2.0.0",
    "kaleido >= 0.2.1",
    "matplotlib >= 3.5.1",
    "numpy >= 1.22.3",
    "pandas >= 1.4.2",
    "pyinstaller >= 5.2",
    "scipy >= 1.8.0",
    "simplejson >= 3.17.6",
    "sklearn >= 0.0",
    "statsmodels >= 0.13.2",
    "steamctl >= 0.9.1",
    "tqdm >= 4.64.0",
    "typing_extensions >= 4.3.0",
    "waitress >= 2.1.2"
    ]

if USE_TORCH:
    base_dependencies.append("torch >= 1.11.0")

setup(
    name='PCGSEPy',
    version='0.0.1',
    author='Roberto Gallotta',
    author_email='roberto_gallotta@araya.org',
    packages=['pcgsepy'],
    scripts=[],
    url='https://github.com/arayabrain/space-engineers-ai-spaceship-generator',
    license='LICENSE.md',
    description='PCG Python package for Space Engineers',
    long_description='This package provides methods and classes to run a Procedural Content Generation task in the videogame Space Engineers.',
    install_requires=base_dependencies
)