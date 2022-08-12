from setuptools import setup

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
    install_requires=[
        "bs4 >= 0.0.1",
        "dash >= 2.3.1",
        "dash-bootstrap-components >= 1.2.1",
        "dash-core-components >= 2.0.0",
        "dash-html-components >= 2.0.0",
        "matplotlib >= 3.5.1",
        "numpy >= 1.22.3",
        "pandas >= 1.4.2",
        "pyinstaller >= 5.2",
        "scipy >= 1.8.0",
        "simplejson >= 3.17.6",
        "sklearn >= 0.0",
        "steamctl >= 0.9.1",
        "torch >= 1.11.0",
        "tqdm >= 4.64.0"
    ],
)