from setuptools import setup

setup(
    name='PCGSEPy',
    version='0.0.1',
    author='gallorob',
    # author_email='tbd@tmp.com',
    packages=['pcgsepy', 'pcgsepy.common'],
    scripts=[],
    # url='tbd',
    # license='tbd.txt',
    description='PCG Python package for Space Engineers',
    # long_description=open('tbd.txt').read(),
    install_requires=[
        "numpy >= 1.20.1"
    ],
)