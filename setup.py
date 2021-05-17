from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = ['numpy', 'sympy']

if get_dist('tensorflow') is None and get_dist('tensorflow_gpu') is None:
    install_deps.append('tensorflow')

setup(
    name='catamount',
    version='0.9',
    description='Catamount: Compute Graph Analysis Tool',
    author='Catamount Developers',
    author_email='joel@baidu.com',
    packages=find_packages(),
    install_requires=install_deps,
)
