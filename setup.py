from setuptools import setup, find_packages

setup(
    name='catamount',
    version='0.9',
    description='Catamount: Compute Graph Analysis Tool',
    author='Catamount Developers',
    author_email='joel@baidu.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sympy',
        'tensorflow>=1.7',
    ],
)
