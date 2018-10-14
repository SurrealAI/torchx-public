import os
from setuptools import setup, find_packages


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


setup(
    name='torchx',
    version='0.9',
    author='Jim Fan',
    url='http://github.com/SurrealAI/TorchX',
    description='PyTorch on steroids',
    long_description=read('README.rst'),
    keywords=['Deep Learning',
              'Machine Learning'],
    license='GPLv3',
    packages=[
        package for package in find_packages() if package.startswith("torchx")
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.5',
    include_package_data=True,
    zip_safe=False
)
