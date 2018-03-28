import os
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


setup(
    name='torchx',
    version='0.0.1',
    author='Jim Fan',
    url='http://github.com/SurrealAI/TorchX',
    description='PyTorch on steroids',
    # long_description=read('README.rst'),
    keywords=['Deep Learning',
              'Machine Learning'],
    license='GPLv3',
    packages=['torchx'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
    include_package_data=True,
    zip_safe=False
)
