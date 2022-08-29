from setuptools import setup, find_packages
from setuptools.command.install import install

# TODO fix this so we actually have a version.py
__version__ = "0.1.0-0"  # This will get replaced when reading version.py


with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='distrib-rl',
    packages=find_packages(),
    version=__version__,
    description='A distributed reinforcement learning platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matthew Allen',
    url='https://github.com/AechPro/distrib-rl',
    install_requires=[
        'gym==0.25.0',
        'pyjson5==1.6.1',
        'lz4==4.0.0',
        'matplotlib>=3.1, <4',
        'msgpack>=1.0.2, <2',
        'msgpack-numpy==0.4.8',
        'numpy>=1.21.4, <2',
        'psutil==5.8.0',
        'redis==3.5.3',
        'scipy>=1.8.0, <2.0.0',
        'torch==1.12.1',
        'trueskill==0.4.5',
        'wandb==0.13.1'
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['reinforcement-learning', 'reinforcement-learning-algorithms', 'gym', 'machine-learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        "Operating System :: Microsoft :: Windows",
    ],
    package_data={}
)
