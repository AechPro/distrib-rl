from setuptools import setup, find_packages
from setuptools.command.install import install


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
        'absl-py==0.7.1',
        'gym==0.21.0',
        'gym3==0.3.3',
        'rlgym==1.2.0',
        'pyjson5==1.6.1',
        'lz4==4.0.0',
        'matplotlib>=3.1, <4',
        'MinAtar==1.0.10',
        'msgpack==1.0.2',
        'msgpack-numpy==0.4.7.1',
        'numpy==1.21.4',
        'pandas==1.3.5',
        'psutil==5.8.0',
        'pywin32==228',
        'redis==3.5.3',
        'scipy>=1.8.0, <2.0.0',
        'rlgym-tools==1.7.0',
        'stable-baselines3==1.5.0',
        'torch==1.11.0',
        'torchvision==0.12.0',
        'trueskill==0.4.5',
        'typing-extensions==3.7.4.3',
        'typing-inspect==0.6.0',
        'zuper-typing-z6==6.1.8',
        'wandb==0.10.33'
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
