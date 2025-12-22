from setuptools import setup, find_packages

setup(
    name='tweakfig',
    version='0.1',
    author='Petro Maksymovych',
    author_email='pmax20@gmail.com',
    maintainer='Petro Maksymovych',
    maintainer_email='pmax20@gmail.com',
    packages=find_packages(),
    description='data analysis for Tunneling Andreev Reflection (TAR)',
    long_description=open('README.md').read(),
    install_requires=[],
    url='https://github.com/amplipy/pubplotlib',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
