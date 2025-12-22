from setuptools import setup, find_packages

setup(
    name='tweakfig',
    version='0.2.0',
    author='Petro Maksymovych',
    author_email='pmax20@gmail.com',
    maintainer='Petro Maksymovych',
    maintainer_email='pmax20@gmail.com',
    packages=find_packages(),
    description='Quick matplotlib figure enhancement for publication-ready plots',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'pillow',
    ],
    extras_require={
        'dev': ['pytest', 'jupyter'],
    },
    url='https://github.com/amplipy/tweakfig',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.8',
)
