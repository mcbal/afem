from setuptools import find_packages, setup

setup(
    name='afem',
    packages=find_packages(exclude=['images']),
    version='0.0.0',
    license='MIT',
    description='Approximate Free-Energy Minimization',
    author='Matthias Bal',
    author_email='matthiascbal@gmail.com',
    url='https://github.com/mcbal/afem',
    keywords=['artificial intelligence', 'attention mechanism', 'free energy', 'partition function', 'transformers'],
    install_requires=['einops>=0.3', 'numpy>=1.19', 'torch>=1.9'],
)
