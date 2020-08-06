from setuptools import setup

setup(
    name='VectorGraphingToolkit',
    version='0.1.0',
    license='MIT',
    author='Braedyn Lettinga',
    author_email='braedynlettinga@gmail.com',
    description='A simplistic vector/vector field visualization tool built on top of matplotlib.',
    url='https://github.com/braedynl/VectorGraphingToolkit',
    packages=['vgtk'],
    install_requires=[
        'sympy>=1.6',
        'matplotlib>=3.2.1',
        'numpy>=1.17.10',
        'scipy>=1.4.1'
    ]
)
