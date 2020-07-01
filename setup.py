from setuptools import setup

setup(
    name='VectorGraphingToolkit',
    version='0.0.1-beta',
    license='MIT',
    author='Braedyn Lettinga',
    author_email='braedynlettinga@gmail.com',
    description='A simplistic vector/vector field visualization tool built on top of matplotlib.',
    url='https://github.com/braedynl/VectorGraphingToolkit',
    packages=['vgtk'],
    install_requires=[
        'sympy',
        'matplotlib',
        'numpy',
        'scipy'
    ]
)
