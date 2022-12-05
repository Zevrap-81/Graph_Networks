from setuptools import find_packages, setup

setup(
    name='graph-networks',
    packages=find_packages(exclude=("test", "examples", "saved_data", "env")),
    version='1.0.0',
    description="""A Thesis Project employing different graph-neural 
                   network models to learn mesh-based deep-drawing simulations.""",
    author='Parvez Mohammed',
    license='MIT',
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={'train': 'Trainer'},
    test_suite='tests'
    )