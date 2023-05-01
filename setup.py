from setuptools import setup, find_packages

setup(
        name='project3',
        version='1.0',
        author='Shiv Patel',
        author_email='shiv.s.patel@ou.edu',
        packages=find_packages(exclude=('tests', 'docs')),
        setup_requires=['pytest-runner'],
        tests_require=['pytest']    
)