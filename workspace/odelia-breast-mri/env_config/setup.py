from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# with open('requirements.txt', encoding='utf-8') as f:
#     install_requires = f.read()

setup(
    name='ODELIA - Breast MRI Classification',
    author="Gustav Müller-Franzes",
    maintainer="Jeff",
    version="0.1.0",
    description="Baseline code for breast MRI classification", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    # install_requires=install_requires,
)