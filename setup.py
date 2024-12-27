from setuptools import find_packages, setup 
from typing import List

IGNORE_E = '-e .'
def get_requirements(file_name:str)->List[str]:
    requirements = []
    with open(file_name) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", '') for req in requirements]

        if IGNORE_E in requirements:
            requirements.remove(IGNORE_E)

    return requirements

setup(
name = 'Prediction of LC50 value',
version = '2.0',
author = 'Hariprasath',
author_email = 'hariprasath167@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)