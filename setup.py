from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'  # Ignoring -e . as it is not a package & I donot want to install it

# file_path in string form and will return a List of string type
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    # readlines() will read each line from the file and append it to the requirements
        requirements = [req.replace("\n", "") for req in requirements] 
    # replacing the newline character with blank
    # by doing this I will have the info. of my packages and not the \n

    # to ignore -e . 
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements

setup(
    name = 'Diamond Price Prediction',
    version = '0.0.1',
    author = 'Chirag Verma',
    author_email = 'chirag.yep@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()
)

# install_requires tells us about what libraries I need to install while building my package.