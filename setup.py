from setuptools import setup
import re


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="mcmc_yplus",
    version=get_property("__version__", "mcmc_yplus"),
    description="Bayesian model of helium abundance in ionized gas",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["mcmc_yplus"],
    install_requires=install_requires,
    python_requires=">=3.9",
    license="GNU GPLv3",
    url="https://github.com/tvwenger/mcmc_yplus",
)
