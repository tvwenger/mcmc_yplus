from setuptools import setup

setup(
    name="mcmc_yplus",
    version="1.0",
    description="Derive y+ via MCMC",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["mcmc_yplus"],
    install_requires=["numpy", "scipy", "matplotlib", "pymc", "corner", "scikit-learn"],
)
