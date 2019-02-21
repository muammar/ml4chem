import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlchem",
    version="0.0.0",
    author="Muammar El Khatib",
    author_email="muammarelkhatib@gmail.com",
    description="Machine learning for chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muammar/mlchem",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
