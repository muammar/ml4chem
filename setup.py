import setuptools
import ml4chem

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/README.md", "r") as fh:
        long_description = fh.read()


version = ml4chem.__version__

setuptools.setup(
    name="ml4chem",
    version=version,
    author="Muammar El Khatib",
    author_email="muammarelkhatib@gmail.com",
    description="Machine learning for chemistry and materials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muammar/ml4chem",
    packages=setuptools.find_packages(),
    scripts=["bin/ml4chem"],
    data_files = [("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
