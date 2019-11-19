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

# Load requirements from requirements.txt
try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements


def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]


version = ml4chem.__version__
install_requires = load_requirements("requirements.txt")

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
    install_requires=install_requires,
    scripts=["bin/ml4chem"],
    data_files=[("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
