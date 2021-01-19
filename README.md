![alt text](https://raw.githubusercontent.com/muammar/ml4chem/master/docs/source/_static/ml4chem.png "Logo")

--------------------------------------------------------------------------------

## About
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/muammar/mkchromecast/)
[![Build Status](https://travis-ci.com/muammar/ml4chem.svg?branch=master)](https://travis-ci.com/muammar/ml4chem)
[![License](https://img.shields.io/badge/license-BSD-green)](https://github.com/muammar/ml4chem/blob/master/LICENSE)
[![Downloads](https://img.shields.io/github/downloads/muammar/ml4chem/total.svg?maxAge=2592000?style=flat-square)](https://github.com/muammar/ml4chem/releases)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ml4chem)
[![GitHub release](https://img.shields.io/github/release/muammar/ml4chem.svg)](https://github.com/muammar/ml4chem/releases/latest)
[![Documentation Status](https://readthedocs.org/projects/ml4chem/badge/?version=latest)](https://ml4chem.readthedocs.io/en/latest/?badge=latest)
[![Slack channel](https://img.shields.io/badge/slack-ml4chem-yellow.svg?logo=slack)](https://ml4chem.slack.com/)



ML4Chem is a package to deploy machine learning for chemistry and materials
science. It is written in Python 3, and intends to offer modern and rich
features to perform machine learning (ML) workflows for chemical physics.

A list of features and ML algorithms are shown below.

- PyTorch backend.
- Completely modular. You can use any part of this package in your project.
- Free software <3. No secrets! Pull requests and additions are more than
  welcome!
- Documentation (work in progress).
- Explicit and idiomatic: `ml4chem.get_me_a_coffee()`.
- Distributed training in a data parallel paradigm aka mini-batches.
- Scalability and distributed computations are powered by Dask.
- Real-time tools to track status of your computations.
- Easy scaling up/down.
- Easy access to intermediate quantities: `NeuralNetwork.get_activations(X, numpy=True)` or `VAE.get_latent_space(X)`.
- [Messagepack serialization](https://msgpack.org/index.html).

## Notes 

This package is under heavy development and might break at some points until
it gets stabilized. It is in its infancy, so if you find there is an error,
you might want to report it so that it can be improved. We also welcome pull
requests if you find any part of ML4Chem should be improved. That would be
very nice.

## Citing

If you find this software useful, please use this bibtex to cite it:

```
@article{Elkhatib2020ml4chem,
    title={ML4Chem: A Machine Learning Package for Chemistry and Materials Science},
    author={Muammar El Khatib and Wibe A de Jong},
    year={2020},
    eprint={2003.13388},
    archivePrefix={arXiv},
    primaryClass={physics.chem-ph}
}
```

## Documentation

To get started, read the documentation at
[https://ml4chem.dev](https://ml4chem.dev). It is arranged in a way that you
can go through the theory as well as some code snippets to understand how to
use this software. Additionally, you can dive through the [module
index](https://ml4chem.dev/genindex.html) to get more information about
different classes and functions of ML4Chem. If you think the documentation
has to be improved do not hesistate to state so in the bug reports and help
out if you feel like it.


## Visualizations

![](https://raw.githubusercontent.com/muammar/ml4chem/master/docs/source/_static/dask_dashboard.png)

## Copyright

License: BSD 3-clause "New" or "Revised" License.

```
ML4Chem: Machine Learning for Chemistry and Materials (ML4Chem) Copyright (c)
2019, The Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.
```