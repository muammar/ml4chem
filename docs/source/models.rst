===================
Introduction
===================

.. contents:: :local:


The `atomistic` module is designed to deploy models where the atom is the
central object. Machine learning potentials might be one of the best known
cases of atom-centered models. These models are known to be very accurate
to predict energy and atomic forces and are very powerful because
models can generalize to any molecular size as long as the training data
contained atoms with similar chemical environments.

========
Theory
========
The basic idea behind atomistic machine learning is that the prediction of a
molecule or bulk can be obtained as the sum of atomic contributions:

.. math::

   P = \sum_{i=1}^n P_{atom}(R^{local})

where :math:`P_{atom}` is a functional of atomic positions.

==========================
Atomic Features
==========================

ML4Chem support Gaussian symmetry functions and atomic latent features
obtained with the `Autoencoder` class.

  - Gaussian symmetry functions.
  - Atomic latent features.


==========================
Models
==========================

Deep Learning
===============

Neural Networks
----------------

Something here


Autoencoders
-------------

Something here

Support Vector Machines
========================

Kernel Ridge Regression
------------------------

Something here. 

Gaussian Process Regression
------------------------

Something here.

===================
Semi-supervised Learning
===================

Model Merger class
============