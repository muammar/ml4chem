
==============
Introduction
==============
Data is central in Machine Learning and ML4Chem provides some tools to
prepare your datasets. We support the following:

1. `Atomic Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`_.

We will be adding support to other libraries, soon. 


===================
Data Handler
===================

.. contents:: :local:

The :mod:`ml4chem.data.handler` module allows users to adapt data to the
right format to inter-operate with any other module of Ml4Chem.

Its usage is very simple::

    from ml4chem.data.handler import DataSet
    from ase.io import Trajectory

    images = Trajectory("images.traj")
    data_handler = DataSet(images, purpose="training")
    traing_set, targets = data_handler.get_images(purpose="training")

In the example above, an ASE trajectory file is loaded into memory and passed
as an argument to instantiate the ``DataSet`` class with
``purpose="training"``. The ``.get_images()`` class method returns a hashed
dictionary with the molecules in ``images.traj`` and the ``targets`` variable
as a list of energies.

For more information please refer to :mod:`ml4chem.data.handler`.

===================
Visualization
===================

We also offer a :mod:`ml4chem.data.visualization` module to plot interesting
graphics about your model, features, or even monitor the progress of the loss
function and error minimization.

Two backends are supported to plot in ML4Chem: Seaborn and Plotly. 

An example is shown below::

    from ml4chem.data.visualization import plot_atomic_features
    plot_atomic_features("latent_space.db", 
                         method="tsne", 
                         dimensions=3, 
                         backend="plotly")

This will produce an interactive plot with plotly where dimensionality was
reduced using T-SNE.

.. image:: https://raw.githubusercontent.com/muammar/ml4chem/master/docs/source/_static/tsne_visual.png 

