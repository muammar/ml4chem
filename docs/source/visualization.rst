
===================
Visualization
===================
.. contents:: :local:

We also offer a :mod:`ml4chem.visualization` module to plot interesting
graphics about your model, features, or even monitor the progress of the loss
function and error minimization.

Two backends are supported to plot in ML4Chem: Seaborn and Plotly.

An example is shown below::

    from ml4chem.visualization import plot_atomic_features
    fig = plot_atomic_features("latent_space.db",
                               method="pca",
                               dimensions=3,
                               backend="plotly")
    fig.write_html("latent_example.html")

This will produce an interactive plot with plotly where dimensionality was
reduced using PCA, and an html with the name `latent_example.html` is
created.

.. raw:: html
   :file: _static/pca_visual.html

To activate plotly in Jupyter or JupyterLab follow the instructions shown in
`https://plot.ly/python/getting-started/#jupyter-notebook-support <https://plot.ly/python/getting-started/#jupyter-notebook-support>`_

If plotly is not rendering correctly you need to install the jupyter
extension::

    jupyter labextension install @jupyterlab/plotly-extension