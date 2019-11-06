import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ml4chem.data.serialization import load
import time


def parity(predictions, true, scores=False, filename=None, **kwargs):
    """A parity plot function

    Parameters
    ----------
    predictions : list or ndarray
        Model predictions in a list.
    true : list or ndarray
        Targets or true values.
    scores : bool
        Print scores in parity plot.
    filename : str
        A name to save the plot to a file. If filename is non existent, we
        call plt.show().

    Notes
    -----
    kwargs accepts all valid keyword arguments for matplotlib.pyplot.savefig.
    """

    min_val = min(true)
    max_val = max(true)
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111)
    ax.plot(true, predictions, "r.")
    ax.plot([min_val, max_val], [min_val, max_val], "k-", lw=0.3)
    plt.xlabel("True Values")
    plt.ylabel("ML4Chem Predictions")

    if scores:
        rmse = np.sqrt(mean_squared_error(true, predictions))
        mae = mean_absolute_error(true, predictions)
        correlation = r2_score(true, predictions)
        plt.text(
            min_val,
            max_val,
            "R-squared = {:.2f} \n"
            "RMSE = {:.2f}\n"
            "MAE = {:.2f}\n".format(correlation, rmse, mae),
        )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, **kwargs)


def read_log(logfile, metric="loss", refresh=None):
    """Read the logfile

    Parameters
    ----------
    logfile : str
        Path to logfile.
    metric : str
        Metric to plot. Supported are loss and rmse.
    refresh : float
        Interval in seconds before refreshing log file plot.
    """

    if refresh is not None:
        # This means that there is no dynamic update of the plot
        # We create an interactive plot
        plt.ion()
        fig = plt.figure()
        axes = fig.add_subplot(111)
        # This is for autoscale
        axes.set_autoscale_on(True)
        axes.autoscale_view(True, True, True)
        axes.set_xlabel("Epochs")
        annotation = axes.text(0, 0, str(""))
        plt.show(block=False)

    metric = metric.lower()

    f = open(logfile, "r")

    check = "Epoch"
    start = False
    epochs = []
    loss = []
    rmse = []

    initiliazed = False
    while refresh is not None:
        for line in f.readlines():
            if check in line:
                start = True

            if start:
                try:
                    line = line.split()
                    epochs.append(int(line[0]))
                    loss.append(float(line[3]))
                    rmse.append(float(line[4]))
                except ValueError:
                    pass

        if initiliazed is False:
            if metric == "loss":
                (fig,) = plt.plot(epochs, loss, label="loss")

            elif metric == "rmse":
                (fig,) = plt.plot(epochs, rmse, label="rmse")

            else:
                (fig,) = plt.plot(epochs, loss, label="loss")
                (fig,) = plt.plot(epochs, rmse, label="rmse")
        else:
            if metric == "loss":
                fig.set_data(epochs, loss)

            elif metric == "rmse":
                fig.set_data(epochs, rmse)

            else:
                fig.set_data(epochs, loss)
                fig.set_data(epochs, rmse)

            # Updating annotation
            if metric == "loss":
                values = loss
            elif metric == "rmse":
                values = rmse

            reported = values[-1]
            x = int(epochs[-1] * 0.9)
            y = float(reported * 1.3)
            annotation.set_text("{:.5f}".format(reported))
            annotation.set_position((x, y))

        plt.legend(loc="upper left")
        axes.relim()
        axes.autoscale_view(True, True, True)

        # Draw the plot
        plt.draw()
        plt.pause(refresh)
        initiliazed = True
    else:
        for line in f.readlines():
            if check in line:
                start = True

            if start:
                try:
                    line = line.split()
                    epochs.append(int(line[0]))
                    loss.append(float(line[3]))
                    rmse.append(float(line[4]))
                except ValueError:
                    pass

        if metric == "loss":
            (fig,) = plt.plot(epochs, loss, label="loss")

        elif metric == "rmse":
            (fig,) = plt.plot(epochs, rmse, label="rmse")

        else:
            (fig,) = plt.plot(epochs, loss, label="loss")
            (fig,) = plt.plot(epochs, rmse, label="rmse")

        plt.show(block=True)


def plot_atomic_features(latent_space, method="PCA", dimensions=2, backend="seaborn"):
    """Plot high dimensional atomic feature vectors

    This function can take a feature space dictionary, or a database file
    and plot the atomic features using PCA or t-SNE.

    $ ml4chem --plot tsne --file path.db

    Parameters
    ----------
    latent_space : dict or str
        Dictionary of atomic features of path to database file.
    method : str, optional
        Dimensionality reduction method to employed, by default "PCA".
        Supported are: "PCA" and "TSNE".
    dimensions : int, optional
        Number of dimensions to reduce the high dimensional atomic feature
        vectors, by default 2.
    backend : str, optional
        Select the backend to plot features. Supported are "plotly" and
        "seaborn", by default "plotly".
    """
    method = method.lower()
    backend = backend.lower()
    dot_size = 4.0

    supported_methods = ["pca", "tsne"]

    if method not in supported_methods:
        raise NotImplementedError

    if backend == "seaborn":
        # This hack is needed because it seems plotly import overwrite everything.
        import matplotlib.pyplot as plt

    axis = ["x", "y", "z"]

    if dimensions > 3:
        raise NotImplementedError
    elif dimensions == 2:
        axis.pop(-1)

    if isinstance(latent_space, str):
        latent_space = load(latent_space)

    full_ls = []
    full_symbols = []

    # This conditional is needed if you are passing an atomic feature database.
    if b"feature_space" in latent_space.keys():
        latent_space = latent_space[b"feature_space"]

    for hash, feature_space in latent_space.items():
        for symbol, feature_vector in feature_space:
            try:
                symbol = symbol.decode("utf-8")
            except AttributeError:
                pass

            if isinstance(feature_vector, np.ndarray) is False:
                feature_vector = feature_vector.numpy()

            full_symbols.append(symbol)
            full_ls.append(feature_vector)

    if method == "pca":
        from sklearn.decomposition import PCA

        labels = {str(axis[i]): "PCA-{}".format(i + 1) for i in range(len(axis))}
        pca = PCA(n_components=dimensions)
        pca_result = pca.fit_transform(full_ls)

        to_pandas = []

        entry = []
        for i, element in enumerate(pca_result):
            entry = [full_symbols[i]]
            for d in range(dimensions):
                entry.append(element[d])
            to_pandas.append(entry)

        columns = ["Symbol"]
        args = {}

        for key in axis:
            columns.append(labels[key])
            args[key] = labels[key]

        df = pd.DataFrame(to_pandas, columns=columns)

        if dimensions == 3 and backend == "plotly":
            args["color"] = "Symbol"
            plt = px.scatter_3d(df, **args)
            plt.update_traces(marker=dict(size=dot_size))
        elif dimensions == 2 and backend == "plotly":
            args["color"] = "Symbol"
            plt = px.scatter(df, **args)
            plt.update_traces(marker=dict(size=dot_size))
        elif dimensions == 3 and backend == "seaborn":
            raise ("This backend is for 2D visualization")
        elif dimensions == 2 and backend == "seaborn":
            sns.scatterplot(**labels, data=df, hue="Symbol")

    elif method == "tsne":
        from sklearn import manifold

        labels = {str(axis[i]): "t-SNE-{}".format(i + 1) for i in range(len(axis))}

        tsne = manifold.TSNE(n_components=dimensions)

        tsne_result = tsne.fit_transform(full_ls)

        to_pandas = []

        entry = []
        for i, element in enumerate(tsne_result):
            entry = [full_symbols[i]]
            for d in range(dimensions):
                entry.append(element[d])
            to_pandas.append(entry)

        columns = ["Symbol"]
        args = {}

        for key in axis:
            columns.append(labels[key])
            args[key] = labels[key]

        df = pd.DataFrame(to_pandas, columns=columns)

        if dimensions == 3 and backend == "plotly":
            args["color"] = "Symbol"
            plt = px.scatter_3d(df, **args)
            plt.update_traces(marker=dict(size=dot_size))
        elif dimensions == 2 and backend == "plotly":
            args["color"] = "Symbol"
            plt = px.scatter(df, **args)
            plt.update_traces(marker=dict(size=dot_size))
        elif dimensions == 3 and backend == "seaborn":
            raise ("This backend is for 2D visualization")
        elif dimensions == 2 and backend == "seaborn":
            sns.scatterplot(**labels, data=df, hue="Symbol")

    try:
        plt.show()
    except:
        pass

    return plt
