import matplotlib.pyplot as plt


def parity(predictions, true, filename=None, **kwargs):
    """A parity plot function

    Parameters
    ----------
    predictions : list or numpy.array
        Model predictions in a list.
    true : list or numpy.array
        Targets or true values.
    filename : str
        A name to save the plot to a file. If filename is non exisntent, we
        call plt.show().

    Notes
    -----
    kargs accepts all valid keyword arguments for matplotlib.pyplot.savefig.
    """

    min_val = min(true)
    max_val = max(true)
    fig = plt.figure(figsize=(6., 6.))
    ax = fig.add_subplot(111)
    ax.plot(true, predictions, 'r.')
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', lw=0.3,)
    plt.xlabel('True Values')
    plt.ylabel('ML4Chem Predictions')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, **kwargs)
