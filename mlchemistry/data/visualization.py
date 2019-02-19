import matplotlib.pyplot as plt


def parity(predictions, true):
    """A parity plot function"""

    min_val = min(true)
    max_val = max(true)
    fig = plt.figure(figsize=(6., 6.))
    ax = fig.add_subplot(111)
    ax.plot(true, predictions, 'r.')
    ax.plot([min_val, max_val], [min_val, max_val], 'k-', lw=0.3,)

    plt.show()
