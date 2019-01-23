import matplotlib.pyplot as plt

def parity(predictions, true):
    """A parity plot function
    """

    min_val = min(true)
    max_val = max(true)

    plt.plot(true, predictions,'r.')
    plt.plot(true, true, 'k-')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.show()
