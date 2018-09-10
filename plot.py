import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker


def confusion_matrix(gold, pred, out='image/confusion.png'):

    all_categories = sorted(list(set(gold)))
    n_categories = len(all_categories)
    c2i = dict((c, i) for i, c in enumerate(all_categories))

    confusion = np.zeros((n_categories, n_categories))

    for p, g in zip(gold, pred):
        confusion[c2i[g]][c2i[p]] += 1

    # Normalize by dividing every row by its sum
    confusion = confusion / confusion.sum(1)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # cax = ax.matshow(confusion, vmin=0, vmax=1)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(out)
