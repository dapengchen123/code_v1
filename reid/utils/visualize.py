from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np


def plot_kernels_numpy(numpyarray, inputc=None, outputc=None, save=True):
    #sample image
    """
    
    :param numpyarray: tensor kernel in pytorch framwork, which has been transformed to numpy 
    :param inputc:   the input channel
    :param outputc:  the output channel
    :param save:  whether to save the image 
    :return: 
    """
    if inputc == None:
        inputc = range(numpyarray.shape[1])
        inputlen = len(inputc)
    else:
        inputc = range(inputc)
        inputlen = 1
    if outputc == None:
        outputc = range(numpyarray.shape[0])
        outputlen = len(outputc)
    else:
        outputc = range(outputc)
        outputlen = 1

    num = inputlen*outputlen

    numh = int(np.floor(np.sqrt(num)))
    numw = int(np.floor(num/numh))
    fig, axes = plt.subplots(numh, numw)
    # use global min/max to ensure all weights are shown on the same scale
    vmin, vmax = numpyarray.min(),  numpyarray.max()
    index = 0
    axs = axes.ravel()
    for i in inputc:
        for j in outputc:
            ax =axs[index]
            index = index + 1
            ax.matshow(numpyarray[j, i, :, :], cmap= plt.cm.gray, vmin= .5*vmin, vmax = .5*vmax)
            ax.set_xticks(())
            ax.set_yticks(())

    plt.show()
    if save:
        name = 'weight_matrix_{}_{}'.format(inputc, outputc)
        plt.savefig(name)


