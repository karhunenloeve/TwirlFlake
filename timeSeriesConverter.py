from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from itertools import product
from itertools import groupby
from operator import itemgetter
from sklearn import datasets
from scipy.stats import multivariate_normal as mvn
from ripser import Rips
from persim import PersImage
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import numpy as np
import plotly
import plotly.graph_objects as go
import os
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib

matplotlib.use("WebAgg")


def persistence_giotto_to_matplotlib(
    diagram: np.ndarray, plot: bool = True, tikz: bool = True
) -> np.ndarray:
    """
    **Help function to convert `giotto-tda` persistence diagram to one from `matplotlib`.**

    `giotto-tda` uses plotly in a proprietary. The plotting function is part of the pipeline, not
    accessible as an object. We use the coordinates returned by the function and create our own
    own `matplotlib` plot. Currently the scales are lost.

    + param **plotlib**: persistence plot from giotto-tda, type `np.ndarray`.
    + param **plot**: whether to plot or not, type `bool`.
    + param **tikz**: whether to save the file as a tikz object or not, type `bool`.
    + return **persistence_diagram**: the original persistence_diagram, type `np.ndarray`.
    """

    def add_identity(axes, *line_args, **line_kwargs):
        (identity,) = axes.plot([], [], *line_args, **line_kwargs)

        def callback(axes):
            low_x, high_x = axes.get_xlim()
            low_y, high_y = axes.get_ylim()
            low = max(low_x, low_y)
            high = min(high_x, high_y)
            identity.set_data([low, high], [low, high])

        callback(axes)
        axes.callbacks.connect("xlim_changed", callback)
        axes.callbacks.connect("ylim_changed", callback)
        return axes

    persistence_diagram = []
    colors = cm.rainbow(np.linspace(0, 1, diagram.shape[-1]))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    add_identity(ax1, color="black", ls="--")
    maximum_value = 0

    for group in range(diagram.shape[-1]):
        homology_group = np.delete(
            np.split(diagram[0], np.where(np.diff(diagram[0][:, 2]))[0])[group], 2, 1
        )
        persistence_diagram.append(homology_group)
        x = persistence_diagram[group][:, 0]
        y = persistence_diagram[group][:, 1]
        ax1.scatter(x, y, color=colors[group])

        try:
            if np.max(x) > maximum_value:
                maximum_value = np.max(x)
            if np.max(y) > maximum_value:
                maximum_value = np.max(y)
        except ValueError:
            pass

    if plot:
        ax1.set_xlim(0, maximum_value)
        ax1.set_ylim(0, maximum_value)
        plt.show()
    if tikz:
        tikzplotlib.save("fig1.tex")
    return persistence_diagram
