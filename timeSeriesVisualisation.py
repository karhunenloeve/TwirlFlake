# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:42:29 2020

@author: Luciano Melodia
"""

import os
import numpy as np
import random
import timeSeriesConfig as cfg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly

from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from scipy.interpolate import make_interp_spline, BSpline
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from cycler import cycler
from gtda.diagrams import HeatKernel
from gtda.plotting import plot_heatmap, plot_betti_surfaces, plot_betti_curves
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import to_tree

matplotlib.use("WebAgg")


def export_cmap_to_cpt(
    cmap,
    vmin: float = 0,
    vmax: float = 1,
    N: int = 255,
    filename: str = "test.cpt",
    **kwargs
):
    """
    **Exports a custom matplotlib color map to files.**

    Generates a color map for matplotlib at a desired normalized interval of choice. The map is not returned, but
    saved as a text file. The default name for this file is `test.cpt`. This file then contains the information
    for the color maps in matplotlib and can then be loaded.

    + param **cmap**: Name of the color map, type `str`.
    + param **vmin**: lower limit of normalization, type `float`.
    + param **vmax**: upper limit of normalization, type `float`.
    + param **N**: highest color value in `RGB`, type `int`.
    + param **filename**: name of the color map file, type `str`.
    + param **kwargs**: additional arguments like `B`, `F` or `N` for color definition, type `str`.
    """

    # Create string for upper, lower colors.
    b = np.array(kwargs.get("B", cmap(0.0)))
    f = np.array(kwargs.get("F", cmap(1.0)))
    na = np.array(kwargs.get("N", (0, 0, 0))).astype(float)
    ext = (np.c_[b[:3], f[:3], na[:3]].T * 255).astype(int)
    extstr = "B {:3d} {:3d} {:3d}\nF {:3d} {:3d} {:3d}\nN {:3d} {:3d} {:3d}"
    ex = extstr.format(*list(ext.flatten()))

    # Create colormap.
    cols = (cmap(np.linspace(0.0, 1.0, N))[:, :3] * 255).astype(int)
    vals = np.linspace(vmin, vmax, N)
    arr = np.c_[vals[:-1], cols[:-1], vals[1:], cols[1:]]

    # Save to file.
    fmt = "%e %3d %3d %3d %e %3d %3d %3d"
    np.savetxt(
        filename, arr, fmt=fmt, header="# COLOR_MODEL = RGB", footer=ex, comments=""
    )


def plot_embedding3D(path: str):
    """
    **Plots 3-1 embeddings iteratively within a directory.

    Plots a set of embeddings always with three dimensions and a time delay of 1.
    This can be changed arbitrarily, according to the estimated parameters.

    + param **path**: Path to the directory containing `.npy` files, type `.npy`.
    """
    files = os.listdir(path)

    for i in files:
        if "embedded_3-1" in i:
            data = np.load(path + i)
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            zdata = data.transpose()[0]
            xdata = data.transpose()[1]
            ydata = data.transpose()[2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="Blues")
            plt.show()


def mean_heatkernel2D(
    directory: str,
    homgroup: int,
    limit: int = 0,
    store: bool = False,
    plot: bool = False,
    filename: str = "figure",
    filetype: str = "svg",
    colormap: str = "Reds",
):
    """
    **Calculates a mean heat core over a large collection of files in a directory.**

    Calculates a mean heat core map from a folder full of `.npy` files with heat maps.
    This can optionally be saved or displayed as a plot in the browser.

    + param **directory**: directory of `.npy` files for line plots, type `str`.
    + param **homgroup**: specify which homology group to plot, type `int`.
    + param **limit**: limit the number of files to display, type `int`.
    + param **store**: whether to store the file or not, type `bool`.
    + param **filename**: name of the file to be saved, type `str`.
    + param **colormap**: plot color scales, type `str`.
    + return **fig**: figure object, type `plotly.graph_objects.Figure`.
    """
    files = []
    count = 0

    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            file = os.path.abspath(os.path.join(dirpath, f))
            try:
                data = np.load(file)
                files.append(data[0][homgroup])

                if limit > 0:
                    count += 1

                    if count == limit:
                        count = 0
                        break
            except IndexError:
                pass

    data = np.mean(files, axis=0)
    fig = plot_heatmap(data, colorscale=colormap)

    if plot:
        fig.show()
    if store:
        plotly.io.write_image(fig, filename, format=filetype)

    return fig


def massive_surface_plot3D(
    directory: str,
    homgroup: int,
    title: str = "Default",
    limit: int = 45000,
    store: bool = False,
    plotting: bool = False,
    filename: str = "figure",
    filetype: str = "svg",
    colormap: str = "Reds",
):
    """
    **Calculates a solid surface from curves.**

    Calculates a surface from a directory full of `npy` files of curves (intended for Betti curves
    from persistence diagrams). For the `x` and `y` coordinates, the corresponding indices of the Betti
    curves themselves and the filtration index are selected. The value of the function is then visible
    on the 'z' axis. Optionally, these can be displayed as a graph in the browser or also saved.

    Example:
    ``Python
    massive_surface_plot3D(
        "/home/lume/documents/siemens_samples/power_plant_silhouette/",
        homgroup=1,
        store=True,
        plotting=True,
        limit=1000
    )
    ```
    + param **directory**: directory of `.npy` files for line plots, type `str`.
    + param **homgroup**: determines which homology group to plot, type `int`.
    + param **limit**: limit on the number of files to display, type `int`.
    + param **plotting**: whether the file should be plotted or not, type `bool`.
    + param **store**: whether the file should be stored or not, type `bool`.
    + param **filename**: name of the file to be saved, type `str`.
    + param **colormap**: plot color scales, type `str`.
    + return **fig**: figure object, type `plotly.graph_objects.Figure`.
    """
    files = []
    count = 0

    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            file = os.path.abspath(os.path.join(dirpath, f))
            try:
                data = np.load(file)[0][homgroup]
                files.append(data)

                if limit > 0:
                    count += 1

                    if count == limit:
                        count = 0
                        break
            except IndexError:
                pass

    files = np.array(files)
    x = np.linspace(0, 1, files.shape[0])

    fig = go.Figure(
        data=[go.Surface(z=files, x=x, y=x, colorscale=colormap, reversescale=True)]
    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=0.25),
    )
    fig.update_layout(
        title=title,
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(l=65, r=50, b=90, t=90),
        scene_camera=camera,
    )

    if plotting:
        fig.show()
    if store:
        fig.write_image(filename + "." + filetype)

    return fig


def massive_line_plot3D(
    directory: str,
    homgroup: int,
    resolution: int = 300,
    k: int = 3,
    limit: int = 0,
    elev: float = 20,
    azim: int = 135,
    KKS: str = "LBB",
    fignum: int = 0,
    plot: bool = False,
):
    """
    **Function creates a massive line chart from a directory of `.npy` files that contain the data.**

    This function creates a line graph from a set of `.npy` files. The line graph will be three dimensional
    and each line will be plotted along the `z` axis, while the other two axes will represent the plot
    or time step. It is assumed that the `.npy` file stores a one-dimensional array. The method iterates over
    to populate a directory of `.npy` files, each of which contains a one-dimensional time series.

    Examples:
    `python
    massive_line_plot3D(
        directory="/home/lume/documents/siemens_samples/kraftwerk_betticurve/", homgroup=0
    )
    ```

    Example of multiple plots of Betti curves / persistence silhouettes:
    ``python
    number = 0
    for i in cfg.pptcat:
        massive_line_plot3D(
            directory="/home/lume/documents/siemens_kraftwerk_samples/kraftwerk_bettikurve/",
            homgroup=0,
            KKS=i,
            fignum=count,
        )
        count += 1
    plt.show()
    plt.close()
    ```

    + param **directory**: directory of `.npy` files for line plots, type `str`.
    + param **homgroup**: specify which homology group to plot, type `int`.
    + param **resolution**: number of points added between min and max for interpolation, type `int`.
    + param **k**: B-spline degree, type `int`.
    + param **limit**: limit of the number of files to be displayed, type `int`.
    + param **elev**: angle of horizontal shift, type `float`.
    + param **azim**: degree of counterclockwise rotation, type `int`.
    + param **plot**: whether to plot or not, type `bool`.
    + param **fignum**: figure number for multiple figures, type `int`.
    + return **True**: True if plot was successful, type `bool`.
    """
    files = []
    count = 0
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection="3d")

    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if KKS in f:
                file = os.path.abspath(os.path.join(dirpath, f))
                try:
                    data = np.load(file)[0][homgroup]
                    files.append(data)

                    if limit > 0:
                        count += 1

                        if count == limit:
                            count = 0
                            break
                except IndexError:
                    pass
            else:
                pass

    if not files:
        return False

    # files = sorted(files, key=sum)
    for f in files:
        x_array = np.full(len(f), count, dtype=int)
        y_array = np.arange(start=0, stop=len(f), step=1)
        evenly_spaced_interval = np.linspace(0, 1, len(files))

        if "betticurve" in directory:
            interval = evenly_spaced_interval[count]
            color1 = cm.RdGy(interval)
            color2 = cm.RdBu(interval)
        else:
            color1 = cm.RdGy(evenly_spaced_interval[count])
            color2 = cm.RdBu(evenly_spaced_interval[count])

        ax.plot(x_array, y_array, f, "x", markersize=2, color=color1)
        ax.plot(x_array, y_array, f, color=color2)
        ax.set_title(KKS + "_homgroup_" + str(homgroup), y=1.08)
        ax.view_init(elev=elev, azim=azim)
        ax.ticklabel_format(style="sci", useOffset=True)

        ax.w_xaxis.set_pane_color((1, 0.921, 0.803, 0.1))
        ax.w_yaxis.set_pane_color((1, 0.921, 0.803, 0.1))
        ax.w_zaxis.set_pane_color((1, 0.921, 0.803, 0.1))
        ax.w_xaxis.set_tick_params(
            labelsize=12,
            which="major",
            pad=-5,
            colors="black",
            labelrotation=23,
            direction="in",
        )
        ax.w_yaxis.set_tick_params(
            labelsize=12,
            which="major",
            pad=-5,
            colors="black",
            labelrotation=-23,
            direction="in",
        )
        ax.w_zaxis.set_tick_params(
            labelsize=12, which="major", pad=4, colors="black", direction="in"
        )

        formatter = ticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_powerlimits((-2, 3))
        ax.w_xaxis.set_major_formatter(formatter)

        # ax.w_yaxis.set_major_formatter(formatter)
        # ax.w_zaxis.set_major_formatter(formatter)
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_zticklabels([])

        count += 1

        if limit > 0:
            if count == limit:
                if plot:
                    plt.show()
                return True
    if plot:
        plt.show()
    return True
