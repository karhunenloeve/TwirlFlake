import typing
import os
import functools
import glob
import operator
import timeSeriesConfig as cfg
import ntpath
import numpy as np
import matplotlib
import matplotlib.cm as cm
import plotly.graph_objs as go
import chart_studio.plotly as py
import timeSeriesHelper as hp
import shutil

from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from plotly.tools import FigureFactory as FF
from functools import reduce
from tempfile import TemporaryFile
from gtda.plotting import plot_point_cloud
from gtda.time_series import SingleTakensEmbedding

matplotlib.use("WebAgg")


def list_files_in_dir(directory: str) -> list:
    """
    **Lists all files inside a directory.**

    Simple function that lists all files inside a directory as `str`.

    + param **str**: an absolute path, type `str`.
    + return **directory**: list of files, type `list`.
    """
    return os.listdir(directory)


def read_csv_data(path: str, delimiter: str = ",") -> np.ndarray:
    """
    **Reads `.csv` files into an `np.ndarray`**.

    Convert all columns of a `.csv` file into an `np.ndarray`.

    + param **path**: an absolute path, type `str`.
    + param **delimiter: the delimiter used within the `.csv` files, type `str`.
    + return **data**: data, type `np.ndarray`.
    """
    return np.genfromtxt(path, delimiter=delimiter, encoding="utf-8", dtype=float)


def fit_embedder(y: np.ndarray, embedder: callable, verbose: bool = True) -> tuple:
    """
    **Fits a Takens embedding and displays optimal search parameters.

    Determines the optimal parameters for the searched embedding according to the theory of toroidal embeddings resulting from the discrete Fourier transform.

    + param **y**: embedding array, type `np.ndarray`.
    + param **embedder**: slidingWindow embedding, `callable`.
    + param **verbose**: output result numbers, type `bool`.
    + return **(y_embedded.shape, embedder.dimension_, embedder.time_delay_)**: embedding, embedding dimension and delay, `tuple`.
    """
    try:
        y_embedded = embedder.fit_transform(y.reshape(-1, 1))
    except ValueError:
        y_embedded = embedder.fit_transform(y)
    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )

    return (y_embedded, embedder.dimension_, embedder.time_delay_)


def get_single_signal(index: int, file: int, plot: bool = False) -> np.ndarray:
    """
    **Gets a column directly from the `.csv` file.**

    Extracts a column from a `.csv` file according to the index and returns it as an
    `np.darray` for further processing.

    + param **index**: index of the signal column within the file, type `int`.
    + param **file**: index of the desired file within the folder structure, type `int`.
    + param **plot**: plot the signal via a web interface, type `bool`.
    + return **(signal, path)**: the desired signal and file path, type `np.array`.
    """
    paths = list_files_in_dir(cfg.paths["files"])
    signal = read_csv_data(cfg.paths["files"] + "/" + paths[file]).transpose()[index]

    if not isinstance(signal.flat[0], np.floating):
        raise ValueError(
            "Ooops! That was not a valid column within your sheet. Provide numerical values."
        )

    signal = signal[~np.isnan(signal)]  # Removing NAN-values.

    if plot:
        pyplot.plot(signal)
        pyplot.show()

    return (signal, cfg.paths["files"] + "/" + paths[file])


def get_sliding_window_embedding(
    index: int,
    file: int,
    width: int = 2,
    stride: int = 3,
    plot: bool = False,
    L: float = 1,
    B: float = 0,
    plot_signal: bool = False,
) -> np.ndarray:
    """
    **Sliding window embedding in a commutative Lie group**.

    This is an embedding which provably yields a commutative Lie group as an embedding space
    which is a smooth manifold with group structure. It is a connected manifold, so it has suitable
    properties to infer the dimension of homology groups. It is intuitive and can be used to
    It is intuitive and can be used to detect periodicities since it has direct connections to the theory of Fourier sequences.

    + param **index**: index of the signal column within the file, type `int`.
    + param **file**: index of the desired file within the folder structure, type `int`.
    + param **width**: determines the embedding dimension with `width+1`, type `int`.
    + param **stride**: Step size of the sliding window, type `int`.
    + param **plot**: represents the embedding of the signal in a web interface, type `bool`.
    + param **plot_signal**: plot of the original signal in a web interface, type `bool`.
    + return **signal_windows.transpose()**: embedding for signal, type `np.ndarray`.
    """
    try:
        signal, path = get_single_signal(index=index, file=file, plot=plot_signal)
        signal = np.fft.fft(signal)
        real_embedding = np.column_stack((signal.real, signal.imag))
        windows = SlidingWindow(width=width, stride=stride)
        signal_windows = windows.fit_transform(real_embedding)

        return signal_windows.transpose()
    except ValueError:
        print(
            "Ooops! That was not a valid column within your sheet. Provide numerical values."
        )


def get_periodic_embedding(
    index: int = 3,
    file: int = 1,
    plot_signal: bool = False,
    parameters_type: str = "fixed",
    max_time_delay: int = 3,
    max_embedding_dimension: int = 11,
    stride: int = 2,
    plot: bool = False,
    store: bool = False,
    fourier_transformed: bool = False,
):
    """
    **Adapts a single-tailed embedder and displays optimal search parameters.**

    This function uses a search algorithm to obtain optimal parameters for a time series embedding.
    The search can be neglected if the `parameters_type` parameter is selected as `fixed`. The time delay
    and the embedding dimension are determined by the algorithm. Optionally, the embedded
    time series signal as a `.np` file by setting the `store` parameter to `True`.

    + param **index**: column within the `.csv` file, type `int`.
    + param **file**: index of the file in the directory, type `int`.
    + param **plot_signal**: plot the raw signal, type `bool`.
    + param **parameters_type**: either `search` for optimal parameters or take them `fixed`, type `str`.
    + param **max_time_delay**: .maximum time delay, type `int`.
    + param **max_embedding_dimension**: . maximum embedding dimension, type `int`.
    + param **stride**: .maximum window displacement, type `int`.
    + param **plot**: plot the 3D embedding, type `bool`.
    + param **store**: store the 3D embedding, type `bool`.
    + param **fourier_transformed**: uses the Fourier transform, type `bool`.
    + return **embedded_signal**: the 3D embedding of the time series, type `np.ndarray`.
    """
    if fourier_transformed:
        timeseries, path = get_single_signal(index=index, file=file, plot=plot_signal)
        signal = np.fft.fft(timeseries)
        real_embedding = np.column_stack((signal.real, signal.imag)).flatten()
    else:
        real_embedding, path = get_single_signal(
            index=index, file=file, plot=plot_signal
        )

    embedder_periodic = SingleTakensEmbedding(
        parameters_type=parameters_type,
        time_delay=max_time_delay,
        dimension=max_embedding_dimension,
        stride=stride,
        n_jobs=-1,
    )
    embedded_signal, optimal_dimension, optimal_time_delay = fit_embedder(
        real_embedding, embedder_periodic
    )

    if plot:
        pca = PCA(n_components=3)
        y_periodic_embedded_pca = pca.fit_transform(embedded_signal)

        def split_nearest(arr):
            """
            **This function generates an approximate plot by periods of the Fourier transform.**

            The function checks whether all elements from the sorted distance to the barycenter of the embedding
            are really the closest points to this barycenter. It finds all indices corresponding to the nearest points.
            Points correspond, namely the periods that are close to the barycenter. This is then used for a split to find the
            Embedding.

            + param **arr**: periodic time series data, `np.ndarray`.
            + return **splitted_list**: list of `np.ndarray`s splitted by centroid, type `list`.
            """
            # TODO: Use barycenter instead of mean.
            newList = np.linalg.norm(
                y_periodic_embedded_pca, np.mean(y_periodic_embedded_pca), axis=1
            )
            sortedindices = np.argsort(newList).astype("float")

            for i in range(0, len(sortedindices) - 1):
                if any(j > sortedindices[i + 1] for j in sortedindices[0:i]):
                    sortedindices[i + 1] = np.nan

            indices = sortedindices[~np.isnan(sortedindices)].astype("int")
            return np.split(arr, indices, axis=1)

        # TODO: Color plot according to the computed split.
        split = split_nearest(y_periodic_embedded_pca)
        fig = plot_point_cloud(y_periodic_embedded_pca)

        for i in range(0, split):
            ax.plot(
                x[i : i + 2], y[i : i + 2], z[i : i + 2], color=plt.cm.jet(255 * i / N)
            )
            plt.show()
    if store:
        filename = ntpath.basename(path[0 : len(path) - 4])
        with open(cfg.paths["target"] + "/" + filename + ".npy", "wb") as f:
            np.save(f, embedded_signal)
    return embedded_signal


def count_dimension_distribution(
    path: str = cfg.paths["split"] + "**/*",
    recurr: bool = True,
    dimensions_dict: dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
    delays_dict: dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
    keyword: str = "_embedded_",
    position_dimension: int = -7,
    position_delay: int = -5,
):
    """
    **Determine the dimension used for embedding from the filenames and count them.**

    We encode the embedding dimension and the time delay in the filename of the processed
    Files representing the time series. We need to pass the determined values for dimension and
    Time Delay per signal through our algorithm. This function returns a tuple of dictionaries with
    these counts.

    + param **path**: path to the dictionary, type `str`.
    + param **recurr**: whether or not to recur to the directory, type `bool`.
    + param **dimensions_dict**: dictionary with counts for dimensions, type `dict`.
    + param **delays_dict**: dictionary with counts for delays, type `dict`.
    + param **keyword**: keyword for encoding the embedding in the filename, type `str`.
    + param **position_dimension**: position of dimension encoding, type `int`.
    + param **position_delay**: position of delay encoding, type `int`.
    + return **(dimensions, delays)**: tuple of dictionaries with values for dimensions and delays, type `tuple`.
    """
    files = glob.glob(path, recursive=recurr)
    dimensions = dimensions_dict
    delays = delays_dict

    for file in files:
        if keyword in file:
            dimension = file[position_dimension]
            delay = file[position_delay]
            dimensions[str(dimension)] += 1
            delays[str(delay)] += 1

    return (dimensions, delays)
