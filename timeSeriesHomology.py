import numpy as np
import os
import timeSeriesConfig as cfg
import timeSeriesEmbedding as tse

from gtda.homology import VietorisRipsPersistence
from gtda.homology import WeakAlphaPersistence
from gtda.time_series import SingleTakensEmbedding
from gtda.diagrams import BettiCurve
from gtda.diagrams import HeatKernel
from gtda.diagrams import Silhouette
from pathlib import Path


def compute_persistence_representations(
    path: str,
    parameters_type: str = "fixed",
    filetype: str = ".csv",
    delimiter: str = ",",
    n_jobs: int = -1,
    embedding_dimension: int = 3,
    embedding_time_delay: int = 5,
    stride: int = 10,
    index: int = 3,
    enrico_betti: bool = True,
    enrico_silhouette: bool = True,
    enrico_heatkernel: bool = True,
    n_bins: int = 100,
    store: bool = True,
    truncate: int = 3000,
) -> np.ndarray:
    """
    **Procedure computes the persistent homology representations for some data.**

    This is a collection of representations from `giotto-tda`. The examined folder structure has two sublevels.
    We find each file in this duplicate folder structure and compute all desired persistence representations
    for a fixed hyperparameter setting. The hyperparameters must be estimated beforehand. Optionally
    store the persistence diagram, the silhouette, a persistence heat kernel and the persistent Betti curve together with the embedded signal.
    With the embedded signal. The embedding dimension and time delay are encoded in the filename `embedding dimension - time delay`.

    + param **path**: path to the destination directory, type `str`.
    + param **filetype**: type of file to process, type `str`.
    + param **delimiter**: delimiter for files such as `csv`, type `str`.
    + param **n_jobs**: number of processors to use, type `int`.
    + param **embedding_dimension**: dimension of toroidal embedding, type `int`.
    + param **embedding_time_delay**: window size or time delay for the embedding, type `int`.
    + param **stride**: shift or step size for the embedding, type `int`.
    + param **index**: index of the signal within the `csv` file, type `int`.
    + param **enrico_betti**: whether to calculate Betti curves or not, type `bool`.
    + param **enrico_silhouette**: whether solhouettes should be calculated or not, type `bool`.
    + param **enrico_heatkernel**: whether heatkernels should be calculated or not, type `bool`.
    + param **n_bins**: resolution for persistence representations, type `int`.
    + param **store**: whether the calculated data should be stored as `npy` files or not, type `bool`.
    """
    homology_dimensions = list(range(0, embedding_dimension - 1))

    for d, dirs, files in os.walk(path):
        for f in files:
            filepath = os.path.join(d, f)
            file, extension = os.path.splitext(filepath)

            try:
                signal = np.genfromtxt(
                    filepath, delimiter=delimiter, encoding="utf-8", dtype=float
                ).transpose()[index]
                signal = signal[~np.isnan(signal)]  # Removing NAN-values.

                # Truncate the signal if it is to computationally demanding.
                if len(signal) > truncate:
                    signal = signal[0:truncate]

                # Option for fixed parameters.
                if parameters_type == "fixed":
                    embedder = SingleTakensEmbedding(
                        parameters_type=parameters_type,
                        n_jobs=n_jobs,
                        time_delay=embedding_time_delay,
                        dimension=embedding_dimension,
                        stride=stride,
                    )
                    signal_embedded = embedder.fit_transform(signal)

                # Optimizer searches for best fitting parameters.
                elif parameters_type == "search":
                    embedder = SingleTakensEmbedding(
                        parameters_type=parameters_type, n_jobs=n_jobs
                    )
                    (
                        signal_embedded,
                        embedding_dimension,
                        embedding_time_delay,
                    ) = tse.fit_embedder(signal, embedder)

                persistence = VietorisRipsPersistence(
                    homology_dimensions=homology_dimensions, n_jobs=-1
                )

                print("Processed file is: " + file)
                persistence_path = Path(file + "_persistence_diagram" + ".npy")
                betti_curve_path = Path(file + "_betti_curve" + ".npy")
                silhouette_path = Path(file + "_silhouette" + ".npy")
                heat_kernel_path = Path(file + "_heat_kernel" + ".npy")
                embedding_path = Path(
                    file
                    + "_embedded_"
                    + str(embedding_dimension)
                    + "-"
                    + str(embedding_time_delay)
                    + ".npy"
                )

                if not persistence_path.is_file():
                    persistence_diagram = persistence.fit_transform(
                        signal_embedded.reshape(1, *signal_embedded.shape)
                    )
                    if store:
                        np.save(persistence_path, persistence_diagram)
                if not embedding_path.is_file() and store:
                    np.save(embedding_path, signal_embedded)
                if enrico_betti and not betti_curve_path.is_file():
                    betti_curve = BettiCurve(n_bins=n_bins, n_jobs=n_jobs)
                    persistence_curve = betti_curve.fit_transform(persistence_diagram)
                    if store:
                        np.save(betti_curve_path, persistence_curve)
                if enrico_silhouette and not silhouette_path.is_file():
                    silhouette = Silhouette(power=1.0, n_bins=n_bins, n_jobs=n_jobs)
                    persistence_silhouette = silhouette.fit_transform(
                        persistence_diagram
                    )
                    if store:
                        np.save(silhouette_path, persistence_silhouette)
                if enrico_heatkernel and not heat_kernel_path.is_file():
                    heat_kernel = HeatKernel(sigma=0.1, n_bins=n_bins, n_jobs=n_jobs)
                    persistence_heatmap = heat_kernel.fit_transform(persistence_diagram)
                    if store:
                        np.save(heat_kernel_path, persistence_heatmap)
            except (ValueError, UnicodeDecodeError) as e:
                continue
