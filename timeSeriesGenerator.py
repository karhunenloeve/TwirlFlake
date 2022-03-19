import os
import os.path
import numpy as np
import timeSeriesConfig as cfg
import random as rand
import timeSeriesEmbedding as tse

from os import path
from gtda.time_series import SingleTakensEmbedding
from gtda.diagrams import PersistenceImage
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering
from gtda.time_series import SingleTakensEmbedding
from gtda.diagrams import BettiCurve

from gtda.point_clouds import ConsistentRescaling
from PIL import Image


def chunkIt(seq, num):
    """
    **Chunks a list into a partition of specified size.**

    + param **seq**: sequence to be chunked, dtype `list`.
    + param **num**: number of chunks, dtype `int`.
    + return **out**: chunked list, dtype `list`.
    """
    out = []
    last = 0.0

    while last < len(seq):
        if int(last + num) > len(seq):
            break
        else:
            out.append(seq[int(last) : int(last + num)])
            last += num

    return out


def numpy_to_img(
    directory: str,
    target: str,
    filetype: str = ".npy",
    target_filetype: str = ".png",
    color_mode: str = "RGB",
):
    """
    **Converts a set of numpy arrays with the form (x,x,3) to RGB images.**

    Example:
    ```python
        numpy_to_img("./data","./images")
    ```

    + param **directory**: directory to be processed, dtype `str`.
    + param **target**: directory to create, dtype `str`.
    """
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if not os.path.exists(target + "/" + d):
                os.makedirs(target + "/" + d)

        for file in files:
            nparray = np.load(root + "/" + file, allow_pickle=True)[0].transpose()

            if np.count_nonzero(nparray) > 0:
                img = Image.fromarray(nparray, color_mode)
                file = root + "/" + file
                img.save(
                    file.replace(directory, target).replace(filetype, target_filetype)
                )
            else:
                continue


def create_timeseries_dataset_persistence_images(
    path_to_data: str,
    target: str = cfg.paths["data"],
    label_delimiter: str = "@",
    file_extension_marker: str = ".",
    embedding_dimension: int = 3,
    embedding_time_delay: int = 1,
    stride: int = 1,
    delimiter: str = ",",
    n_jobs: int = -1,
    n_bins: int = 100,
    column: int = 3,
    homology_dimensions: tuple = (0, 1, 2),
    delete_old: bool = True,
    window_size: int = 200,
    filtering_epsilon: float = 0.23,
):
    """
    **Creates a directory from a directory with time series `.csv` data for use with Keras.**

    We have based the folder on the following structure: In a folder File there are several subfolders. In each of these of these subfolders are `.csv` files. The subfolders themselves have no meaning for the classification or the the assignment of the names. The files are named as they will be labeled later. Optionally the filename can be can be extended, then the label should be placed after a `@` in the file names. The files are loaded and transformed into a `n`-dimensional Torus using Taken's embedding. We create artificial examples by starting from this embedding. The persistent homology and resulting persistent images are then generated for each example from a time series. A corresponding folder is created where the generated persistent image is stored. The file is numbered and
    stored in the folder with its name.

    + param **path_to_data**: path to the directory containing the time series files `.npy`, dType `str`.
    + param **target**: path to target directory containing time series files `.png`, dtype `str`.
    + param **label_delimiter**: delimiter for the labels, dtype `str`.
    + param **file_extension_marker**: user-defined marker for file extensions (one dot), dtype `str`.
    + param **embedding_dimension**: default embedding dimension for sliding window, dtype `int`.
    + param **embedding_time_delay**: given embedding time delay for sliding window, dtype `int`.
    + param **stride**: default step size for sliding window, dtype `int`.
    + param **delimiter**: delimiter within the `.csv` file containing the time series, dtype `str`.
    + param **n_jobs**: number of CPU cores to use, dtype `int`.
    + param **n_bins**: resolution for persistence images (n_bins,n_bins), dtype `int`.
    + param **column**: which column contains the desired time series, dtype `int`.
    + param **homology_dimensions**: dimension for which the rank of the homology vector spaces is calculated, dtype `tuple`.
    + param **delete_old**: whether to delete the old files or not, dtype `bool`.
    + param **window_size**: the size of the window for a training sample, dtype `int`.
    + param **filtering_epsilon**: epsilon to filter the persistence noise from the off-diagonal line, dtype `float`.
    """
    count = 0
    for dirpath, _, files in os.walk(path_to_data):
        for file in files:
            try:
                label_delimiter_index = file.index(label_delimiter)
            except ValueError:
                print("Label delimiter not found.")
                continue

            file_extension_index = file.index(file_extension_marker)
            label = file[label_delimiter_index + 1 : file_extension_index]
            new_dir_path = target + label

            try:
                f = os.path.abspath(os.path.join(dirpath, file))
                data = np.genfromtxt(f, delimiter=delimiter).transpose()[column]
            except ValueError:
                print("This file can not be processed due to wrong format.")
                continue

            embedder = SingleTakensEmbedding(
                parameters_type="fixed",
                n_jobs=n_jobs,
                time_delay=embedding_time_delay,
                dimension=embedding_dimension,
                stride=stride,
            )

            try:
                data_embedded = embedder.fit_transform(data)
            except ValueError:
                print("This file can not be processed as it is empty.")
                continue

            # Check if some values are NaN.
            # Further check if the file has been created already.
            if np.isnan(data_embedded).any():
                continue
            elif path.exists(
                new_dir_path + "/file" + str(count) + "_chunk" + str(0) + ".npy"
            ):
                print("This file has been skipped, as it already exists.")
                continue
            else:
                chunks = chunkIt(data_embedded, window_size)
                chunkcount = 0

                for i in range(0, len(chunks)):
                    persistence = VietorisRipsPersistence(
                        homology_dimensions=homology_dimensions, n_jobs=n_jobs
                    )

                    persistence_diagram = persistence.fit_transform(
                        chunks[i].reshape(1, *chunks[i].shape)
                    )
                    diagramScaler = Scaler()
                    persistence_diagram_scaled = diagramScaler.fit_transform(
                        persistence_diagram
                    )
                    diagramFiltering = Filtering(
                        epsilon=filtering_epsilon,
                        homology_dimensions=homology_dimensions,
                    )
                    persistence_diagram_scaled = diagramFiltering.fit_transform(
                        persistence_diagram_scaled
                    )

                    persistence_image = PersistenceImage(n_bins=n_bins, n_jobs=n_jobs)
                    persistence_picture = persistence_image.fit_transform(
                        persistence_diagram_scaled
                    )

                    if path.exists(new_dir_path):
                        pass
                    else:
                        os.mkdir(new_dir_path)

                    np.save(
                        new_dir_path
                        + "/file"
                        + str(count)
                        + "_chunk"
                        + str(chunkcount),
                        persistence_picture,
                    )
                    chunkcount += 1
                count += 1


def create_timeseries_dataset_ts_betti_curves(
    path_to_data: str,
    target: str = cfg.paths["data"],
    sample_size: int = 200,
    minimum_ts_size: int = 2 * 10 ** 3,
    class_size: int = 10 ** 3,
    label_delimiter: str = "@",
    file_extension_marker: str = ".",
    delimiter: str = ",",
    n_jobs: int = -1,
    column: int = 3,
    homology_dimensions: tuple = (0, 1, 2),
    color_mode: str = "L",
    saveasnumpy: bool = False,
):
    """
    **Creates from a directory of time series `.csv` data a directory for use with Keras ImageDataGenerator.**

    This function generates from a dataset consisting of `.csv` files a pixel-oriented collection of
    of multivariate time series consisting of the original signal section and the persistent Betti curve of this section.
    Each line of the image corresponds to a time series, i.e. the first line corresponds to the original signal and the second
    line corresponds to the topological progression in the form of the Betti curve. These are grouped in folders named after their classes.
    The classes must be stored as values in a separate column for each entry within the `csv` file. Then a
    corresponding folder will be created, which is compatible with `ImageGenerator` of `Keras`.

    Example of usage:
    ```python
    create_timeseries_dataset_ts_betti_curves(cfg.paths["split_ordered"], cfg.paths["data"])
    ```

    + param **path_to_data**: path to directory containing time series `.npy` files, dtype `str`.
    + param **target**: Path to target directory containing time series files `.png`, dtype `str`.
    + param **sample_size**: size of a sample within the whole dataset, dtype `int`.
    + param **minimum_ts_size**: the minimum size of the time series to be considered as part of the data set, dtype `int`.
    + param **class_size**: the class size, dtype `int`.
    + param **label_delimiter**: the label delimiter within the `csv` file, dtype `str`.
    + param **file_extension_marker**: marker for the file extension, typically `.`, dtype `str`.
    + param **delimiter**: delimiter within the `.csv` file containing the time series, dtype `str`.
    + param **n_jobs**: number of CPU cores to use, dtype `int`.
    + param **column**: which column contains the desired time series, dtype `int`.
    + param **homology_dimensions**: dimension for which the rank of the homology vector spaces is calculated, dtype `tuple`.
    + param **color_mode**: `grayscale` or `rgb`, dtype `str`.
    + param **saveasnumpy**: whether these files should be saved as a numpy array or images, dtype `bool`.
    """
    for dirpath, _, files in os.walk(path_to_data):
        for file in files:
            try:
                label_delimiter_index = file.index(label_delimiter)
            except ValueError:
                print("Label delimiter not found.")
                continue

            file_extension_index = file.index(file_extension_marker)
            label = file[label_delimiter_index + 1 : file_extension_index]
            new_dir_path = target + label

            try:
                f = os.path.abspath(os.path.join(dirpath, file))
                data = np.genfromtxt(f, delimiter=delimiter).transpose()[column]
                if len(data) < minimum_ts_size:
                    continue
            except (ValueError, TypeError) as e:
                print("This file can not be processed due to wrong format. %s", e)
                continue

            i = 0
            tmp = []

            while i <= class_size:
                treshold = len(data) - sample_size
                dice = rand.randint(0, treshold) % (treshold)
                # If this sample has already been drawn, continue.
                if dice in tmp:
                    continue
                tmp.append(dice)
                i = i + 1
                sample = data[dice : sample_size + dice]

                # Check for messy data from the database.
                if np.isnan(sample).any():
                    print(
                        "This file does contain NAN values due to some formatting error."
                    )
                    continue
                elif path.exists(new_dir_path + "/sample_" + str(i) + ".npy"):
                    # Ignore files, that already exist.
                    print("This file has been skipped, as it already exists.")
                    continue

                embedder = SingleTakensEmbedding(
                    parameters_type="search", n_jobs=n_jobs
                )
                (
                    signal_embedded,
                    embedding_dimension,
                    embedding_time_delay,
                ) = tse.fit_embedder(sample, embedder)

                persistence = VietorisRipsPersistence(
                    homology_dimensions=homology_dimensions, n_jobs=n_jobs
                )
                persistence_diagram = persistence.fit_transform(
                    signal_embedded.reshape(1, *signal_embedded.shape)
                )
                betti_curve = BettiCurve(n_bins=sample_size, n_jobs=n_jobs)
                persistence_curve = betti_curve.fit_transform(persistence_diagram)

                betti_rank_0 = persistence_curve[0][0]
                betti_rank_1 = persistence_curve[0][1]
                multivariate_sample = np.array([sample, betti_rank_0, betti_rank_1])

                if not path.exists(new_dir_path):
                    os.mkdir(new_dir_path)

                if saveasnumpy:
                    np.save(
                        new_dir_path
                        + "/sample_"
                        + str(embedding_dimension)
                        + "-"
                        + str(embedding_time_delay)
                        + "-"
                        + str(i),
                        multivariate_sample,
                    )
                else:
                    img = Image.fromarray(multivariate_sample, color_mode)
                    file = (
                        new_dir_path
                        + "/sample_"
                        + str(embedding_dimension)
                        + "-"
                        + str(embedding_time_delay)
                        + "-"
                        + str(i)
                    )
                    img.save(file + ".png")
