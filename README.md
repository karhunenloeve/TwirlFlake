# Homological Time Series Analysis of Sensor Signals from Power Plants
[![License](https://img.shields.io/:license-mit-blue.svg)](https://badges.mit-license.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

In this paper, we use topological data analysis techniques to construct a suitable neural network classifier for the task of learning sensor signals of entire power plants according to their reference designation system. We use representations of persistence diagrams to derive necessary preprocessing steps and visualize the large amounts of data. We derive architectures with deep one-dimensional convolutional layers combined with stacked long short-term memories as residual networks suitable for processing the persistence features. We combine three separate sub-networks, obtaining as input the time series itself and a representation of the persistent homology for the zeroth and first dimension. We give a mathematical derivation for most of the used hyper-parameters. For validation, numerical experiments were performed with sensor data from four power plants of the same construction type.

**Keywords:** Power Plants · Time Series Analysis · Signal processing · Geometric embeddings · Persistent homology · Topological data analysis.

+ [This is the link to the arxiv article.](https://arxiv.org/abs/2106.02493)
+ [This is the link to the presentation hold at ML4ITS within the ECMLPKDD.](https://karhunenloeve.github.io/TwirlFlake/beamer.pdf)
+ [This is the link to the transcript of the talk given at ML4ITS.](https://karhunenloeve.github.io/TwirlFlake/transcript)

# Citation
```
@inproceedings{ML4ITS/MelodiaL21,
  author    = {Luciano Melodia and
               Richard Lenz},
  editor    = {Michael Kamp and
               Irena Koprinska and
               Adrien Bibal and
               Tassadit Bouadi and
               Beno{\^{\i}}t Fr{\'{e}}nay and
               Luis Gal{\'{a}}rraga and
               Jos{\'{e}} Oramas and
               Linara Adilova and
               Yamuna Krishnamurthy and
               Bo Kang and
               Christine Largeron and
               Jefrey Lijffijt and
               Tiphaine Viard and
               Pascal Welke and
               Massimiliano Ruocco and
               Erlend Aune and
               Claudio Gallicchio and
               Gregor Schiele and
               Franz Pernkopf and
               Michaela Blott and
               Holger Fr{\"{o}}ning and
               G{\"{u}}nther Schindler and
               Riccardo Guidotti and
               Anna Monreale and
               Salvatore Rinzivillo and
               Przemyslaw Biecek and
               Eirini Ntoutsi and
               Mykola Pechenizkiy and
               Bodo Rosenhahn and
               Christopher L. Buckley and
               Daniela Cialfi and
               Pablo Lanillos and
               Maxwell Ramstead and
               Tim Verbelen and
               Pedro M. Ferreira and
               Giuseppina Andresini and
               Donato Malerba and
               Ib{\'{e}}ria Medeiros and
               Philippe Fournier{-}Viger and
               M. Saqib Nawaz and
               Sebasti{\'{a}}n Ventura and
               Meng Sun and
               Min Zhou and
               Valerio Bitetta and
               Ilaria Bordino and
               Andrea Ferretti and
               Francesco Gullo and
               Giovanni Ponti and
               Lorenzo Severini and
               Rita P. Ribeiro and
               Jo{\~{a}}o Gama and
               Ricard Gavald{\`{a}} and
               Lee A. D. Cooper and
               Naghmeh Ghazaleh and
               Jonas Richiardi and
               Damian Roqueiro and
               Diego Saldana Miranda and
               Konstantinos Sechidis and
               Guilherme Gra{\c{c}}a},
  title     = {Homological Time Series Analysis of Sensor Signals from Power Plants},
  booktitle = {Machine Learning and Principles and Practice of Knowledge Discovery in Databases. ECML PKDD 2021},
  series    = {Communications in Computer and Information Science},
  volume    = {1524},
  pages     = {283--299},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-93736-2_22},
  doi       = {10.1007/978-3-030-93736-2_22},
}
```

# Contents
1. [Time Series Helper `timeSeriesHelper.py`](#timeSeriesHelper)
    - [zip_to_csv](#zip_to_csv)
    - [zip_to_npy](#zip_to_npy)
    - [sql_to_csv](#sql_to_csv)
    - [sql_to_npy](#sql_to_npy)
    - [csv_to_sql](#csv_to_sql)
    - [csv_to_npy](#csv_to_npy)
    - [npy_to_sql](#npy_to_sql)
    - [gen_GAF](#gen_GAF)
    - [gen_GAF_exec](#gen_GAF_exec)
    - [exit](#exit)
    - [switchoption](#switchoption)
    - [checkpath](#checkpath)
    - [split_csv_file](#split_csv_file)
    - [create_datasets](#create_datasets)
2. [Time Series Converter `timeSeriesConverter.py`](#timeSeriesConverter)
  	- [persistence_giotto_to_matplotlib](#persistence_giotto_to_matplotlib)
3. [Time Series Generator `timeSeriesGenerator.py`](#timeSeriesGenerator)
    - [chunkIt](#chunkIt)
    - [numpy_to_img](#numpy_to_img)
    - [create_timeseries_dataset_persistence_images](#create_timeseries_dataset_persistence_images)
    - [create_timeseries_dataset_ts_betti_curves](#create_timeseries_dataset_ts_betti_curves)
4. [Time Series Embedding `timeSeriesEmbedding.py`](#timeSeriesEmbedding)
    - [list_files_in_dir](#list_files_in_dir)
    - [read_csv_data](#read_csv_data)
    - [fit_embedder](#fit_embedder)
    - [get_single_signal](#get_single_signal)
    - [get_sliding_window_embedding](#get_sliding_window_embedding)
    - [get_periodic_embedding](#get_periodic_embedding)
    - [count_dimension_distribution](#count_dimension_distribution)
5. [Time Series Homology `timeSeriesHomology.py`](#timeSeriesHomology)
    - [compute_persistence_representations](#compute_persistence_representations)
6. [Time Series Visualisation `timeSeriesVisualisation.py`](#timeSeriesVisualisation)
    - [export_cmap_to_cpt](#export_cmap_to_cpt)
    - [plot_embedding3D](#plot_embedding3D)
    - [mean_heatkernel2D](#mean_heatkernel2D)
    - [massive_surface_plot3D](#massive_surface_plot3D)
    - [massive_line_plot3D](#massive_line_plot3D)
7. [Time Series Models `timeSeriesModels.py`](#timeSeriesModels)
    - [convolution_layer](#convolution_layer)
    - [lstm_layer](#lstm_layer)
    - [create_lstm](#create_lstm)
    - [create_subnet](#create_subnet)
    - [create_three_nets](#create_three_nets)
8. [Time Series Classifier `timeSeriesClassifier.py`](#timeSeriesClassifier)
    - [recall](#recall)
    - [precision](#precision)
    - [f1](#f1)
    - [check_files](#check_files)
    - [train_neuralnet_images](#train_neuralnet_images)


## timeSeriesHelper
### zip_to_csv
```python
zip_to_csv(path: str)
```
**Converts a packed `sql` file to a `csv` file.**

This function unpacks the specified `zip` file at its location and calls the specified `sql_to_csv`
function on the `sql` file, with the same name as the zip file.

+ param **path**: as an absolute path to the `zip` file with the `sql` in it, type `str`.

### zip_to_npy
```python
zip_to_npy(path: str)
```
**Converts a packed `sql` file to an npy file.**

This function unpacks the specified zip file at its location and calls the specified sql_to_npy
function on the `sql` file, with the same name as the zip file.

+ param **path**: as an absolute path to the zip file containing `sql`, type `str`.

### sql_to_csv
```python
sql_to_csv(path: str, delimiter: str = "\n")
```
**Converts a set of `INSERT` statements to `csv` format.**

Extracts the data from a set of `INSERT` statements stored in a `sql` file, this function
converts the data into a `csv` file, where each non `INSERT` line is stored in a separate pickle file
file, and the data of the `INSERT` statements is stored line by line, with the specified delimiter
at the end of each line.

+ param **path**: as absolute path to the `sql` file, type `str`.
+ param **delimiter**: as delimiter at the end of each line, type `str`.

### sql_to_npy
```python
sql_to_npy(path: str, delimiter: str = ",")
```
**Converts a set of `INSERT` statements into a numpy array.**

Similar to the csv function, this function also stores unused data in a pickle file and creates
a brand new file with the extracted data, this time in npy format, but this time the
the delimiter must be the delimiter used in the `sql` file, plus an additional
missing_values string used to represent missing data.

+ param **path**: as the absolute path to the `sql` file, type `str`.
+ param **delimiter**: as the string used in the `sql` file to separate the data, type `str`.
+ param **missing_values**: the string used for missing data, type `str`.

### csv_to_sql
```python
csv_to_sql(path: str, delimiter: str = "\n")
```
**Converts an `csv` file into a set of `INSERT` statements.**

This function converts each set of data separated by the specified separator character
of a `csv` file into an `INSERT` statement. It also inserts data
stored in a pickle file which has the same name as the `csv` file,
as a comment at the beginning, so as not to interfere with functionality.

+ param **path**: as absolute path to the `csv` file, type `str`.
+ param **delimiter**: as string to recognize the different records, type `str`.

### csv_to_npy
```python
csv_to_npy(path: str, delimiter: str = ",")
```
**Converts a `csv` file to a Numpy array representation.**

This function converts a `csv` file into a 2-dimensional Numpy representation,
where each record separated by the specified delimiter is interpreted as a new line.

+ param **path**: as absolute path to the `csv` file, type `str`.
+ param **delimiter**: the string used to determine the rows of the numpy array, type `str`.
+ param **missing_values**: as the string used to represent missing data, type `str`.

### npy_to_sql
```python
npy_to_sql(path: str)
```
**Converts an npy file into a series of `INSERT` statements.**

This function is the reverse of sql_to_npy and if you use it in conjunction with
you will have the same file at the end as at the beginning.

+ param **path**: as absolute path to the npy file, type `str`.

### gen_GAF
```python
gen_GAF(path: str)
```
**Generate a gramian angle field with user input.**

This function receives user input via the console to generate
either a Gramian Angular Summation Field or a Gramian Angular Difference Field
from the data of a Numpy array using the function gen_GAF_exec.

+ param **path**: as absolute path to the npy file, type `str`.

### gen_GAF_exec
```python
gen_GAF_exec(
    data: list,
    sample_range: None or tuple = (-1, 1),
    method: str = "summation",
    null_value: str = "0",
)
```
**Generate a Gramian Angular Field.**

This is the actual function when it comes to generating a Gramian Angular Field
from the data of a Numpy array. This function takes several variables to determine
how the field should be scaled, what size it should have
and whether it is a sum or difference field.

+ param **data**: as the contents of an npy file , type `list`.
 param **size**: this is the size of the square output image, type int or `float`.
+ param **sample_range**: as the range to scale the data to, type None or `tuple`.
+ param **method**: as the type of field to scale to, type `sum` or `difference`, type `str`.
+ param **null_value**: as the number to use instead of NULL, type `str`.

### false_input
```python
false_input(path: str)
```
**Output error information and return to main.**

This function prints an error message to the console and invokes main.

### exit
```python
exit()
```
**Print message and exit program.**

This function prints a message on the console and terminates the program.

### switchoption
```python
switchoption(n: int, path: str)
```
**Call a function.**

This function calls one of the functions of this program according to the `n`.
and gives it the path as input.

+ param **n**: this number specifies which function to call, type `int`.
+ param **path**: this is the path to the file used for the function to be called, type `str`.

### checkpath
```python
checkpath(path: str)
```
**Check the path if it is relative.**

This function removes all quotes from a path and checks whether it is relative or absolute
it returns a *cleaned* path which is the absolute representation of the given path.

+ param **path**: the string to use as path, type `str`.
+ return **path**: the absolute path, type `str`.

### split_csv_file
```python
split_csv_file(
    path: str, header: bool = False, column: int = 1, delimiter: str = ";"
) -> bool
```
**Splits a `csv` file according to a column.**

Takes as input a `csv` file path. Groups the file according to a certain label and stores the
data into multiple files named after the label.

+ param **path**: path to the desired `csv` file, type `str`.
+ param **header**: whether to remove the header or not, type `bool`.
+ param **column**: index of the desired column to group by, type `bool`.
+ param **delimiter**: delimiter for the `csv` file, default `;`, type `str`.

### create_datasets
```python
create_datasets(root: str)
```
**Splits the various files into directories.**

This function recursively lists all files in a directory and divides them into the
hard-coded folder structure for persistence diagrams, heat kernels, the embeddings, the
persistent silhouette, and the Betti curves.

+ param: root, type `str`.

## timeSeriesConverter
### persistence_giotto_to_matplotlib
```python
persistence_giotto_to_matplotlib(
    diagram: np.ndarray, plot: bool = True, tikz: bool = True
) -> np.ndarray
```
**Help function to convert `giotto-tda` persistence diagram to one from `matplotlib`.**

`giotto-tda` uses plotly in a proprietary. The plotting function is part of the pipeline, not
accessible as an object. We use the coordinates returned by the function and create our own
own `matplotlib` plot. Currently the scales are lost.

+ param **plotlib**: persistence plot from giotto-tda, type `np.ndarray`.
+ param **plot**: whether to plot or not, type `bool`.
+ param **tikz**: whether to save the file as a tikz object or not, type `bool`.
+ return **persistence_diagram**: the original persistence_diagram, type `np.ndarray`.

## timeSeriesGenerator
### chunkIt
```python
chunkIt(seq, num)
```
**Chunks a list into a partition of specified size.**

+ param **seq**: sequence to be chunked, dtype `list`.
+ param **num**: number of chunks, dtype `int`.
+ return **out**: chunked list, dtype `list`.

### numpy_to_img
```python
def numpy_to_img(
    directory: str,
    target: str,
    filetype: str = ".npy",
    target_filetype: str = ".png",
    color_mode: str = "RGB",
)
```
**Converts a set of numpy arrays with the form (x,x,3) to RGB images.**

Example:
```python
    numpy_to_img("./data","./images")
```

+ param **directory**: directory to be processed, dtype `str`.
+ param **target**: directory to create, dtype `str`.

### create_timeseries_dataset_persistence_images
```python
create_timeseries_dataset_persistence_images(
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
)
```
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

### create_timeseries_dataset_ts_betti_curves
```python
create_timeseries_dataset_ts_betti_curves(
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
)
```
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

## timeSeriesEmbedding
### list_files_in_dir
```python
list_files_in_dir(directory: str)
```
**Lists all files inside a directory.**

Simple function that lists all files inside a directory as `str`.

+ param **str**: an absolute path, type `str`.
+ return **directory**: list of files, type `list`.

### read_csv_data
```python
read_csv_data(path: str, delimiter: str = ",") -> np.ndarray
```
**Reads `.csv` files into an `np.ndarray`**.

Convert all columns of a `.csv` file into an `np.ndarray`.

+ param **path**: an absolute path, type `str`.
+ param **delimiter: the delimiter used within the `.csv` files, type `str`.
+ return **data**: data, type `np.ndarray`.

### fit_embedder
```python
fit_embedder(y: np.ndarray, embedder: callable, verbose: bool = True) -> tuple
```
**Fits a Takens embedding and displays optimal search parameters.**

Determines the optimal parameters for the searched embedding according to the theory of toroidal embeddings resulting from the discrete Fourier transform.

+ param **y**: embedding array, type `np.ndarray`.
+ param **embedder**: slidingWindow embedding, `callable`.
+ param **verbose**: output result numbers, type `bool`.
+ return **(y_embedded.shape, embedder.dimension_, embedder.time_delay_)**: embedding, embedding dimension and delay, `tuple`.

### get_single_signal
```python
get_single_signal(index: int, file: int, plot: bool = False) -> np.ndarray
```
**Gets a column directly from the `.csv` file.**

Extracts a column from a `.csv` file according to the index and returns it as an
`np.darray` for further processing.

+ param **index**: index of the signal column within the file, type `int`.
+ param **file**: index of the desired file within the folder structure, type `int`.
+ param **plot**: plot the signal via a web interface, type `bool`.
+ return **(signal, path)**: the desired signal and file path, type `np.array`.

### get_sliding_window_embedding
```python
get_sliding_window_embedding(
    index: int,
    file: int,
    width: int = 2,
    stride: int = 3,
    plot: bool = False,
    L: float = 1,
    B: float = 0,
    plot_signal: bool = False,
) -> np.ndarray
```
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

### get_periodic_embedding
```python
get_periodic_embedding(
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
)
```
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

### count_dimension_distribution
```python
count_dimension_distribution(
    path: str = cfg.paths["split"] + "**/*",
    recurr: bool = True,
    dimensions_dict: dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
    delays_dict: dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
    keyword: str = "_embedded_",
    position_dimension: int = -7,
    position_delay: int = -5,
)
```
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

## timeSeriesHomology
### compute_persistence_representations
```python
compute_persistence_representations(
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
) -> np.ndarray
```
**Procedure computes the persistent homology representations for some data.**

This is a collection of representations from `giotto-tda`. The examined folder structure has two sublevels. We find each file in this duplicate folder structure and compute all desired persistence representations for a fixed hyperparameter setting. The hyperparameters must be estimated beforehand. Optionally store the persistence diagram, the silhouette, a persistence heat kernel and the persistent Betti curve together with the embedded signal. With the embedded signal. The embedding dimension and time delay are encoded in the filename `embedding dimension - time delay`.

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

## timeSeriesVisualisation
### export_cmap_to_cpt
```python
export_cmap_to_cpt(
    cmap,
    vmin: float = 0,
    vmax: float = 1,
    N: int = 255,
    filename: str = "test.cpt",
    **kwargs
)
```
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

### plot_embedding3D
```python
plot_embedding3D(path: str)
```
**Plots 3-1 embeddings iteratively within a directory.**

Plots a set of embeddings always with three dimensions and a time delay of 1.
This can be changed arbitrarily, according to the estimated parameters.

+ param **path**: Path to the directory containing `.npy` files, type `.npy`.

### mean_heatkernel2D
```python
mean_heatkernel2D(
    directory: str,
    homgroup: int,
    limit: int = 0,
    store: bool = False,
    plot: bool = False,
    filename: str = "figure",
    filetype: str = "svg",
    colormap: str = "Reds",
)
```
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

### massive_surface_plot3D
```python
massive_surface_plot3D(
    directory: str,
    homgroup: int,
    title: str = "Default",
    limit: int = 45000,
    store: bool = False,
    plotting: bool = False,
    filename: str = "figure",
    filetype: str = "svg",
    colormap: str = "Reds",
)
```
**Calculates a solid surface from curves.**

Calculates a surface from a directory full of `npy` files of curves (intended for Betti curves
from persistence diagrams). For the `x` and `y` coordinates, the corresponding indices of the Betti
curves themselves and the filtration index are selected. The value of the function is then visible
on the 'z' axis. Optionally, these can be displayed as a graph in the browser or also saved.

Example:
```python
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

### massive_line_plot3D
```python
massive_line_plot3D(
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
)
```
**Function creates a massive line chart from a directory of `.npy` files that contain the data.**

This function creates a line graph from a set of `.npy` files. The line graph will be three dimensional
and each line will be plotted along the `z` axis, while the other two axes will represent the plot
or time step. It is assumed that the `.npy` file stores a one-dimensional array. The method iterates over
to populate a directory of `.npy` files, each of which contains a one-dimensional time series.

Examples:
```python
massive_line_plot3D(
    directory="/home/lume/documents/siemens_samples/kraftwerk_betticurve/", homgroup=0
)
```

Example of multiple plots of Betti curves / persistence silhouettes:
```python
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

+ param **directory**: Directory of `.npy` files for line plots, type `str`.
+ param **homgroup**: Specify which homology group to plot, type `int`.
+ param **resolution**: Number of points added between min and max for interpolation, type `int`.
+ param **k**: B-spline degree, type `int`.
+ param **limit**: Limit of the number of files to be displayed, type `int`.
+ param **elev**: Angle of horizontal shift, type `float`.
+ param **azim**: Degree of counterclockwise rotation, type `int`.
+ param **plot**: whether to plot or not, type `bool`.
+ param **fignum**: figure number for multiple figures, type `int`.
+ return **True**: True if plot was successful, type `bool`.

## timeSeriesModels
### convolution_layer
```python
convolution_layer(
    x,
    name: str,
    i: int,
    filters: int = cfg.cnnmodel["filters"],
    kernel_size: int = cfg.cnnmodel["kernel_size"],
    activation: str = cfg.cnnmodel["activation"],
    padding: str = cfg.cnnmodel["padding"],
    l1_regu: float = cfg.cnnmodel["l1"],
    l2_regu: float = cfg.cnnmodel["l2"],
    batchnorm: bool = True,
)
```
**This is a convolution layer with stack normalization**.

Here we implement a convolutional layer with activation function, padding, optional filters
and stride parameters, and some regularizers on the Keras framework. This function is to
be used to recursively define a neural network architecture.

+ param **x**: data passed to a Keras layer via an iterator, `tf.tensor`.
+ param **name**: name of the layer, `str`.
+ param **i**: iterator to name the layers, `int`.
+ param **filters**: number of filters to be used within the convolution layer, `int`.
+ param **stride**: number of strides to be used within the convolution layer, `int`.
+ param **activation**: Activation function, `str`.
+ param **padding**: equal, causal or valid, `str`.
+ param **l1_regu**: regularization for l1 norm, type `float`.
+ param **l2_regu**: regularization for l2 norm, type `float`.
+ param **batchnorm**: whether to apply batchnorm or not, type `bool`.
+ return **layer**: Keras layer, either batchnorm or convolution, type `tf.keras.layers`.

### lstm_layer
```python
lstm_layer(
    x,
    name: str,
    i: int,
    units: int = cfg.lstmmodel["units"],
    return_state: bool = cfg.lstmmodel["return_state"],
    go_backwards: bool = cfg.lstmmodel["go_backwards"],
    stateful: bool = cfg.lstmmodel["stateful"],
    l1_regu: float = cfg.lstmmodel["l1"],
    l2_regu: float = cfg.lstmmodel["l2"],
)
```
**This is a CuDNNLSTM layer.**

We implement here a CuDNNLSTM layer with activation function, padding, optional filters
and stride parameters, and some regularizers on the Keras framework. This function is to
be used to recursively define a neural network architecture.

+ param **x**: data passed to a Keras layer via an iterator, `tf.tensor`.
+ param **name**: name of the layer, `str`.
+ param **i**: iterator to name the layers, `int`.
+ param **return_state**: whether to return the last state in addition to the output, `bool`.
+ param **go_backwards**: if `true`, the input sequence will be processed backwards and return the reverse sequence, `bool`.
+ param **stateful**: if `True`, the last state for each sample at index i in a batch as the initial state for the sample at index `i` in the following batch, `bool`.
+ param **l1_regu**: regularization for l1 norm, type `float`.
+ param **l2_regu**: regularization for l2 norm, type `float`.
+ return **CuDNNLSTM**: Keras CuDNNLSTM layer over data, `tf.keras.layers`.

### create_lstm
```python
create_lstm(
    x,
    name: str,
    layers: int,
    differential: int,
    max_pool: list,
    avg_pool: list,
    dropout_rate: float = cfg.lstmmodel["dropout_rate"],
    pooling_size: int = cfg.lstmmodel["pooling_size"],
)
```
**This function returns one of the subnetworks of the remaining CuDNNLSTM.**

This function generates the core neural network according to the theory of `Ck`-differentiable neural networks. It applies residual connections according to the `k` degree of differentiability and adds dropout and pooling layers at the desired location in the architecture. Some of the specifications can be passed as argument to be passed to perform hyperparameter optimization.

+ param **x**: data passed by iterator to the Keras layer, `tf.tensor`.
+ param **name**: name of the layer, `str`.
+ param **layers**: number of hidden layers, `int`.
+ param **differential**: number of residual connections, `int`.
+ param **dropout_rate**: dropout probability, `float`.
+ param **pooling_size**: size for a pooling operation, `int`.
+ return **x**: processed data through some layers, dtype `tf.tensor`.

### create_subnet
```python
create_subnet(
    x,
    name: str,
    layers: int,
    differential: int,
    dropouts: list,
    max_pool: list,
    avg_pool: list,
    dropout_rate: float = cfg.cnnmodel["dropout_rate"],
    pooling_size: int = cfg.cnnmodel["pooling_size"],
    batchnorm: bool = True,
)
```
**This function returns one of the subnets of the rest of the CNN.

This function generates the core neural network according to the theory of `Ck`-differentiable neural networks. It applies residual connections according to the `k` degree of differentiability and adds dropout and pooling layers at the desired location in the architecture. Some of the specifications can be passed as argument to be passed to perform hyperparameter optimization. Currently it is just implemented for `C1`-architectures.

+ param **x**: data passed by iterator to the Keras layer, `tf.tensor`.
+ param **name**: name of the layer, `str`.
+ param **layers**: number of hidden layers, `int`.
+ param **differential**: number of residual connections, `int`.
+ param **dropouts**: list of layer indexes to apply dropouts to, `int`.
+ param **dropout_rate**: dropout probability, `float`.
+ param **pooling_size**: size for a pooling operation, `int`.
+ param **batchnorm**: whether to apply batchnorm or not, type `bool`.
+ return **x**: processed data through some layers, dtype `tf.tensor`.

### create_c2_1DCNN_model
```python
create_three_nets(shape, classes_number, image_size) -> callable
```
**The CNN architecture for classifying the lane sensor data.

This function results in three subnets with the same architecture.
The purpose is to process the raw data within one of the subnets,
and in the other subnets to process the Betti curves of the
zeroth and first homology group, which are generated during the filtration of the
sample. The result is summed within a final layer and a corresponding
decision for a class is made by a usual vanilla density layer.

+ param **shape**: shape of the input data, dType `tuple`.
+ param **classes_number**: number of classes, dType `int`.
+ param **image_size**: size of the image to be processed, dtype `tuple`.
+ return **model**: keras model, dtype `tf.keras.Model`.

## timeSeriesClassifier
### recall
```python
recall(y_true, y_pred)
```
**Compute recall of the classification task.**

Calculates the proportion of true positives within the classified samples divided by the true positives and false negatives.

+ param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
+ param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.

### precision
```python
precision(y_true, y_pred)
```
**Compute precision of the classification task.**

Calculates the proportion of true positives within the classified samples divided by the true positives and false positives.

+ param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
+ param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.

### f1
```python
f1(y_true, y_pred)
```
**Compute f1-score of the classification task.**

Calculates the f1 score, which is two times the precision times the recall divided by the precision plus the recall. A smoothing constant is used to avoid division by zero in the denominator.

+ param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
+ param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.

### check_files
```python
check_files(folder_path: str)
```
**This function checks the compatibility of the images with pillow.**

We read the files with pillow and check for errors. If this function traverses the whole directory without stopping, the dataset will work fine with the given version of pillow and the `ImageDataGenerator` class of `Keras`.

+ param **folder_path**: path to the dataset consisting of images, dtype `str`.

### train_neuralnet_images
```python
train_neuralnet_images(
    directory: str,
    model_name: str,
    checkpoint_path=None,
    classes=None,
    seed=None,
    dtype=None,
    color_mode: str = "grayscale",
    class_mode: str = "categorical",
    interpolation: str = "bicubic",
    save_format: str = "tiff",
    image_size: tuple = cfg.cnnmodel["image_size"],
    batch_size: int = cfg.cnnmodel["batch_size"],
    validation_split: float = cfg.cnnmodel["validation_split"],
    follow_links: bool = False,
    save_to_dir: bool = False,
    save_prefix: bool = False,
    shuffle: bool = True,
    nb_epochs: int = cfg.cnnmodel["epochs"],
    learning_rate: float = cfg.cnnmodel["learning_rate"],
    reduction_factor: float = 0.2,
    reduction_patience: int = 10,
    min_lr: float = 1e-6,
    stopping_patience: int = 100,
    verbose: int = 1,
)
```
**Main training procedure for neural networks on image datasets.**

Conventional function based on the implementation of Keras in the `Tensorflow`-package for classifying image data. Works for images with `RGB`
and grayscale. The dataset of images to be classified must be organized as a folder structure, so that each class is in a folder.
ptionally test datasets can be created, then a two level hierarchical folder structure is needed. In the first level there are two folders,
once the training and the test dataset. In the second level then the folders with the examples for the individual classes.
Adam is used as the optimization algorithm. As metrics accuracy, precision, recall, f1-score and AUC are calculated.
All other parameters can be set optionally. The layers of the neural network are all labeled.

Example os usage:
```python
train_neuralnet_images(
    "./data",
    "./model_weights/C" + str(cfg.lstmmodel["differential"]) + "_CNNCuDNNLSTM_Betticurves_" + str(cfg.cnnmodel["filters"]) + "_" + str(cfg.lstmmodel["units"]) + "_" + str(cfg.cnnmodel["layers_x"]) + "_" + str(cfg.lstmmodel["layers"]) + "layers",
    "./model_weights/C" + str(cfg.lstmmodel["differential"]) + "_CNNCuDNNLSTM_Betticurves_" + str(cfg.cnnmodel["filters"]) + "_" + str(cfg.lstmmodel["units"]) + "_" + str(cfg.cnnmodel["layers_x"]) + "_" + str(cfg.lstmmodel["layers"]) + "layers",
)
```

+ param **directory**: directory of the dataset for the training, dType `str`.
+ param **model_name**: name of the model file to be saved, dtype `str`.
+ param **checkpoint_path**: path to the last checkpoint, dtype `str`.
+ param **classes**: optional list of classes (e.g. `['dogs', 'cats']`). Default is `none`. If not specified, the list of `classes` is automatically derived from the `y_col` mapped to the label indices will be alphanumeric). The dictionary containing the mapping of `class names` to `class indices` can be obtained from the `class_indices` attribute, dtype `list`.
+ param **seed**: optional random seed for shuffles and transformations, dtype `float`.
+ param **dtype**: dtype to be used for generated arrays, dtype `str`.
+ param **color_mode**: one of `grayscale`, `rgb`, `rgba`. Default: `rgb`. Whether the images are converted to have `1`, `3`, or `4` channels, dtype `str`.
+ param **class_mode**: one of `binary`, `categorical`, `input`, `multi_output`, `raw`, `sparse` or `None`, dtype `str`.
+ param **interpolation**: string, the interpolation method used when resizing images. The default is `bilinear`. Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`, `gaußian`, `mitchellcubic`, dtype `str`.
+ param **save_format**: one of `png`, `jpeg`, dtype `str`.
+ param **batch_size**: default is `32`, dtype `int`.
+ param **image_size**: the dimensions to scale all found images to, dtype `tuple`.
+ param **validation_split**: what percentage of the data to use for validation, dtype `float`.
+ param **follow_links**: whether to follow symlinks within class subdirectories, dtype `bool`.
+ param **shuffle**: whether to shuffle the data. Default: True. If set to False, the data will be sorted in alphanumeric order, dtype `bool`,
+ param **save_to_dir**: this optionally specifies a directory where to save the generated augmented images (useful for visualizing what you are doing), dtype `bool`.
+ param **save_prefix**: prefix to be used for the filenames of the saved images, dtype `bool`.
+ param **nb_epochs**: number of maximum number of epochs to train the neural network, dtype `int`.
+ param **learning_rate**: learning rate for the optimizer, dtype `float`.
+ param **reduction_factor**: reduction factor for plateaus, dtype `float`.
+ param **reduction_patience**: how many epochs to wait before reducing the learning rate, dtype `int`.
+ param **min_lr**: minimum learning rate, dtype `float`.
+ param **stopping_patience**: how many epochs to wait before stopping early, dtype `int`.
+ param **verbose**: 0 or 1, dtype `int`.
