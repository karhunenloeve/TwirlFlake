import timeSeriesConfig as cfg
import timeSeriesModels as mod
import tensorflow.keras.backend as K
import tensorflow as tf
import datetime
import os.path
import pandas as pd

from tensorflow.keras import layers, optimizers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.metrics import AUC
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.layers import (
    Dense,
    Input,
    Lambda,
    Reshape,
    Add,
    Conv1D,
    BatchNormalization,
    Flatten,
)

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def recall(y_true, y_pred):
    """
    **Compute recall of the classification task.**

    Calculates the proportion of true positives within the classified samples divided by the true positives and false negatives.

    + param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
    + param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """
    **Compute precision of the classification task.**

    Calculates the proportion of true positives within the classified samples divided by the true positives and false positives.

    + param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
    + param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    """
    **Compute f1-score of the classification task.**

    Calculates the f1 score, which is two times the precision times the recall divided by the precision plus the recall. A smoothing constant is used to avoid division by zero in the denominator.

    + param **y_true**: data tensor of labels with true values, dtype `tf.tensor`.
    + param **y_pred**: data tensor of labels with predicted values, dtype `tf.tensor`.
    """
    prec = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 2 * ((prec * re) / (prec + re + K.epsilon()))


def check_files(folder_path: str):
    """
    **This function checks the compatibility of the images with pillow.**

    We read the files with pillow and check for errors. If this function traverses the whole directory without stopping, the dataset will work fine with the given version of pillow and the `ImageDataGenerator` class of `Keras`.

    + param **folder_path**: path to the dataset consisting of images, dtype `str`.
    """
    import os
    from PIL import Image

    extensions = []
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            print("** Path: {}  **".format(file_path), end="\r", flush=True)
            im = Image.open(file_path)
            rgb_im = im.convert("RGB")
            if filee.split(".")[1] not in extensions:
                extensions.append(filee.split(".")[1])


def train_neuralnet_images(
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
):
    """
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
    """
    datagen = ImageDataGenerator(validation_split=validation_split)

    train_generator = datagen.flow_from_directory(
        directory,
        target_size=image_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset="training",
        interpolation=interpolation,
    )

    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=image_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset="validation",
        interpolation=interpolation,
    )

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    # Open a strategy scope and create/restore the model.
    # Strategy scope for computations on multiple GPUs.
    with strategy.scope():
        classes_number = max([v for k, v in validation_generator.class_indices.items()])
        shape = (image_size[0], image_size[1], 1)
        model = mod.create_three_nets(
            shape=shape, classes_number=classes_number, image_size=image_size
        )
        model.summary()

        model.compile(
            loss=SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.AUTO),
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy", precision, recall, f1, AUC()],
        )

        model_plateau_callback = ReduceLROnPlateau(
            monitor="loss",
            factor=reduction_factor,
            patience=reduction_patience,
            min_lr=min_lr,
        )
        model_earlystoping_callback = EarlyStopping(
            monitor="loss", patience=stopping_patience, verbose=verbose
        )

        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path + ".hdf5",
            save_weights_only=False,
            save_best_only=True,
            monitor="loss",
            mode="auto",
            save_freq=10 ** 4,
            verbose=1,
        )

        if os.path.exists(checkpoint_path + ".hdf5"):
            print("Loaded model: " + checkpoint_path + ".hdf5.")
            model.load_weights(checkpoint_path + ".hdf5", by_name=True)
        history_filename = "history.csv"
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=nb_epochs,
            callbacks=[
                model_plateau_callback,
                model_earlystoping_callback,
                model_checkpoint,
            ],
        )

        history_df = pd.DataFrame(history.history)
        with open(history_filename, mode="w") as f:
            history_df.to_csv(f)


train_neuralnet_images(
    "./data",
    "./model_weights/C"
    + str(cfg.lstmmodel["differential"])
    + "_CNNCuDNNLSTM_Betticurves_"
    + str(cfg.cnnmodel["filters"])
    + "_"
    + str(cfg.lstmmodel["units"])
    + "_"
    + str(cfg.cnnmodel["layers_x"])
    + "_"
    + str(cfg.lstmmodel["layers"])
    + "layers",
    "./model_weights/C"
    + str(cfg.lstmmodel["differential"])
    + "_CNNCuDNNLSTM_Betticurves_"
    + str(cfg.cnnmodel["filters"])
    + "_"
    + str(cfg.lstmmodel["units"])
    + "_"
    + str(cfg.cnnmodel["layers_x"])
    + "_"
    + str(cfg.lstmmodel["layers"])
    + "layers",
)
