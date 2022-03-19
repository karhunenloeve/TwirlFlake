import timeSeriesConfig as cfg

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2, L1L2
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import (
    Dense,
    Input,
    Lambda,
    Reshape,
    Add,
    Multiply,
    Conv1D,
    BatchNormalization,
    LayerNormalization,
    Flatten,
    Dropout,
    MaxPooling1D,
    AveragePooling1D,
    TimeDistributed,
    LSTM,
)


def convolution_layer(
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
):
    """
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
    """
    if batchnorm:
        return BatchNormalization()(
            Conv1D(
                filters,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                name="conv_" + name + "_" + str(i),
                kernel_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            )(x)
        )
    else:
        return Conv1D(
            filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name="conv_" + name + "_" + str(i),
            kernel_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
        )(x)


def lstm_layer(
    x,
    name: str,
    i: int,
    units: int = cfg.lstmmodel["units"],
    return_state: bool = cfg.lstmmodel["return_state"],
    go_backwards: bool = cfg.lstmmodel["go_backwards"],
    stateful: bool = cfg.lstmmodel["stateful"],
    l1_regu: float = cfg.lstmmodel["l1"],
    l2_regu: float = cfg.lstmmodel["l2"],
):
    """
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
    """
    if i == cfg.lstmmodel["layers"]:
        return CuDNNLSTM(
            units,
            return_sequences=False,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            kernel_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            recurrent_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            bias_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            activity_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
        )(x)
    else:
        return CuDNNLSTM(
            units,
            return_sequences=True,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            kernel_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            recurrent_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            bias_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
            activity_regularizer=L1L2(l1=l1_regu, l2=l2_regu),
        )(x)


def create_lstm(
    x,
    name: str,
    layers: int,
    differential: int,
    max_pool: list,
    avg_pool: list,
    dropout_rate: float = cfg.lstmmodel["dropout_rate"],
    pooling_size: int = cfg.lstmmodel["pooling_size"],
):
    """
    **This function returns one of the subnetworks of the remaining CuDNNLSTM.**

    This function generates the core neural network according to the theory of `Ck`-differentiable neural networks.
    It applies residual connections according to the `k` degree of differentiability and adds dropout and pooling
    layers at the desired location in the architecture. Some of the specifications can be passed as
    argument to be passed to perform hyperparameter optimization.

    + param **x**: data passed by iterator to the Keras layer, `tf.tensor`.
    + param **name**: name of the layer, `str`.
    + param **layers**: number of hidden layers, `int`.
    + param **differential**: number of residual connections, `int`.
    + param **dropout_rate**: dropout probability, `float`.
    + param **pooling_size**: size for a pooling operation, `int`.
    + return **x**: processed data through some layers, dtype `tf.tensor`.
    """
    neuralNetwork = {}

    for i in range(1, layers + 1):
        if i == 1:
            neuralNetwork["lstm_" + name + "_" + str(i)] = lstm_layer(x, name, i)
        else:
            if differential == 0:
                neuralNetwork["lstm_" + name + "_" + str(i)] = lstm_layer(
                    neuralNetwork["lstm_" + name + "_" + str(i - 1)], name, i
                )
            elif differential == 1:
                neuralNetwork["lstm_" + name + "_" + str(i)] = Add()(
                    [
                        neuralNetwork["lstm_" + name + "_" + str(i - 1)],
                        lstm_layer(
                            neuralNetwork["lstm_" + name + "_" + str(i - 1)], name, i
                        ),
                    ]
                )
            if i in max_pool:
                neuralNetwork["lstm_" + name + "_" + str(i)] = MaxPooling1D(
                    pooling_size, padding=cfg.lstmmodel["padding"]
                )(neuralNetwork["lstm_" + name + "_" + str(i)])
            if i in avg_pool:
                neuralNetwork["lstm_" + name + "_" + str(i)] = AveragePooling1D(
                    pooling_size, padding=cfg.lstmmodel["padding"]
                )(neuralNetwork["lstm_" + name + "_" + str(i)])

    return neuralNetwork["lstm_" + name + "_" + str(layers)]


def create_subnet(
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
):
    """
    **This function returns one of the subnets of the rest of the CNN.**

    This function generates the core neural network according to the theory of `Ck`-differentiable neural
    networks. It applies residual connections according to the `k` degree of differentiability and adds
    dropout and pooling layers at the desired location in the architecture. Some of the specifications
    can be passed as argument to be passed to perform hyperparameter optimization. Currently it is just
    implemented for `C1`-architectures.

    + param **x**: data passed by iterator to the Keras layer, `tf.tensor`.
    + param **name**: name of the layer, `str`.
    + param **layers**: number of hidden layers, `int`.
    + param **differential**: number of residual connections, `int`.
    + param **dropouts**: list of layer indexes to apply dropouts to, `int`.
    + param **dropout_rate**: dropout probability, `float`.
    + param **pooling_size**: size for a pooling operation, `int`.
    + param **batchnorm**: whether to apply batchnorm or not, type `bool`.
    + return **x**: processed data through some layers, dtype `tf.tensor`.
    """
    neuralNetwork = {}

    for i in range(1, layers + 1):
        if i == 1:
            neuralNetwork["conv_" + name + "_" + str(i)] = convolution_layer(
                x, name, i, batchnorm=batchnorm
            )
        else:
            if differential == 0:
                neuralNetwork["conv_" + name + "_" + str(i)] = convolution_layer(
                    neuralNetwork["conv_" + name + "_" + str(i - 1)],
                    name,
                    i,
                    batchnorm=batchnorm,
                )
            elif differential == 1:
                neuralNetwork["conv_" + name + "_" + str(i)] = Add()(
                    [
                        neuralNetwork["conv_" + name + "_" + str(i - 1)],
                        convolution_layer(
                            neuralNetwork["conv_" + name + "_" + str(i - 1)],
                            name,
                            i,
                            batchnorm=batchnorm,
                        ),
                    ]
                )
            if i in dropouts:
                neuralNetwork["conv_" + name + "_" + str(i)] = Dropout(dropout_rate)(
                    neuralNetwork["conv_" + name + "_" + str(i)]
                )
            if i in max_pool:
                neuralNetwork["conv_" + name + "_" + str(i)] = MaxPooling1D(
                    pooling_size, padding=cfg.cnnmodel["padding"]
                )(neuralNetwork["conv_" + name + "_" + str(i)])
            if i in avg_pool:
                neuralNetwork["conv_" + name + "_" + str(i)] = AveragePooling1D(
                    pooling_size, padding=cfg.cnnmodel["padding"]
                )(neuralNetwork["conv_" + name + "_" + str(i)])

    return neuralNetwork["conv_" + name + "_" + str(layers)]


def create_three_nets(shape, classes_number, image_size) -> callable:
    """
    **The CNN architecture for classifying the lane sensor data.**

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
    """
    inputs = Input(shape=shape)
    x = Reshape((image_size[0], 1))(Lambda(lambda x: x[:, :, 0])(inputs))
    y = Reshape((image_size[0], 1))(Lambda(lambda x: x[:, :, 1])(inputs))
    z = Reshape((image_size[0], 1))(Lambda(lambda x: x[:, :, 2])(inputs))

    x = create_subnet(
        x,
        name=cfg.cnnmodel["name_x"],
        layers=cfg.cnnmodel["layers_x"],
        differential=cfg.cnnmodel["differential_x"],
        dropouts=cfg.cnnmodel["dropouts_x"],
        max_pool=cfg.cnnmodel["max_pool_x"],
        avg_pool=cfg.cnnmodel["avg_pool_x"],
        batchnorm=True,
    )
    y = create_subnet(
        y,
        name=cfg.cnnmodel["name_y"],
        layers=cfg.cnnmodel["layers_y"],
        differential=cfg.cnnmodel["differential_y"],
        dropouts=cfg.cnnmodel["dropouts_y"],
        max_pool=cfg.cnnmodel["max_pool_y"],
        avg_pool=cfg.cnnmodel["avg_pool_y"],
        batchnorm=False,
    )
    z = create_subnet(
        z,
        name=cfg.cnnmodel["name_z"],
        layers=cfg.cnnmodel["layers_z"],
        differential=cfg.cnnmodel["differential_z"],
        dropouts=cfg.cnnmodel["dropouts_z"],
        max_pool=cfg.cnnmodel["max_pool_z"],
        avg_pool=cfg.cnnmodel["avg_pool_z"],
        batchnorm=False,
    )

    o = Add()([x, y, z])
    o = create_lstm(
        o,
        name=cfg.lstmmodel["name_a"],
        layers=cfg.lstmmodel["layers"],
        differential=cfg.lstmmodel["differential"],
        max_pool=cfg.lstmmodel["max_pool"],
        avg_pool=cfg.lstmmodel["avg_pool"],
    )
    o = Flatten()(o)

    output = Dense(classes_number + 1, activation="softmax", name="DENSE_o")(o)

    return Model(inputs=inputs, outputs=output)
