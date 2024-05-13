#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool1D,
    concatenate,
)


def mnase_Etienne(winsize=2001, **kwargs):
    """
    Builds a Deep neural network model from simple convolutions

    Arguments
    ---------
    (optional) winsize: the sequence length of the input

    Returns
    -------
    The uncompiled model

    """
    kernel_init = VarianceScaling()
    model = Sequential(
        [
            Conv1D(
                64,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_initializer=kernel_init,
                input_shape=(winsize, 4),
            ),
            MaxPool1D(2, padding="same"),
            BatchNormalization(),
            Dropout(0.2),
            Conv1D(
                16,
                kernel_size=8,
                padding="same",
                activation="relu",
                kernel_initializer=kernel_init,
            ),
            MaxPool1D(2, padding="same"),
            BatchNormalization(),
            Dropout(0.2),
            Conv1D(
                8,
                kernel_size=80,
                padding="same",
                activation="relu",
                kernel_initializer=kernel_init,
            ),
            MaxPool1D(2, padding="same"),
            BatchNormalization(),
            Flatten(),
            Dense(1, activation="relu"),
        ]
    )
    return model


def bassenji_Etienne(winsize=32768, **kwargs):
    """
    Builds a Deep neural network model with the bassenji architecture

    Arguments
    ---------
    (optional) winsize: the sequence length of the input

    Returns
    -------
    The uncompiled model

    """
    kernel_init = VarianceScaling()

    # build the CNN model
    input_layer = Input(shape=(winsize, 4))

    x = Conv1D(
        32,
        kernel_size=12,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
    )(input_layer)
    x = MaxPool1D(pool_size=8, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
    )(x)
    x = MaxPool1D(pool_size=4, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
    )(x)
    x = MaxPool1D(pool_size=4, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(
        16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        dilation_rate=2,
    )(x)
    x = BatchNormalization()(x)
    x1 = Dropout(0.2)(x)

    x = x1
    x = Conv1D(
        16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        dilation_rate=4,
    )(x)
    x = BatchNormalization()(x)
    x2 = Dropout(0.2)(x)

    x = concatenate([x1, x2], axis=2)
    x = Conv1D(
        16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        dilation_rate=8,
    )(x)
    x = BatchNormalization()(x)
    x3 = Dropout(0.2)(x)

    x = concatenate([x1, x2, x3], axis=2)
    x = Conv1D(
        16,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        dilation_rate=16,
    )(x)
    x = BatchNormalization()(x)
    x4 = Dropout(0.2)(x)

    x = concatenate([x1, x2, x3, x4], axis=2)
    x = Conv1D(
        1,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
    )(x)
    model = tf.keras.Model(input_layer, x)
    return model
