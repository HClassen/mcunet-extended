from typing import Any
from pathlib import Path

# Shut Tensorflow up.
from .shutup import silence
silence()

import tensorflow as tf

import keras

import numpy as np

from ..tinynas.mobilenet import LAST_CONV_CHANNELS, DROPOUT
from ..tinynas.searchspace import ConvOp, Model

from .layers import *


__all__ = [
    "convert_weights", "build_model", "to_tflite", "train_fast", "to_tflite_fast"
]


def _convert(entry: tuple[list[str], list[np.ndarray]]) -> list[np.ndarray]:
    # Only change axis for convolution and linear operations.
    # Convolution:
    #   Pytorch has (c_out, c_in, h, w)
    #   Tensorflow has (h, w, c_in, c_out)
    # Linear:
    #   Pytorch has (in, out)
    #   Tensorflow has (out, in)
    weight: np.ndarray = entry[1].numpy()

    match len(weight.shape):
        case 4:  # Conv2d
            swap = [-2, -1] if weight.shape[1] == 1 else [-1, -2]
            return np.moveaxis(weight, [0, 1], swap)
        case 2:  # Linear
            return weight.T
        case _:
            return weight


def convert_weights(state: dict[str, Any]) -> list[np.ndarray]:
    """
    Converts the weights of a Pytorch model to be suitable for a Tensorflow/Keras
    model. Expects the weights obtained by the ``state_dict`` method. The
    converted can then be loaded into a equivalent Tensorflow/Keras model with
    ``set_weights``.

    Args:
        state (dict):
            The weights of the Pytorch model.

    Returns:
        list[np.ndarray]:
            The converted weights.
    """
    filtered = filter(
        lambda entry: "num_batches_tracked" not in entry[0],
        state.items()
    )

    converted = map(_convert, filtered)

    return list(converted)


def _build_op(
    op: ConvOp,
    in_channels: int,
    out_channels: int,
    expansion_ratio: int,
    se_ratio: float,
    shortcut: bool,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    norm_layer: str,
    activation_layer: str
) -> KerasBaseLayer:
    match op:
        case ConvOp.CONV2D:
            return KerasConv2dLayer(
                in_channels,
                out_channels,
                se_ratio,
                shortcut,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.DWCONV2D:
            return KerasDWConv2dLayer(
                in_channels,
                out_channels,
                se_ratio,
                shortcut,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.BDWRCONV2D:
            return KerasBDWRConv2dLayer(
                in_channels,
                out_channels,
                expansion_ratio,
                se_ratio,
                shortcut,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case _:
            raise ValueError(f"unknown convolution operation: {op}")


def build_model(
    model: Model,
    classes: int,
    dropout: float = DROPOUT,
    norm_layer: str = "batchnorm",
    activation_layer: str = "relu6"
) -> keras.Sequential:
    """
    Build a `keras.Sequential` from a given `Model`.

    Args:
        model (Model):
            A model sampled from the MnasNet search space.
        classes (int).
            The amout of classes for the classifier to recognize.
        dropout (float):
            The percentage of dropout used in the classifier.
        norm_layer (str):
            The type of normalization to use for the norm layer (only supports
            'batchnorm').
        activation_layer (str):
            The type of activation function to use for the activation layer.

    Returns:
        keras.Sequential:
            The created Keras model.

    """
    in_channels = model.blocks[0].layers[0].in_channels
    first = [
        KerasConv2dNormActivation(
            in_channels,
            3,
            2,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )
    ]

    blocks: list[KerasBaseLayer] = []
    for block in model.blocks:
        for layer in block.layers:
            blocks.append(
                _build_op(
                    layer.op,
                    layer.in_channels,
                    layer.out_channels,
                    layer.expansion_ratio,
                    layer.se_ratio,
                    layer.shortcut,
                    layer.kernel_size,
                    layer.stride,
                    norm_layer,
                    activation_layer
                )
            )

    last: list[keras.Layer] = [
        KerasConv2dNormActivation(
            LAST_CONV_CHANNELS,
            1,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        ),
        keras.layers.GlobalAvgPool2D(
            data_format="channels_last", keepdims=True
        ),
        keras.layers.Flatten(data_format="channels_last"),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(classes)
    ]

    return keras.Sequential(first + blocks + last)


def get_train_datasets(
    path: str | Path,
    image_size: int,
    validation_split: float,
    batch_size: int,
    shuffle: bool
):
    if validation_split > 0.0:
        seed = 42
        subset = "both"
    else:
        seed = None
        subset = None

    return keras.utils.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        data_format="channels_last",
        verbose=False
    )


def _sample(ds: tf.data.Dataset):
    def sample():
        for image, _ in ds.take(10):
            yield [image]

    return sample


def to_tflite(net: keras.Sequential, ds: tf.data.Dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(net)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _sample(ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    return converter.convert()


def train_fast(net: keras.Sequential, resolution: int) -> None:
    net.compile(
        optimizer="sgd",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    net.fit(
        tf.ones([1, resolution, resolution, 3]),
        tf.ones([1]),
        epochs=1,
        verbose=0
    )


def to_tflite_fast(net: keras.Sequential, resolution: int) -> bytes:
    def sample():
        data = np.random.rand(1, resolution, resolution, 3)
        yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(net)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = sample
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    del converter

    return tflite


def clear_keras(free_memory: bool = True) -> None:
    keras.utils.clear_session(free_memory)

    silence()
