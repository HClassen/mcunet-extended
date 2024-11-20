from abc import ABC, abstractmethod

import tensorflow as tf

import keras


__all__ = [
    "KerasConv2dNormActivation", "KerasDWConv2dNormActivation",
    "KerasSqueezeExcitation",
    "KerasBaseOp", "KerasConv2dOp", "KerasDWConv2dOp", "KerasMBConv2dOp"
]


class KerasConvNormActivation(keras.layers.Layer):
    conv2d: keras.layers.Layer
    norm_layer: keras.layers.BatchNormalization | None
    activation_layer: keras.layers.Activation | None

    def __init__(
        self,
        conv2d: keras.layers.Layer,
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        super().__init__()

        self.conv2d = conv2d

        if norm_layer == "batchnorm":
            self.norm_layer = keras.layers.BatchNormalization()
        else:
            raise NotImplementedError(f"only supports 'batchnorm'")

        self.activation_layer = keras.layers.Activation(activation_layer)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.conv2d(inputs)

        if self.norm_layer is not None:
                inputs = self.norm_layer(inputs)

        if self.activation_layer is not None:
            inputs = self.activation_layer(inputs)

        return inputs


class KerasConv2dNormActivation(KerasConvNormActivation):
    """
    A functional copy of torchvision.ops.Conv2dNormActivation.
    """
    conv2d: keras.layers.Conv2D

    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = (1, 1),
        padding: str = "same",  # needed for the same padding as the pytorch version
        groups: int = 1,
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        conv2d = keras.layers.Conv2D(
            filters,
            kernel_size,
            stride,
            padding,
            data_format="channels_last",
            groups=groups,
            use_bias=False
        )

        super().__init__(conv2d, norm_layer, activation_layer)


class KerasDWConv2dNormActivation(KerasConvNormActivation):
    """
    A functional copy of torchvision.ops.Conv2dNormActivation explicitly using
    the ``DepthwiseConv2D`` layer.
    """
    conv2d: keras.layers.DepthwiseConv2D

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = (1, 1),
        padding: str = "same",  # needed for the same padding as the pytorch version
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        conv2d = keras.layers.DepthwiseConv2D(
            kernel_size,
            stride,
            padding,
            depth_multiplier=1,
            data_format="channels_last",
            use_bias=False
        )

        super().__init__(conv2d, norm_layer, activation_layer)


class KerasSqueezeExcitation(keras.layers.Layer):
    """
    A functional copy of torchvision.ops.SqueezeExcitation.
    """
    avgpool: keras.layers.AvgPool2D
    fc1: keras.layers.Conv2D
    fc2: keras.layers.Conv2D
    activation: keras.layers.Activation
    scale_activation: keras.layers.Activation

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        activation: str = "relu6",
        scale_activation: str = "sigmoid"
    ) -> None:
        super().__init__()

        self.avgpool = keras.layers.GlobalAvgPool2D(
            data_format="channels_last", keepdims=True
        )
        self.fc1 = keras.layers.Conv2D(
            squeeze_channels, 1, data_format="channels_last"
        )
        self.fc2 = keras.layers.Conv2D(
            in_channels, 1, data_format="channels_last"
        )
        self.activation = keras.layers.Activation(activation)
        self.scale_activation = keras.layers.Activation(scale_activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        scale = self.avgpool(inputs)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * inputs


class KerasBaseOp(ABC, keras.layers.Layer):
    """
    A functional copy of mcunet.tinynas.mnasnet.layers.BaseOp.
    """
    use_residual: bool
    se: KerasSqueezeExcitation | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip: bool,
        stride: int | tuple[int, int],
    ) -> None:
        super().__init__()

        self.use_residual = (stride == 1 or stride == (1, 1)) and \
                            in_channels == out_channels and \
                            skip

    @abstractmethod
    def _call(self, input: tf.Tensor) -> tf.Tensor:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.use_residual:
            return inputs + self._call(inputs)
        else:
            return self._call(inputs)

    # def build(self, input_shape) -> None:
    #     if input_shape[0] is None:
    #         input_shape = (1, *input_shape[1:])

    #     self._call(tf.zeros(input_shape))

    #     super().build(input_shape)


def _add_se(
    in_channels: int,
    se_ratio: float,
    activation_layer: str
) -> KerasSqueezeExcitation | None:
    if se_ratio == 0:
        return None

    squeeze_channels = max(1, int(in_channels * se_ratio))
    return KerasSqueezeExcitation(in_channels, squeeze_channels, activation_layer)


class KerasConv2dOp(KerasBaseOp):
    """
    A functional copy of mcunet.tinynas.mnasnet.layers.Conv2dOp.
    """
    conv2d: KerasConv2dNormActivation

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        super().__init__(in_channels, out_channels, skip, stride)

        self.conv2d = KerasConv2dNormActivation(
            out_channels,
            kernel_size,
            stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        self.se = _add_se(out_channels, se_ratio, activation_layer)

    def _call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.conv2d(inputs)
        if self.se is not None:
            inputs = self.se(inputs)

        return inputs


class KerasDWConv2dOp(KerasBaseOp):
    """
    A functional copy of mcunet.tinynas.mnasnet.layersDW.Conv2dOp.
    """
    conv2d: KerasDWConv2dNormActivation
    depthwise: keras.layers.Conv2D
    norm: keras.layers.BatchNormalization

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        super().__init__(in_channels, out_channels, skip, stride)

        self.conv2d = KerasDWConv2dNormActivation(
            kernel_size,
            stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        self.se = _add_se(in_channels, se_ratio, activation_layer)

        self.depthwise = keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            data_format="channels_last",
            use_bias=False
        )

        if norm_layer == "batchnorm":
            self.norm = keras.layers.BatchNormalization()
        else:
            raise NotImplementedError(f"only supports 'batchnorm'")

    def _call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.conv2d(inputs)
        if self.se is not None:
            inputs = self.se(inputs)

        inputs = self.depthwise(inputs)
        return self.norm(inputs)


class KerasMBConv2dOp(KerasBaseOp):
    """
    A functional copy of mcunet.tinynas.mnasnet.layers.MBConv2dOp.
    """
    expand: KerasConv2dNormActivation | None
    conv2d: KerasDWConv2dNormActivation
    depthwise: keras.layers.Conv2D
    norm: keras.layers.BatchNormalization

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        skip: bool,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        norm_layer: str = "batchnorm",
        activation_layer: str = "relu6"
    ) -> None:
        super().__init__(in_channels, out_channels, skip, stride)

        hidden = in_channels * expansion_ratio
        if expansion_ratio > 1:
            self.expand = KerasConv2dNormActivation(
                hidden,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        else:
            self.expand = None

        self.conv2d = KerasDWConv2dNormActivation(
            kernel_size,
            stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )

        self.se = _add_se(hidden, se_ratio, activation_layer)

        self.depthwise = keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            data_format="channels_last",
            use_bias=False
        )

        if norm_layer == "batchnorm":
            self.norm = keras.layers.BatchNormalization()
        else:
            raise NotImplementedError(f"only supports 'batchnorm'")

    def _call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.expand is not None:
            inputs = self.expand(inputs)

        inputs = self.conv2d(inputs)
        if self.se is not None:
            inputs = self.se(inputs)

        inputs = self.depthwise(inputs)
        return self.norm(inputs)
