import logging
import tensorflow as tf
import common.tfv1wrap.layer as layer
import common.tfv1wrap.modules as modules
import common.util.imutil as imutil
from tfv1.gan.models.dcgan.network import output_layer


def mini_unet(
    input,
    layers,
    output_shape,
    channel=3,
    filter_size=[3, 3],
    normalization=None,
    reuse=False,
    name="",
):
    """Residual Generator
    Resnetを用いたUnet
    ネットワークのパラメータ数を大幅に減らした。

    Args:
        input (tensor):
            入力画像
        layers (Array):
            内部のレイヤの出力する次元数。
            この配列の数だけ自動に層が加算される。
        output_shape(Array):
            出力する画像の縦横のサイズ([w, h])
        channel (int):
            出力するカラーチャネル
        filter_size (Array):
            フィルターのサイズ
        normalization (layer):
            適応するnormalization手法
        reuse (Boolean):
            同じネットワークが呼び出された時に再定義し直すかどうか
        name (String):
            ネットワークの名前
    Returns:
        output (tensor):
            output_shapeに復元された画像
    """
    print("[NETWORK]\tRes-net Generator")
    with tf.compat.v1.variable_scope("Generator" + name, reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()

        # conv1
        with tf.compat.v1.variable_scope("conv_layer_1") as scope:
            conv_1 = layer.conv2d(
                input=input,
                stride=2,
                filter_size=filter_size,
                output_dim=64,
                name="Conv_1",
                batch_norm=False,
            )
            conv_1 = layer.leakyReLU(conv_1)

        # conv2
        with tf.compat.v1.variable_scope("conv_layer_2") as scope:
            conv_2 = layer.conv2d(
                input=conv_1,
                stride=2,
                filter_size=filter_size,
                output_dim=128,
                name="Conv_2",
                batch_norm=False,
            )
            conv_2 = layer.leakyReLU(conv_2)

        # Residual Block
        with tf.compat.v1.variable_scope("conv_layer_3") as scope:
            conv_3 = modules.ResidualBlock_preact(
                input=conv_2,
                stride=1,
                filter_size=filter_size,
                output_dim=256,
                normalization=normalization,
                name="Residual_3",
            )

        with tf.compat.v1.variable_scope("conv_layer_4") as scope:
            conv_4 = modules.ResidualBlock_preact(
                input=conv_3,
                stride=1,
                filter_size=filter_size,
                output_dim=512,
                normalization=normalization,
                name="Residual_4",
            )
            conv_4 = tf.concat([conv_4, conv_3], axis=3)

        with tf.compat.v1.variable_scope("conv_layer_5") as scope:
            conv_5 = modules.ResidualBlock_preact(
                input=conv_4,
                stride=1,
                filter_size=filter_size,
                output_dim=256,
                normalization=normalization,
                name="Residual_5",
            )
            conv_5 = tf.concat([conv_5, conv_2], axis=3)

        # deconv2
        with tf.compat.v1.variable_scope("deconv_layer1") as scope:
            deconv_1 = layer.deconv2d(
                conv_5,
                stride=2,
                filter_size=filter_size,
                output_shape=[conv_5.shape[1] * 2, conv_5.shape[2] * 2],
                output_dim=128,
                batch_norm=False,
                name="Deconv_1",
            )
            deconv_1 = layer.ReLU(deconv_1)
            deconv_1 = tf.concat([deconv_1, conv_1], axis=3)

        # deconv3
        with tf.compat.v1.variable_scope("deconv_layer2") as scope:
            deconv_2 = layer.deconv2d(
                deconv_1,
                stride=2,
                filter_size=filter_size,
                output_shape=[deconv_1.shape[1] * 2, deconv_1.shape[2] * 2],
                output_dim=64,
                batch_norm=False,
                name="Deconv_2",
            )
            deconv_2 = layer.ReLU(deconv_2)

        # Outputで画像に復元
        output = output_layer(
            deconv_2, output_shape=output_shape, channel=channel, stride=1
        )
    return output
