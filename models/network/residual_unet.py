import logging
import tensorflow as tf
import common.layer as layer
import common.modules as modules
import common.util.imutil as imutil
from models.network.output_layer import output_layer


def residual_unet(
    input,
    output_shape,
    channel=3,
    filter_size=[3, 3],
    normalization=None,
    reuse=False,
    name="",
):
    """Residual Generator
    Resnetを用いたUnet

    Args:
        input (tensor):
            入力画像
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

        # conv2
        with tf.compat.v1.variable_scope("conv_layer_1") as scope:
            output = layer.conv2d(
                input=input,
                stride=2,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=64,
                name="Conv_1",
                batch_norm=True,
            )
            output = layer.leakyReLU(output)

        # conv2
        with tf.compat.v1.variable_scope("conv_layer_2") as scope:
            output = layer.conv2d(
                input=output,
                stride=2,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=128,
                name="Conv_2",
                batch_norm=True,
            )
            output = layer.leakyReLU(output)

        # conv3
        with tf.compat.v1.variable_scope("conv_layer_3") as scope:
            output = layer.conv2d(
                input=output,
                stride=2,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=256,
                name="Conv_3",
                batch_norm=True,
            )
            output = layer.leakyReLU(output)

        # Residual Block
        with tf.compat.v1.variable_scope("conv_layer_4") as scope:
            output = modules.ResidualBlock_preact(
                input=output,
                stride=1,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=256,
                normalization=normalization,
                name="Residual_4",
            )

        with tf.compat.v1.variable_scope("conv_layer_5") as scope:
            output = modules.ResidualBlock_preact(
                input=output,
                stride=1,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=256,
                normalization=normalization,
                name="Residual_5",
            )

        with tf.compat.v1.variable_scope("conv_layer_6") as scope:
            output = modules.ResidualBlock_preact(
                input=output,
                stride=1,
                filter_size=[
                    3,
                    3,
                ],
                output_dim=256,
                normalization=normalization,
                name="Residual_6",
            )

        # 画像の生成
        before_output = output
        layers = [64, 128]
        # 逆畳み込みによって画像を再構成する．
        for i, input_shape in enumerate(reversed(layers)):
            # 初期情報
            layer_no = len(layers) - i
            output_dim = layers[layer_no - 1]
            output_h, output_w = imutil.calcImageSize(
                output_shape[0], output_shape[1], stride=2, num=layer_no
            )
            logging.debug(
                "[OUTPUT]\t(batch_size, output_height:{0}, output_width:{1}, output_dim:{2})".format(
                    output_h, output_w, output_dim
                )
            )

            # deconv
            with tf.compat.v1.variable_scope(
                "deconv_layer{0}".format(layer_no)
            ) as scope:
                deconv = layer.deconv2d(
                    before_output,
                    stride=2,
                    filter_size=filter_size,
                    output_shape=[output_h, output_w],
                    output_dim=output_dim,
                    batch_norm=True,
                    name="Deconv_{}".format(layer_no),
                )
                before_output = layer.ReLU(deconv)

        # Outputで画像に復元
        output = output_layer(before_output, output_shape=output_shape, channel=channel)
    return output
