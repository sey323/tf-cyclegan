import logging
import tensorflow as tf
import common.layer as layer
import common.util.imutil as imutil


def discriminator(input, layers, channel=3, filter_size=[3, 3], reuse=False, name=""):
    """
    Discriminator

    Args:
        input (Tensor):
            本物か偽物か識別したい画像のTensor配列
        channel (int):
            入力するカラーチャネル
        reuse (Boolean):
            同じネットワークが呼び出された時に再定義し直すかどうか
        name (String):
            ネットワークの名前
    """
    logging.info("\t[NETWROK]\tDeep Convolutional Discriminator")
    with tf.compat.v1.variable_scope("Discriminator" + name, reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()

        for i, output_shape in enumerate(layers, 1):
            if i == 1:  # 1層目の時だけ
                before_output = input
            # conv
            with tf.compat.v1.variable_scope("conv_layer{0}".format(i)) as scope:
                conv = layer.conv2d(
                    input=before_output,
                    stride=2,
                    filter_size=filter_size,
                    output_dim=output_shape,
                    batch_norm=True,
                    name="Conv_{}".format(i),
                )
                conv = layer.leakyReLU(conv)

            before_output = conv
    return before_output
