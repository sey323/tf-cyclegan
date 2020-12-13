import tensorflow as tf
import common.layer as layer


def output_layer(
    input, output_shape, channel=3, filter_size=[2, 2], stride=2, reuse=False
):
    """
    Generatorの最後の層の出力から画像を復元する。

    Args:
        input (tensor):
            画像のtensor行列
        channel (int):
            出力する画像のカラーチャネル
    """
    with tf.compat.v1.variable_scope("image_reconstract", reuse=reuse) as scope:
        deconv_out = layer.deconv2d(
            input,
            stride=stride,
            filter_size=filter_size,
            output_shape=output_shape,
            output_dim=channel,
            batch_norm=False,
            name="Deconv_Output",
        )
        output = layer.tanh(deconv_out)
    return output
