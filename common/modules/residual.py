import tensorflow as tf
import common.layer as layer


def ResidualBlock_preact(
    input,
    stride,
    filter_size,
    output_dim,
    padding="SAME",
    activation=layer.ReLU,
    normalization=None,
    drop_prob=1.0,
    projection=True,
    name="",
):
    """
    Residual Block:Best

    Args:
        input (tensor):
        stride (int):
            処理をするストライドのサイズ
        filter_size ([int,int]):
            畳み込みフィルタのサイズ
        output_dim (int):
            出力する特徴次元のサイズ
        padding:
            パディングの方法
        activation:
            Chose activation fnction in layer.py
        drop_prob:
            ドロップアウトの割合
        projection (boolean):
            if input_shape !=output_shape, concating input_img needs reduce dim.
            then use method projection or not
        normalization (layer):
            適応するnormalization手法
        name:
            モジュールに付与する固有の名前
    """
    block_name = "Residual_Block_Preact_" + str(name)
    with tf.compat.v1.variable_scope(block_name):
        if normalization:
            bn_1 = normalization(input)
        else:
            bn_1 = input
        act_1 = activation(bn_1)
        y_1 = layer.conv2d(
            act_1,
            stride,
            filter_size=[
                filter_size[0],
                filter_size[1],
            ],
            output_dim=output_dim,
            batch_norm=False,
            padding=padding,
            name="Conv_c1",
        )
        if normalization:
            bn_2 = normalization(y_1)
        else:
            bn_2 = y_1
        act_2 = activation(bn_2)
        act_2_drop = tf.nn.dropout(act_2, 1 - (drop_prob))
        y_2 = layer.conv2d(
            act_2_drop,
            1,
            filter_size=[
                filter_size[0],
                filter_size[1],
            ],
            output_dim=output_dim,
            batch_norm=False,
            padding=padding,
            name="Conv_c2",
        )
        input_dim = input.get_shape()[-1]
        if input_dim != output_dim:
            if projection:
                input = layer.conv2d(
                    input,
                    stride,
                    filter_size=[
                        1,
                        1,
                    ],
                    output_dim=output_dim,
                    padding=padding,
                    name="Conv_projection",
                )
            else:
                input = tf.pad(
                    tensor=input,
                    paddings=[
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, output_dim - input_dim],
                    ],
                )
        else:
            pass
        output = y_2 + input
    return output
