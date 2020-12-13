import tensorflow as tf
import common.layer as layer


def bottleneck_layer(input, output_dim, drop_prob, name):
    """
    bottleneck layer for dense net

    Args:
        input (tensor):
        output_dim (int):
            出力する特徴次元のサイズ
        drop_prob:
            ドロップアウトの割合
        name:
            モジュールに付与する固有の名前
    """
    with tf.compat.v1.variable_scope("bottleneck_" + name):
        input_dim = input.shape[-1]
        # BottleNec
        x = layer.batch_norm(input)
        x = layer.ReLU(x)
        x = layer.conv2d(
            input=x,
            stride=1,
            filter_size=[
                1,
                1,
            ],
            output_dim=output_dim,
            batch_norm=False,
            name="Conv_1_1",
        )
        x = tf.nn.dropout(x, 1 - (drop_prob))

        x = layer.batch_norm(x)
        x = layer.ReLU(x)
        x = layer.conv2d(
            input=x,
            stride=1,
            filter_size=[
                3,
                3,
            ],
            output_dim=output_dim,
            batch_norm=False,
            name="Conv_1_2",
        )
        x = tf.nn.dropout(x, 1 - (drop_prob))
    return x


def transition_layer(input, output_dim, drop_prob, name):
    """
    bottleneck layer for dense net

    Args:
        input (tensor):
        output_dim (int):
            出力する特徴次元のサイズ
        drop_prob:
            ドロップアウトの割合
        name:
            モジュールに付与する固有の名前
    """
    with tf.compat.v1.variable_scope("transition_" + name):
        x = layer.batch_norm(input)
        x = layer.ReLU(x)

        input_dim = int(x.shape[-1])

        x = layer.conv2d(
            input=x,
            stride=1,
            filter_size=[
                1,
                1,
            ],
            output_dim=output_dim,
            batch_norm=False,
            name="Conv_1_1",
        )
        x = tf.nn.dropout(x, 1 - (drop_prob))
        x = layer.average_pooling(x, pool_size=[2, 2], strides=1, padding="SAME")

        return x


def dense_block(input, nb_layers, output_dim, drop_prob, name):
    """
    Dense Block

    Args:
        input (tensor):
        nb_layers ():
        output_dim (int):
            出力する特徴次元のサイズ
        drop_prob:
            ドロップアウトの割合
        name:
            モジュールに付与する固有の名前
    """
    with tf.compat.v1.name_scope("Densenet_" + name):
        layers_concat = list()
        layers_concat.append(input)

        x = bottleneck_layer(
            input=input,
            output_dim=output_dim,
            drop_prob=drop_prob,
            name=name + "_bottleN_" + str(0),
        )

        layers_concat.append(x)

        # 結合するネットワークを生成
        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(
                x,
                output_dim=output_dim,
                drop_prob=drop_prob,
                name=name + "_bottleN_" + str(i + 1),
            )
            layers_concat.append(x)

        # 全ての入力を結合
        x = tf.concat(layers_concat, axis=3)

        return x
