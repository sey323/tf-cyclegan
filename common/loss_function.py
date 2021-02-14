import logging
import tensorflow as tf


def l1(fake, y, alpha=1, name=""):
    """
    L1正則化

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        alpha (float):
            ロスに対してかける重み
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたL1損失の値
    """
    with tf.compat.v1.variable_scope("L1loss" + name):
        l1_loss = tf.reduce_mean(input_tensor=abs(fake - y)) * alpha

    tf.compat.v1.summary.scalar("L1loss" + name, l1_loss)
    logging.debug("\t[Loss]\tL1 Loss:{}".format(l1_loss))
    return l1_loss


def cycle_constancy(fake, y, alpha=1, name=""):
    """
    Cycle constancy損失
    (For Cycle GAN)

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        alpha (float):
            ロスに対してかける重み
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたCycle constancy損失の値
    """
    with tf.compat.v1.variable_scope("Cycleloss_" + name):
        cycle_constancy_loss = tf.reduce_mean(input_tensor=abs(fake - y)) * alpha
    logging.debug("\t[Loss]\tCycle constancy Loss:{}".format(cycle_constancy_loss))
    return cycle_constancy_loss


def constancy(fx, fgfx, alpha=15, name=""):
    """
    恒常化損失関数(For Domain Transfer Network)
    特徴空間同士の類似度を計算する。

    Args
        fx (tensor):
            入力画像の特徴空間
        fgfx (tensor):
            生成された画像の特徴空間
        alpha (float):
            ロスに対してかける重み
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算された恒常化損失関数の値
    """
    with tf.compat.v1.variable_scope("constancy_" + name):
        constancy_loss = tf.reduce_mean(input_tensor=tf.square(fx - fgfx)) * alpha
    return constancy_loss


def cross_entropy(x, labels, name=""):
    """
    SigmoidのCross Entropy損失。CNNなどの画像分類系のNNで用いる。

    Args
        x (tensor):
            NNの最後の出力
        labels (tensor):
            正解ラベル
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたCross Entropy損失の値
    """
    with tf.compat.v1.variable_scope("cross_entropy_sigmoid" + name):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=x, name="cross_entropy_per_example"
        )
        cross_entropy_sigmoid = tf.reduce_mean(
            input_tensor=cross_entropy, name="cross_entropy_sigmoid"
        )

    tf.compat.v1.summary.scalar(name, cross_entropy_sigmoid)
    logging.debug("\t[Loss]\tCross Entropy Loss:{}".format(cross_entropy_sigmoid))

    return cross_entropy_sigmoid
