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


def l2(fake, y, alpha=0.01, name=""):
    """
    L2正則化

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
            計算されたL2損失の値
    """
    with tf.compat.v1.variable_scope("L2loss" + name):
        l2_loss = tf.reduce_mean(input_tensor=tf.square(fake - y)) * alpha
    return l2_loss


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


def cross_entropy_softmax(x, labels, name=""):
    """
    SoftmaxのCross Entropy損失。CNNなどの画像分類系のNNで用いる。

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
    with tf.compat.v1.variable_scope("cross_entropy_softmax" + name):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=x, name="cross_entropy_per_example"
        )
        cross_entropy_softmax = tf.reduce_mean(
            input_tensor=cross_entropy, name="cross_entropy_softmax"
        )

    tf.compat.v1.summary.scalar("cross_entropy_softmax_" + name, cross_entropy_softmax)
    logging.info(
        "[Loss]\tCross Entropy Loss with Softmax:{}".format(cross_entropy_softmax)
    )

    return cross_entropy_softmax


def ncc(fake, y, name=""):
    """
    正規化相互相関を計算する

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算された正規化相互相関の値
    """
    with tf.compat.v1.variable_scope("Normalized_Cross-Correlation" + name):
        num = tf.reduce_sum(input_tensor=tf.multiply(fake, y))
        den = tf.square(
            tf.reduce_sum(input_tensor=fake ** 2) * tf.reduce_sum(input_tensor=y ** 2)
        )
        ncc = 0 if den == 0 else num / den
    return ncc


def total_variation(input, width, beta=1, name=""):
    """
    Total Variation lossを計算する。

    Args
        input (tensor):
            生成された画像
        width (tensor):
            画像の1辺の長さ
        beta (int):
            損失関数に対する重み
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたTotal Variation lossの値
    """
    with tf.compat.v1.variable_scope("Total_Variation_loss" + name):
        a = tf.nn.l2_loss(input[:, 1:, :, :] - input[:, : width - 1, :, :]) / width
        b = tf.nn.l2_loss(input[:, :, :1, :] - input[:, :, : width - 1, :]) / width
        total_variation_loss = a + b
    tf.compat.v1.summary.scalar("Total_Variation_loss" + name, total_variation_loss)

    return total_variation_loss * beta


def binary_mask(fake, y, name="", threshold=0):
    """
    Binary Mask Lossを計算する。

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        name (str):
            Tensorboardで確認用の名称
        threshold (float):
            白黒の基準となる閾値

    Returns:
        tf.float32:
            計算されたBinary Mask Lossの値
    """
    b_fake = tf.cast((fake > threshold), tf.float32) * 1
    b_y = tf.cast((y > threshold), tf.float32) * 1
    with tf.compat.v1.variable_scope("Binary_Mask_Loss" + name):
        b_loss = tf.reduce_mean(input_tensor=tf.square(b_fake - b_y))

    tf.compat.v1.summary.scalar("Binary_Mask_Loss" + name, b_loss)

    return b_loss


def font_l1(target, fake, batch_num=32, alpha=1, name="", threshold=0.5):
    """
    Font l1Loss

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        batch_num (int):
            バッチ数
        alpha (float):
            損失関数の重み
        name (str):
            Tensorboardで確認用の名称
        threshold (float):
            白黒の基準となる閾値

    Returns:
        tf.float32:
            計算されたBinary Mask Lossの値
    """
    with tf.compat.v1.variable_scope("Font_L1Loss" + name):
        # 黒ピクセルの数を数える，
        black_pixel = tf.greater(threshold, target)
        as_ints = tf.cast(black_pixel, tf.int32)
        as_mean = tf.reduce_sum(input_tensor=as_ints, axis=[1, 2, 3]) + 1
        w_st = 1.0 / tf.cast(as_mean, tf.float32)

        # 黒ピクセルの平均を取得
        zeros = tf.zeros_like(target)
        new_tensor = tf.compat.v1.where(black_pixel, target, zeros)
        mean_pixel_value = tf.reduce_sum(
            input_tensor=new_tensor, axis=[1, 2, 3]
        ) / tf.cast(as_mean, tf.float32)
        w_b = tf.nn.softmax(mean_pixel_value) * batch_num
        weight = w_b * w_st

        # 次元配列に整形
        target = tf.reshape(target, [batch_num, -1])
        fake = tf.reshape(fake, [batch_num, -1])

        font_l1_loss = tf.reduce_mean(
            input_tensor=tf.reduce_sum(input_tensor=tf.abs(target - fake), axis=1)
            * weight
        )
    logging.debug("\t[Loss]\tFont L1 Loss:{}".format(font_l1_loss))
    return font_l1_loss


def hinge_loss(fake, y, name=""):
    """
    Hinge Loss

    Args
        fake (tensor):
            生成された画像
        y (tensor):
            正解画像
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたHinge Lossの値
    """
    with tf.compat.v1.variable_scope("Hinge_Loss" + name):
        hinge_loss_val = tf.compat.v1.losses.hinge_loss(
            fake,
            y,
            weights=1.0,
            scope=None,
            loss_collection=tf.compat.v1.GraphKeys.LOSSES,
            # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
        )
    return hinge_loss_val


def content_loss(fake, y, name=""):
    """
    Neural Style Transferの手法より、コンテンツロスを計算する。

    Args
        fake (tensor):
            生成画像の任意の層
        y (tensor):
            正解画像の任意の層
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたコンテンツロスの値
    """
    with tf.compat.v1.variable_scope("Content_Loss"):
        m, n_h, n_w, n_c = fake.get_shape().as_list()
        con_ten_unrolled = tf.reshape(fake, shape=[n_h * n_w, n_c, -1])
        gen_ten_unrolled = tf.reshape(y, shape=[n_h * n_w, n_c, -1])
        loss = tf.reduce_sum(
            input_tensor=tf.square(con_ten_unrolled - gen_ten_unrolled)
        ) / (4 * n_h * n_w * n_c)

    tf.compat.v1.summary.scalar("Content_Loss" + name, loss)
    return loss


def gram_matrix(mat):
    m, n_h, n_w, n_c = mat.get_shape().as_list()
    mat_reshaped = tf.reshape(tf.transpose(a=mat), shape=[n_c, n_h * n_w, -1])
    gram_mat = tf.matmul(
        mat_reshaped, mat_reshaped, transpose_a=False, transpose_b=True
    )

    return gram_mat


def style_loss_per_layer(sty_ten, gen_ten):
    m, n_h, n_w, n_c = sty_ten.get_shape().as_list()
    sty_ten_gram_mat = gram_matrix(sty_ten)
    gen_ten_gram_mat = gram_matrix(gen_ten)
    sty_loss = tf.reduce_sum(
        input_tensor=tf.square(sty_ten_gram_mat - gen_ten_gram_mat)
    ) / (4 * n_c ** 2 * n_h ** 2 * n_w ** 2)

    return sty_loss


def style_loss(fake_features, y_faetures, name=""):
    """
    Neural Style Transferの手法より、スタイル損失を計算する。
    Args
        fake_features (tensor):
            生成画像のスタイル特徴量のリスト
        y_faetures (tensor):
            正解画像のスタイル特徴量のリスト
        name (str):
            モジュールの名前
    """
    with tf.compat.v1.variable_scope("Style_Loss"):
        loss = 0
        for target_feature, fake_feature in zip(fake_features, y_faetures):
            loss += style_loss_per_layer(target_feature, fake_feature)
    tf.compat.v1.summary.scalar("Style_Loss" + name, loss)
    return loss


def kldivergence(fake, y, name=""):
    """
    入力された２つの分布のKLダイバージェンスを計算する。

    Args
        fake (tensor):
            生成画像
        y (tensor):
            正解画像
        name (str):
            Tensorboardで確認用の名称

    Returns:
        tf.float32:
            計算されたkldivergence損失の値
    """
    with tf.compat.v1.variable_scope("KLDivergence_" + name):
        kl_loss = tf.keras.losses.KLDivergence()(fake, y)
    return kl_loss
