import cv2
import math
import numpy as np

"""
GANによって生成された画像を表示など．
前処理以外のGANのプログラムに影響する画像処理プログラム．
"""


def calcImageSize(height, width, stride, num):
    """
    畳み込み層のストライドの大きさと画像の畳み込みの回数から，
    再構成する画像の縦横を計算するプログラム
    """
    import math

    for i in range(num):
        height = int(math.ceil(float(height) / float(stride)))
        width = int(math.ceil(float(width) / float(stride)))
    return height, width


def pairImage(source, fake, target, channel=3):
    """
    入力画像，出力画像，正解画像を並べて表示
    """
    col = int(len(source) / 10) + 1
    sep = 10 if len(source) > 10 else len(source)
    h = source[0].shape[0] + 1
    w = source[0].shape[1] + 1
    r = np.zeros((h * sep, w * 3 * col, channel), dtype=np.float32)
    for idx, _ in enumerate(source):
        now_col = int(idx / sep) * 3  # 表示する列
        idx_y = idx % sep
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            0 + (w * now_col) : w - 1 + (w * now_col),
            :,
        ] = source[idx]
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            w + (w * now_col) : 2 * w - 1 + (w * now_col),
            :,
        ] = fake[idx]
        r[
            idx_y * h : h * (idx_y + 1) - 1,
            2 * w + (w * now_col) : 3 * w - 1 + (w * now_col),
            :,
        ] = target[idx]
    return r
