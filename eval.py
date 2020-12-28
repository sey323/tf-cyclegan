import os
import sys
import json
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

sys.path.append("./")
import common.dataset.imload as imload
from models.CycleGan import Application as CycleGAN

import logging

# ログレベルを WARN に変更
logging.basicConfig(level=logging.WARNING)


def main(argv):
    img_path = os.getenv("DATASET_FOLDER", "dataset")
    save_path = os.getenv("SAVE_FOLDER", "results")

    # ファイルのパス
    folderA = FLAGS.folderA
    folderB = FLAGS.folderB
    model_path = FLAGS.model_path
    mode = FLAGS.mode

    # JSONからデータの読み込み
    with open(model_path + "/param.json") as param_json:
        param = json.load(param_json)
    resize = param["input_size"]
    channel = param["channel"]
    filter_size = param["filter_size"]
    gray = False if channel == 3 else True

    model = CycleGAN(
        input_size=resize,
        channel=channel,
        filter_size=filter_size,
        save_folder=save_path,
    )

    # モデルの復元
    model.restore(model_path)

    # モデルの評価
    if mode == "eval":
        source_images, target_images, file_names = imload.makeAB(
            folderA, folderB, img_size=resize, gray=gray
        )
        model.eval(source_images, target_images, file_names=file_names)
    elif mode == "freeze":
        model.freeze(model.save_folder)


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # 読み込むデータ
    flags.DEFINE_string("folderA", "", "生成するSource画像のパス")
    flags.DEFINE_string("folderB", "", "生成するTarget画像のパス")
    flags.DEFINE_string("model_path", "", "生成に利用する学習済みのモデルのパス")

    flags.DEFINE_string("mode", "eval", "生成,評価する時: eval, モデルを永続化: freeze")
    app.run(main)
