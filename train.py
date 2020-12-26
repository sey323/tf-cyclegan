import os
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from models.CycleGan import Application as CycleGan
import common.dataset.imload as imload
from common.dataset.Batchgen import Batchgen
from common.dataset.Trainer import TransferTrainer


def main(argv):
    print("[LOADING]\tmodel parameters loding")

    # 環境変数の取得
    save_path = os.getenv("SAVE_FOLDER", "results")

    # ファイルのパス
    folderA = FLAGS.folderA
    folderB = FLAGS.folderB
    folderA_test = FLAGS.folderA_test
    folderB_test = FLAGS.folderB_test

    resize = [FLAGS.resize, FLAGS.resize]
    gray = FLAGS.gray
    channel = 1 if gray else 3

    # 学習に関するパラメータ
    learn_rate = FLAGS.learn_rate
    max_epoch = FLAGS.max_epoch
    batch_num = FLAGS.batch_size
    identity_loss_penalty = FLAGS.identity_loss_penalty
    cycle_loss_penalty = FLAGS.cycle_loss_penalty
    drop_prob = FLAGS.drop_prob

    if FLAGS.save_folder == "":
        save_path = save_path + "/" + FLAGS.save_folder
    else:
        save_path = FLAGS.save_folder
    
    print("[LOADING]\tTraining data")
    # 画像の読み込み
    source_images, target_images, _ = imload.makeAB(
        folderA, folderB, gray=gray, img_size=resize
    )
    # target_images, _ = imload.make(folderB, gray=gray, img_size=resize)

    # テスト画像の読み込み
    var_source_images, var_target_images, _ = imload.makeAB(
        folderA_test, folderB_test, gray=gray, img_size=resize
    )
    # var_target_images, _ = imload.make(folderB_test, gray=gray, img_size=resize)
    file_num = len(source_images)
    test_num = len(var_source_images)
    print("Train File Num\t%d" % (len(source_images)))
    print("Val File Num\t%d" % (len(var_source_images)))

    # 学習用バッチの作成
    file_num = len(source_images)
    dummy = np.ones(file_num)
    source = Batchgen(source_images, label=dummy)
    target = Batchgen(target_images, label=dummy)

    # テスト用バッチの作成．．
    var_dummy = np.ones(test_num)
    var_source = Batchgen(var_source_images, label=var_dummy)
    var_target = Batchgen(var_target_images, label=var_dummy)

    # 学習機の作成
    trainer = TransferTrainer(max_epoch=max_epoch, batch_num=batch_num)

    # モデルの作成
    print("[LOADING]\tCycleGAN")
    model = CycleGan(
        input_size=resize,
        channel=channel,
        cycle_loss_penalty=cycle_loss_penalty,
        identity_loss_penalty=identity_loss_penalty,
        drop_prob=drop_prob,
        learn_rate=learn_rate,
        save_folder=save_path,
    )

    # 学習の実行
    trainer.train(model, source, target, var_source, var_target)

    # モデルの永続保存
    model.freeze(model.save_folder,)


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # 読み込む画像周り
    flags.DEFINE_string("folderA", "", "学習するSource画像のパス")
    flags.DEFINE_string("folderB", "", "学習するTarget画像のパス")
    flags.DEFINE_string("folderA_test", "", "テスト用Source画像のパス")
    flags.DEFINE_string("folderB_test", "", "テスト用Target画像のパス")

    flags.DEFINE_integer("resize", 64, "モデルに入力する画像の1辺のサイズ")
    flags.DEFINE_boolean("gray", False, "濃淡画像に変換するかどうか")

    # GANの学習パラメータ
    flags.DEFINE_integer("batch_size", 64, "ミニバッチサイズ")
    flags.DEFINE_float("learn_rate", 0.002, "学習率")
    flags.DEFINE_integer("max_epoch", 100, "学習Epoch数")
    flags.DEFINE_float("drop_prob", 0.5, "ドロップアウトの確率")
    flags.DEFINE_float("identity_loss_penalty", 10, "identity lossの重み")
    flags.DEFINE_float("cycle_loss_penalty", 10, "cycle lossの重み")

    # 保存フォルダの指定
    flags.DEFINE_string("save_folder", "", "学習結果を保存するパス")

    app.run(main)
