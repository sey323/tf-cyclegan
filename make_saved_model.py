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


def eval_model(tf_model_filename, input_name, output_name, img):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # saved_model load
        tf.compat.v1.saved_model.loader.load(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING], tf_model_filename
        )
        # input
        i = sess.graph.get_tensor_by_name(input_name)
        # output
        r = sess.graph.get_tensor_by_name(output_name)
        return sess.run(r, feed_dict={i: img})


def get_model_info(tf_model_filename):
    ops = {}
    with tf.compat.v1.Session() as sess:
        with tf.compat.v1.gfile.GFile(tf_model_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
            for op in tf.compat.v1.get_default_graph().get_operations():
                ops[op.name] = [str(output) for output in op.outputs]
        writer = tf.compat.v1.summary.FileWriter("./logs")
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

    with open(tf_model_filename + "_param.json", "w") as f:
        f.write(json.dumps(ops))

    return sess


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

    source_images, target_images, file_names = imload.makeAB(
        folderA, folderB, img_size=resize, gray=gray
    )

    imgs = eval_model(
        tf_model_filename=os.path.join(model_path, "saved_model"),
        # input_name='import/source_image:0',
        input_name="source_image:0",
        output_name="Generator_G/image_reconstract/Tanh:0",
        img=source_images,
    )


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # 読み込むデータ
    flags.DEFINE_string("folderA", "", "生成するSource画像のパス")
    flags.DEFINE_string("folderB", "", "生成するTarget画像のパス")
    flags.DEFINE_string("model_path", "", "生成に利用する学習済みのモデルのパス")

    flags.DEFINE_string("mode", "generate", "生成する時: generate, モデルを永続化: freeze")
    app.run(main)
