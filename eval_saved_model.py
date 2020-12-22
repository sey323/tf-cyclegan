import cv2
import os
import sys
import json
from absl import app
from absl import flags
from datetime import datetime
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

sys.path.append("./")
import common.dataset.imload as imload
from models.CycleGan import Application as CycleGAN

import logging

# ログレベルを WARN に変更
logging.basicConfig(level=logging.DEBUG)

def save_created_files(fake_images, file_names, save_folder="results"):
    save_root_folder = os.path.join(save_folder, "fake", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    if not os.path.exists(save_root_folder):
        os.makedirs(save_root_folder)

    for idx, _ in enumerate(fake_images):
        file_name = file_names[idx][:-4]
        logging.debug("Processing :{}".format(file_name))
        cv2.imwrite(
            os.path.join(
                save_root_folder, "{}.png".format(file_name)
            ),
            cv2.resize(fake_images[idx] * 255.0, (1936,1216))
        )
            

def create_image_from_savedmodel(tf_model_filename, input_name, output_name, img)->{}:
    logging.info("Loading Saved Model :{}".format(tf_model_filename))
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # saved_model load
        tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], tf_model_filename)

        # 入出力のplaceholderを定義
        input_images = sess.graph.get_tensor_by_name(input_name)
        output_images = sess.graph.get_tensor_by_name(output_name)

        logging.debug("Input Placeholder :{}".format(input_images))
        logging.debug("Output Placeholder :{}".format(output_images))
        create_images = sess.run(output_images, feed_dict={input_images:img})

    return create_images

def get_model_info(tf_model_filename):
    ops = {}
    with tf.compat.v1.Session() as sess:
        with tf.compat.v1.gfile.GFile(tf_model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
            for op in tf.compat.v1.get_default_graph().get_operations():
                ops[op.name] = [str(output) for output in op.outputs]
        writer = tf.compat.v1.summary.FileWriter('./logs')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

    with open(tf_model_filename+'_param.json', 'w') as f:
        f.write(json.dumps(ops))

    return sess

def main(argv):
    img_path = os.getenv("DATASET_FOLDER", "dataset")
    save_path = os.getenv("SAVE_FOLDER", "results")

    # ファイルのパス
    folder = FLAGS.folder
    model_path = FLAGS.model_path

    # JSONからデータの読み込み
    with open(model_path + "/param.json") as param_json:
        param = json.load(param_json)
    resize = param["input_size"]
    channel = param["channel"]
    gray = False if channel == 3 else True

    source_images, file_names = imload.make(
        folder, img_size=resize, gray=gray
    )
    
    fake_images = create_image_from_savedmodel( 
                tf_model_filename=os.path.join(model_path, "saved_model"),
                input_name='source_image:0',
                output_name='Generator_G/image_reconstract/Tanh:0',
                img=source_images
        )

    save_created_files(fake_images, file_names)

if __name__ == "__main__":
    FLAGS = flags.FLAGS

    # 読み込むデータ
    flags.DEFINE_string("folder", "", "生成するSource画像のパス")
    flags.DEFINE_string("model_path", "", "生成に利用する学習済みのSaved_Modelのパス")

    app.run(main)
