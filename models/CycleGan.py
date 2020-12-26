import os
import sys
import logging
import cv2
import tensorflow as tf
import json
from datetime import datetime

tf.compat.v1.disable_eager_execution()

sys.path.append("./")
from .network import residual_unet, discriminator
import common.layer as layer
import common.loss_function as loss_function
import common.util.imutil as imutil


class Application:
    """
    CycleGAN
    """

    def __init__(
        self,
        input_size,
        layers=[64, 128, 256],
        filter_size=[4, 4],
        channel=3,
        drop_prob=0.5,
        identity_loss_penalty=100,
        cycle_loss_penalty=100,
        learn_rate=2e-4,
        gpu_config=tf.compat.v1.GPUOptions(allow_growth=True),
        save_folder="results/cyclegan",
    ):

        self.input_size = input_size
        self.channel = channel
        self.filter_size = filter_size
        self.drop_prob = drop_prob
        self.learn_rate = learn_rate
        self.identity_loss_penalty = identity_loss_penalty
        self.cycle_loss_penalty = cycle_loss_penalty

        # 保存するディレクトリの作成．
        if save_folder.startswith("@T"):
            self.save_folder = save_folder[2:]
            if os.path.exists(self.save_folder):  # フォルダが存在している場合は上書き
                shutil.rmtree(self.save_folder)
        else:
            now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.save_folder = os.path.join(save_folder, now)

        if self.save_folder and not os.path.exists(
            os.path.join(self.save_folder, "images")
        ):
            os.makedirs(os.path.join(self.save_folder, "images"))
            os.makedirs(os.path.join(self.save_folder, "eval"))
            with open(self.save_folder + "/param.json", "w") as param_file:
                # パラメータをJsonファイルに保存
                json.dump(self.__dict__, param_file, indent=2)

        # 画像保存用のディレクトリを作成
        os.makedirs(os.path.join(self.save_folder, "images", "trg2src"))
        os.makedirs(os.path.join(self.save_folder, "images", "src2trg"))

        self.gpu_config = tf.compat.v1.ConfigProto(gpu_options=gpu_config)

    def build_model(self):
        """
        モデルの生成
        """
        # 変数の定義
        self.source = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.input_size[0], self.input_size[1], self.channel],
            name="source_image",
        )
        self.target = tf.compat.v1.placeholder(
            tf.float32,
            [None, self.input_size[0], self.input_size[1], self.channel],
            name="target_image",
        )
        self._drop_prob = tf.compat.v1.placeholder(tf.float32)

        # Generatorのネットワークの構築
        logging.info("[BUILDING]\tGenerator")
        self.y_fake = residual_unet(
            self.source,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=False,
            name="_G",
        )
        self.y_cycled = residual_unet(
            self.y_fake,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=False,
            name="_F",
        )
        self.x_fake = residual_unet(
            self.target,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_F",
        )
        self.x_cycled = residual_unet(
            self.x_fake,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_G",
        )

        # 入力画像から同じ画像を生成する。
        self.x_same = residual_unet(
            self.source,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_F",
        )
        self.y_same = residual_unet(
            self.target,
            output_shape=self.input_size,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_G",
        )

        self.layers = [32, 64, 128, 256, 256, 256, 512, 512]

        # Discrimnatorのネットワークの構築
        logging.info("[BUILDING]\tGenerator")
        # D(x)の計算
        self.source_dx = discriminator(
            self.source,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=False,
            name="_Dx",
        )
        self.fake_x_dx = discriminator(
            self.x_fake,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_Dx",
        )

        # D(y)の計算
        self.target_dy = discriminator(
            self.target,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=False,
            name="_Dy",
        )
        self.fake_y_dy = discriminator(
            self.y_fake,
            layers=self.layers,
            channel=self.channel,
            filter_size=self.filter_size,
            reuse=True,
            name="_Dy",
        )

        # 最適化関数の定義
        logging.info("[BUILDING]\tOptimizer")
        # Generatorの損失関数
        # Gxの損失関数
        cycleloss_src = loss_function.cycle_constancy(
            self.y_cycled, self.source, alpha=10
        )
        cycleloss_trg = loss_function.cycle_constancy(
            self.x_cycled, self.target, alpha=10
        )
        total_cycle_loss = (cycleloss_src + cycleloss_trg) 

        # G(x)：Src->Trgの損失関数
        identityloss_trg = loss_function.l1(self.target, self.y_same)
        self.g_loss_g = (
            loss_function.cross_entropy(
                x=self.fake_y_dy,
                labels=tf.ones_like(self.fake_y_dy),
                name="d_loss_src2trg",
            )
            + total_cycle_loss * self.cycle_loss_penalty
            + identityloss_trg * self.identity_loss_penalty  # identity_loss
        )

        # F(y)：Trg->Srcの損失関数
        identityloss_src = loss_function.l1(self.source, self.x_same)
        self.g_loss_f = (
            loss_function.cross_entropy(
                x=self.fake_x_dx,
                labels=tf.ones_like(self.fake_x_dx),
                name="d_loss_src2trg",
            )
            + total_cycle_loss * self.cycle_loss_penalty
            + identityloss_src * self.identity_loss_penalty  # identity_loss
        )

        # D(x)：Srcの損失関数
        d_loss_real_x = loss_function.cross_entropy(
            self.source_dx,
            labels=tf.ones_like(self.source_dx),
            name="d_loss_real_x",
        )
        d_loss_fake_x = loss_function.cross_entropy(
            x=self.fake_x_dx,
            labels=tf.zeros_like(self.fake_x_dx),
            name="d_loss_fake_x",
        )  # 本物の画像を本物と識別する損失関数
        self.d_loss_x = d_loss_fake_x + d_loss_real_x

        # D(y)：Trgの損失関数
        d_loss_real_y = loss_function.cross_entropy(
            self.target_dy,
            labels=tf.ones_like(self.target_dy),
            name="d_loss_real_y",
        )
        d_loss_fake_y = loss_function.cross_entropy(
            x=self.fake_y_dy,
            labels=tf.zeros_like(self.fake_y_dy),
            name="d_loss_fake_y",
        )  # 本物の画像を本物と識別する損失関数
        self.d_loss_y = d_loss_fake_y + d_loss_real_y

        # 最適化関数の定義
        logging.info("[BUILDING]\tOptimizer")
        # define optimizer
        self.g_optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learn_rate, beta1=0.5
        ).minimize(
            self.g_loss_g,
            var_list=[
                x for x in tf.compat.v1.trainable_variables() if "Generator_G" in x.name
            ],
        )
        self.f_optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learn_rate, beta1=0.5
        ).minimize(
            self.g_loss_f,
            var_list=[
                x for x in tf.compat.v1.trainable_variables() if "Generator_F" in x.name
            ],
        )

        self.dx_optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learn_rate, beta1=0.5
        ).minimize(
            self.d_loss_x,
            var_list=[
                x
                for x in tf.compat.v1.trainable_variables()
                if "Discriminator_Dx" in x.name
            ],
        )
        self.dy_optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learn_rate, beta1=0.5
        ).minimize(
            self.d_loss_y,
            var_list=[
                x
                for x in tf.compat.v1.trainable_variables()
                if "Discriminator_Dy" in x.name
            ],
        )

        """Tensorboadに保存する設定"""
        logging.info("[BUILDING]\tSAVE Node")
        # パラメータの記録．
        tf.compat.v1.summary.scalar("g_loss_g", self.g_loss_g)
        tf.compat.v1.summary.scalar("g_loss_f", self.g_loss_f)
        tf.compat.v1.summary.scalar("d_loss_x", self.d_loss_x)
        tf.compat.v1.summary.scalar("d_loss_y", self.d_loss_y)
        tf.compat.v1.summary.scalar("identityloss_src", identityloss_src)
        tf.compat.v1.summary.scalar("identityloss_trg", identityloss_trg)
        tf.compat.v1.summary.scalar("cycleloss_trg", cycleloss_trg)
        tf.compat.v1.summary.scalar("cycleloss_src", cycleloss_src)

    def update(self, source_images, target_images):
        # update generator
        for i in range(4):
            _, _, g_loss_g, g_loss_f, summary = self.sess.run(
                [
                    self.g_optimizer,
                    self.f_optimizer,
                    self.g_loss_g,
                    self.g_loss_f,
                    self.summary,
                ],
                feed_dict={
                    self.source: source_images,
                    self.target: target_images,
                },
            )

        _, _, d_loss_x, d_loss_y, summary = self.sess.run(
            [
                self.dx_optimizer,
                self.dy_optimizer,
                self.d_loss_x,
                self.d_loss_y,
                self.summary,
            ],
            feed_dict={
                self.source: source_images,
                self.target: target_images,
            },
        )
        return {
            "g_loss": g_loss_g,
            "g_loss_f": g_loss_f,
            "d_loss_x": d_loss_x,
            "d_loss": d_loss_y,
            "summary": summary,
        }

    def create(self, source, target, save_folder, epoch, drop_prob=1):
        trg2src_image, src2trg_image = self.sess.run(
            [self.x_fake, self.y_fake],
            feed_dict={self.source: source, self.target: target},
        )
        cv2.imwrite(
            os.path.join(save_folder, "src2trg", "img_fixed_%d_result.png" % epoch),
            imutil.pairImage(
                source * 255.0,
                src2trg_image * 255.0,
                target * 255.0,
            ),
        )
        cv2.imwrite(
            os.path.join(save_folder, "trg2src", "img_fixed_%d_result.png" % epoch),
            imutil.pairImage(
                target * 255.0,
                trg2src_image * 255.0,
                source * 255.0,
            ),
        )

    def eval(self, source, target, file_names=None):
        """
        モデルの評価
        Parameters
        ---
            source : tensor
            target : tensor
            file_names : [str]
                入力画像に対応するファイル名の配列
        """
        fake = self.sess.run(self.y_fake, feed_dict={self.source: source})
        self.calculation(
            source, fake, target, save_folder=self.save_folder, file_names=file_names
        )

    def init_session(self):
        """
        TFセッションの初期化
        """
        self.sess = tf.compat.v1.Session(config=self.gpu_config)
        initOP = tf.compat.v1.global_variables_initializer()
        self.sess.run(initOP)

    def decrease_lr(self, decrease_rate=0.5):
        """
        学習の途中で学習率を下げる

        Args:
            decrease_rate (float):
                減少させるパーセンテージ
        """
        self.learn_rate *= decrease_rate

    def init_summry(self):
        """Tensorflowの学習経過を保存する
        Tensorboadで確認可能な実行ログを記録するSaverの作成
        """
        self.saver = tf.compat.v1.train.Saver()
        self.summary = tf.compat.v1.summary.merge_all()
        if self.save_folder:
            self.writer = tf.compat.v1.summary.FileWriter(
                self.save_folder, self.sess.graph
            )

    def add_summary(self, summary, step):
        """学習家庭の保存
        学習の途中結果を保存する

        Args:
            summary ():
                学習の途中結果のサマリー
            step (int):
                学習の回数
        """
        self.writer.add_summary(summary, step)

    def save(self, prefix):
        """モデルを保存
        学習結果を保存する

        Args:
            prefix (String):
                保存するモデルに付与するPrefix
        """
        self.saver.save(self.sess, os.path.join(self.save_folder, "model.ckpt"), prefix)

    def restore(self, file_name):
        """モデルの復元
        checkpointファイルからモデルを復元する。

        Args:
            file_name (String):
                読み込むモデルのパス
        """
        self.build_model()
        self.init_session()
        self.saver = tf.compat.v1.train.Saver()
        # モデルファイルが存在するかチェック
        ckpt = tf.train.get_checkpoint_state(file_name)
        if ckpt:
            print("[LOADING]\t" + file_name)
            ckpt_name = os.path.join(
                file_name, ckpt.model_checkpoint_path.split("/")[-1]
            )
            self.saver.restore(self.sess, ckpt_name)
            print("[LOADING]\t" + ckpt_name + " Complete!")
        else:
            print(file_name + " Not found")
            exit()

    def calculation(
        self, source, fake, target, save_folder, file_names=None, channel=3,
    ):
        """Calculation
        生成画像と正解画像を比較して損失を計算する。計算した画像ごとの損失関数はsave_folder/calc_loss.txtに保存される。

        Args:
            source ():
                入力画像
            fake ():
                生成された画像
            target ():
                正解画像
            save_folder (String):
                生成された画像を保存するフォルダ名
            file_names (String):
                画像のファイル名のリスト。こちらがNone出ない時、保存する画像にファイル名が付与される。
            channel (int):
                生成する画像のチャンネル
        """

        os.makedirs(os.path.join(save_folder, "fake"))
        os.makedirs(os.path.join(save_folder, "target"))
        for idx, _ in enumerate(fake):
            if file_names is None:
                file_name = idx
            else:  # ファイル名を撮ってくる
                file_name = file_names[idx][:-4]
            cv2.imwrite(
                os.path.join(
                    save_folder, "fake", "{}.png".format(file_name)
                ),
                fake[idx] * 255.0
            )
            cv2.imwrite(
                os.path.join(
                    save_folder, "target", "{}.png".format(file_name)
                ),
                target[idx] * 255.0
            )

    def freeze(self, frozen_graph_path, as_text=False):
        """
        学習したモデルと重み情報を.pbファイルに保存し永続化する。

        Args:
            frozen_graph_path (str):
                pbファイルを保存するディレクトリ
            as_text (bool):
                Text形式で保存する時: True, バイナリ形式で保存する時: False
        """
        # パラメータの固定
        # tf.io.write_graph(
        #     self.sess.graph_def, frozen_graph_path, "graph.pb", as_text=as_text
        # )
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(os.path.join(frozen_graph_path, "saved_model"))
        signature = tf.compat.v1.saved_model.predict_signature_def(inputs={'input': self.source}, outputs={'output': self.y_fake})
        builder.add_meta_graph_and_variables(sess=self.sess,
                                            tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                                            signature_def_map={'convert_g': signature})
        builder.save()
