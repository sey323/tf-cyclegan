import os
import time


class TransferTrainer:
    """
    モデルの学習を行う
    """

    def __init__(
        self,
        batch_num=50,
        max_epoch=10000,
    ):
        self.max_epoch = max_epoch
        self.batch_num = batch_num

    def train(
        self, model, source_batch, target_batch, var_source_batch, var_target_batch
    ):
        model.build_model()
        model.init_session()
        model.init_summry()

        # 学習中の性能確認用の画像を取得
        var_source_fixed, _ = var_source_batch.getBatch(self.batch_num)
        var_target_fixed, _ = var_target_batch.getBatch(
            self.batch_num, idx=var_source_batch.getIndex()
        )

        step = epoch = 0
        start = time.time()
        while source_batch.getEpoch() < self.max_epoch:
            step += 1

            source_images, _ = source_batch.getBatch(self.batch_num)
            target_images, _ = target_batch.getBatch(
                self.batch_num, idx=source_batch.getIndex()
            )

            update_result_dict = model.update(source_images, target_images)

            if step > 0 and step % 100 == 0:
                model.add_summary(update_result_dict["summary"], step)

            # Epoch毎に画像を保存．
            if epoch != source_batch.getEpoch():
                epoch = source_batch.getEpoch()

                train_time = time.time() - start
                del update_result_dict["summary"]

                print(
                    "Epoch {}: {}; time/step={} sec".format(
                        epoch,
                        update_result_dict,
                        train_time,
                    )
                )

                start = time.time()

                model.create(
                    var_source_fixed,
                    var_target_fixed,
                    os.path.join(
                        model.save_folder,
                        "images",
                    ),
                    epoch,
                )
                if epoch == int(self.max_epoch * 0.5) or epoch == int(
                    self.max_epoch * 0.75
                ):
                    model.decrease_lr(0.1)
                    print("[INFOMATION]\tReduce Learning Rate")
                model.saver.save(
                    model.sess, os.path.join(model.save_folder, "model.ckpt"), epoch
                )
