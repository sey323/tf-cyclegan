import tensorflow as tf
import sys

try:
    tf_conv = tf.lite.TFLiteConverter.from_saved_model(
        "results/2020-12-16-193019/saved_model"
    )
    # tf_conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # tf_conv.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    tf_conv("\n\nSuccessfully loaded\n\n")
    lite_model = tf_conv.convert()

except Exception as e:
    print("\n\nFailed to load\n\n", e)
    sys.exit()

with open("digit_model.tflite", "wb") as w:
    w.write(lite_model)
