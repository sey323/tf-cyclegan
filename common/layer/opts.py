import logging
import tensorflow as tf


def batch_norm(
    x,
    decay=0.9,
    updates_collections=None,
    epsilon=1e-5,
    scale=True,
    is_training=True,
    scope=None,
):
    """
    batch_norm
    x : input
    """
    return tf.keras.layers.BatchNormalization(
        epsilon=epsilon,
        scale=scale,
        trainable=is_training,
    )(x)
