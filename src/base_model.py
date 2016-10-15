import abc

import keras.backend as K
import numpy as np
import tensorflow as tf

from lib.utils import get_shape

Relu = "relu"


def print_shape(x):
    print(x.get_shape())
    return x


class BaseModel():
    """
    Base model for every model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, batchsize):
        self.name = name
        self.model = None
        self.batch_size = batchsize

    @abc.abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def preprocess(self, batch, gt):
        batch = np.transpose(batch, [0, 4, 1, 2, 3])
        batch = np.array(batch)
        gt = gt.reshape([self.batch_size, 1, self.output_size, self.output_size])
        return (batch, gt)

    def l2_loss(self, y, y_pred):
        return K.sum(K.pow(y - y_pred, 2))

    def loss_DSSIS_tf11(self, y_true, y_pred):
        """Need tf0.11rc to work"""
        y_true = tf.reshape(y_true, [self.batch_size] + get_shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [self.batch_size] + get_shape(y_pred)[1:])
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
        patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

        u_true = K.mean(patches_true, axis=3)
        u_pred = K.mean(patches_pred, axis=3)
        var_true = K.var(patches_true, axis=3)
        var_pred = K.var(patches_pred, axis=3)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= denom
        ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
        return K.mean(((1.0 - ssim) / 2))

    def build_model(self, loss="DSSIS", optimizer="rmsprop"):
        if loss == "DSSIS":
            loss = self.loss_DSSIS_tf11
        elif loss == "l2":
            loss = self.l2_loss

        self.model = self._build_model()
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])
        self.model.summary()

    def get_model(self):
        return self.model
