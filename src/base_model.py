import abc

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils.visualize_util import plot

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
        """
        Base Model for the models
        :param name: Name of the model (Log purpose)
        :param batchsize: batchsize to use
        """
        self.name = name
        self.model = None
        self.batch_size = batchsize
        self.output_size = None
        self.loss = self.loss_DSSIS_tf11
        self.optimizer = "rmsprop"

    @abc.abstractmethod
    def _build_model(self):
        """
        Build the model, this method needs to be implemented by the children class
        :return: None
        """
        raise NotImplementedError

    def preprocess(self, batch, gt):
        """
        Preprocess the datas to fit into the network
        :param batch: Array [samples,c,w,hframes] for input
        :param gt: Array [samples,w,h] the grounftruth
        :return: (batch,gt) batch = [samples,c,frames,w,h], gt = [sample,1,w,h]
        """
        batch = np.transpose(batch, [0, 4, 1, 2, 3])
        batch = np.array(batch)
        gt = gt.reshape([self.batch_size, 1, self.output_size, self.output_size])
        return (batch, gt)

    def l2_loss(self, y, y_pred):
        """
        L2 Loss function
        :param y: groundtruth
        :param y_pred: model's prediction
        :return: sum of squared error
        """
        return K.sum(K.pow(y - y_pred, 2))

    def loss_DSSIS_tf11(self, y_true, y_pred):
        """
        DSSIM loss function to get the structural dissimilarity between y_true and y_pred
        :param y_true: groundtruth
        :param y_pred: output from the model
        :return: The loss value
        :note Need tf0.11rc to work
        """
        y_true = tf.reshape(y_true, [self.batch_size] + get_shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [self.batch_size] + get_shape(y_pred)[1:])
        y_true_tf = tf.transpose(y_true, [0, 2, 3, 1])
        y_pred_tf = tf.transpose(y_pred, [0, 2, 3, 1])
        patches_true = tf.extract_image_patches(y_true_tf, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        patches_pred = tf.extract_image_patches(y_pred_tf, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

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
        norma = K.mean(K.abs(y_true - y_pred))
        norma = tf.select(tf.is_nan(norma), K.ones_like(norma), norma)
        return K.mean(((1.0 - ssim) / 2)) + (norma / 2)

    def build_model(self, loss="DSSIS", optimizer="rmsprop"):
        """
        Build the model
        :param loss: Loss function to use during training, str or function (y_pred,y_pred)
        :param optimizer: Optimizer to use during training string or Optimizer
        :return: None
        """
        if loss == "DSSIS":
            loss = self.loss_DSSIS_tf11
        elif loss == "l2":
            loss = self.l2_loss
        self.loss = loss
        self.optimizer = optimizer

        self.model = self._build_model()
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])
        self.model.summary()
        plot(self.model, to_file='output/{}.png'.format(self.name),show_shapes=True)

    def reset(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])

    def get_model(self):
        """
        Get the model builded
        :return: Model
        """
        return self.model
