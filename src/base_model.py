import abc

import keras.backend as K

Relu = "relu"


def print_shape(x):
    print(x.get_shape())
    return x


class BaseModel():
    """
    Base model for every model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        self.model = None

    def loss_DSSIS(self, y_true, y_pred):
        u_true = K.mean(y_true)
        u_pred = K.mean(y_pred)
        var_true = K.var(y_true)
        var_pred = K.var(y_pred)
        std_true = K.std(y_true)
        std_pred = K.std(y_pred)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        ssim /= (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        return ((1.0 - ssim) / 2 + K.binary_crossentropy(y_pred,y_true,True)) /2.0

    @abc.abstractmethod
    def preprocess(self, batch, gt):
        raise NotImplemented

    @abc.abstractmethod
    def train_on(self, batch, gt):
        raise NotImplemented

    @abc.abstractmethod
    def test_on(self, batch):
        raise NotImplemented

    def get_model(self):
        return self.model
