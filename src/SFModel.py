"""Single frame model"""

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Input, \
    BatchNormalization, Dropout, merge
from keras.models import Model

from src.base_model import BaseModel, Relu


class SFModel(BaseModel):
    def __init__(self, sequence_size, img_size=320, batch_size=1, weight_file=None):
        """
        Initialize SFModel model
        :param sequence_size: number of frame
        :param img_size: input layer size
        :param batch_size: batch size to use in the model
        :param weight_file: Use already initialized model. None for new
        """
        assert sequence_size == 1, "SFModel only accept single frame"
        BaseModel.__init__(self, "SFModel", batch_size)
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.build_model(loss=self.l2_loss)
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def preprocess(self, batch, gt):
        """
        Preprocess the datas to fit into the network
        :param batch: Array [samples,c,frames,w,h] for input
        :param gt: Array [samples,w,h] the grounftruth
        :return: (batch,gt) batch = [samples,c,w,h], gt = [sample,1,w,h]
        """
        batch = np.transpose(batch, [0, 4, 1, 2, 3])
        samples, c, frames, w, h = batch.shape
        assert frames == 1, "PROBLEM"
        batch = batch.reshape([samples, c, w, h])
        batch = np.array(batch)
        gt = gt.reshape([self.batch_size, 1, self.output_size, self.output_size])
        return (batch, gt)

    def _build_model(self):
        inputs = Input(shape=(3, self.img_size, self.img_size))
        # 320
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(inputs)
        last_320 = BatchNormalization(axis=2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), border_mode="same")(last_320)
        # 160
        x = Convolution2D(128, 3, 3, activation=Relu, border_mode="same")(x)
        last_160 = BatchNormalization(axis=2)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(last_160)
        #80
        x = Convolution2D(256, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.25)(x)
        x = Convolution2D(256, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=2)(x)
        #160
        x = Deconvolution2D(256, 3, 3,
                            [self.batch_size, 256, 160, 160],
                            border_mode='same',
                            subsample=(2, 2),
                            activation=Relu)(x)
        x = merge([x, last_160], mode="concat", concat_axis=1)
        x = Convolution2D(512, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization(axis=1)(x)
        # 320
        x = Deconvolution2D(256, 3, 3,
                            [self.batch_size, 256, 320, 320],
                            border_mode='same',
                            subsample=(2, 2),
                            activation=Relu)(x)
        x = merge([x, last_320], mode="concat", concat_axis=1)
        x = Convolution2D(512, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization(axis=1)(x)

        x = Convolution2D(1, 3, 3, activation="sigmoid", border_mode="same")(x)
        model = Model(input=inputs, output=x)
        return model
