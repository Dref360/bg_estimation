import keras.backend as K
import keras.layers as layers
import numpy as np
from keras.models import Model

from src.base_model import BaseModel, Relu


class CRNN(BaseModel):

    def test_on(self, batch):
        batch = np.transpose(batch, [0, 4, 1, 2, 3])
        return self.model.predict(batch, 1)

    def preprocess(self, batch, gt):
        batch = np.transpose(batch, [0, 4, 1, 2, 3])
        batch = np.array(batch)
        gt = gt.reshape([self.batch_size, 1, self.output_size, self.output_size])
        return (batch, gt)

    def train_on(self, batch, gt):
        batch, gt = self.preprocess(batch, gt)
        return self.model.train_on_batch(batch, gt)

    def __init__(self, sequence_size, img_size=321,batch_size=1, weight_file=None):
        BaseModel.__init__(self, "C3DModel")
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.model = self.build_model()
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def build_model(self):
        inputs = layers.Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = layers.TimeDistributed(layers.Convolution2D(16, 3, 3, 3, activation=Relu, border_mode="same"))(inputs)
        x = layers.TimeDistributed(layers.Convolution2D(16, 3, 3, 3, activation=Relu, border_mode="same"))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Convolution2D(16, 3, 3, 3, activation=Relu, border_mode="same"))(x)
        x = layers.TimeDistributed(layers.Convolution2D(16, 3, 3, 3, activation=Relu, border_mode="same"))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        x = layers.TimeDistributed(layers.Convolution2D(16, 3, 3, 3, activation=Relu, border_mode="same"))(x)
        x = layers.LSTM(64,32)(x)
        print(x.get_shape())
        x = layers.Reshape([self.sequence_size,321,321])(x)
        print(x.get_shape())
        x = layers.Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        print(x.get_shape())
        x = layers.Convolution2D(32, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.Convolution2D(32, 3, 3, activation=Relu)(x)
        x = layers.Convolution2D(4, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.Convolution2D(1, 3, 3, activation='sigmoid', border_mode="same")(x)
        print(x.get_shape())
        model = Model(input=inputs, output=x)
        model.compile(optimizer='rmsprop',
                      loss=self.loss_DSSIS,
                      metrics=['accuracy'])
        return model
