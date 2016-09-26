from src.base_model import BaseModel, Relu, print_shape
import keras.backend as K
import keras.layers as layers
from keras.models import Model
import keras
import numpy as np


class C3DModel(BaseModel):
    def loss_img(self,y,y_pred):
        return K.mean(K.binary_crossentropy(y_pred,y))


    def train_on(self, batch, gt):
        batch = np.transpose(batch,[3,0,1,2])
        batch = np.array([batch])
        gt = gt.reshape([self.batch_size,1,self.img_size,self.img_size])
        return self.model.fit(batch ,gt,batch_size=1)

    def __init__(self, sequence_size, img_size=321, weight_file=None):
        BaseModel.__init__(self, "C3DModel")
        self.batch_size = 1
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.model = self.build_model()
        if weight_file:
            self.model.load_weights(weight_file)

    def build_model(self):
        inputs = layers.Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = layers.Convolution3D(16, 3, 3, 3, activation=Relu,border_mode="same")(inputs)
        print(x.get_shape())
        x = layers.Lambda(lambda  x1 :K.mean(x1,axis=2))(x)
        print(x.get_shape())
        x = layers.Convolution2D(64, 3, 3, activation=Relu,border_mode="same")(x)
        print(x.get_shape())
        x = layers.Convolution2D(32, 3, 3, activation=Relu,border_mode="same")(x)
        x = layers.Convolution2D(16, 3, 3, activation=Relu,border_mode="same")(x)
        x = layers.Convolution2D(4, 3, 3, activation=Relu,border_mode="same")(x)
        x = layers.Convolution2D(1, 3, 3, activation='sigmoid',border_mode="same")(x)
        model = Model(input=inputs, output=x)
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
