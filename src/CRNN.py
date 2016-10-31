from keras.layers import TimeDistributed, Convolution2D, Reshape, LSTM, UpSampling2D, MaxPooling2D, Input, Flatten
from keras.models import Model

from src.base_model import BaseModel, Relu#


class CRNN(BaseModel):
    def __init__(self, sequence_size, img_size=321, batch_size=1, weight_file=None):
        BaseModel.__init__(self, "CRNN", batch_size)
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.build_model()
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def _build_model(self):
        inputs = Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = TimeDistributed(Convolution2D(16, 3, 3, activation=Relu, border_mode="same"))(inputs)
        x = TimeDistributed(Convolution2D(32, 3, 3, activation=Relu, border_mode="same"))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Convolution2D(64, 3, 3, activation=Relu, border_mode="same"))(x)
        x = TimeDistributed(Convolution2D(32, 3, 3, activation=Relu, border_mode="same"))(x)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Convolution2D(8, 3, 3, activation=Relu, border_mode="same"))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(256)(x)
        x = Reshape([1, 16, 16])(x)
        x = Convolution2D(32, 3, 3, activation=Relu, border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(32, 3, 3, activation=Relu, border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(1, 3, 3, activation="sigmoid", border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        model = Model(input=inputs, output=x)
        return model
