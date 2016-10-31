import keras.layers as layers
from keras.models import Model

from src.base_model import BaseModel, Relu


class C3DModel(BaseModel):
    def __init__(self, sequence_size, img_size=321,batch_size=1, weight_file=None):
        BaseModel.__init__(self, "C3DModel",batch_size)
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.build_model()
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def _build_model(self):
        inputs = layers.Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = layers.Convolution3D(16, 3, 3, 3, activation=Relu, border_mode="same")(inputs)
        x = layers.Convolution3D(1, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.Reshape([self.sequence_size,321,321])(x)
        x = layers.Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Convolution2D(32, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.Convolution2D(32, 3, 3, activation=Relu)(x)
        x = layers.Convolution2D(4, 3, 3, activation=Relu, border_mode="same")(x)
        x = layers.Convolution2D(1, 3, 3, activation='sigmoid', border_mode="same")(x)
        model = Model(input=inputs, output=x)

        return model
