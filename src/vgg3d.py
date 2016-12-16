from keras.layers import Convolution3D, Convolution2D, MaxPooling3D, Deconvolution2D, Reshape, Input, \
    BatchNormalization, Dropout
from keras.models import Model

from src.base_model import BaseModel, Relu


class VGG3DModel(BaseModel):
    """
    This model is using the first 5 blocks of VGG19 with the Conv2D replaced with Conv3D. This model is our baseline.
    """
    def __init__(self, sequence_size, img_size=321, batch_size=1, weight_file=None):
        """
        Initialize VGG3D model
        :param sequence_size: number of frame
        :param img_size: input layer size
        :param batch_size: batch size to use in the model
        :param weight_file: Use already initialized model. None for new
        """
        BaseModel.__init__(self, "C3DModel", batch_size)
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.build_model()
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def _build_model(self):
        inputs = Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = Convolution3D(64, 3, 3, 3, activation=Relu, border_mode="same")(inputs)
        x = BatchNormalization(axis=2)(x)
        x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), border_mode='valid')(x)

        x = Convolution3D(128, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=2)(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), border_mode='valid')(x)

        x = Convolution3D(256, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Convolution3D(256, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=2)(x)

        x = Reshape([(self.sequence_size // 2) * 256, 80, 80])(x)
        x = Deconvolution2D((self.sequence_size // 2) * 256, 3, 3,
                            [self.batch_size, (self.sequence_size // 2) * 256, 159, 159],
                            border_mode='same',
                            subsample=(2, 2),
                            activation=Relu)(x)
        x = Convolution2D(512, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(1, 3, 3, activation="sigmoid", border_mode="same")(x)
        model = Model(input=inputs, output=x)
        return model
