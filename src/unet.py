from keras.layers import Convolution3D, Convolution2D, MaxPooling3D, Deconvolution2D, Reshape, Input, \
    BatchNormalization, Dropout, merge
from keras.models import Model

from src.base_model import BaseModel, Relu


class UNETModel(BaseModel):
    """
    This model is adding a connection between layer conv_2 and layer deconv_1 this allows the models to remember the input.
    """

    def __init__(self, sequence_size, img_size=321, batch_size=1, weight_file=None):
        """
        Initialize Unet model
        :param sequence_size: number of frame
        :param img_size: input layer size
        :param batch_size: batch size to use in the model
        :param weight_file: Use already initialized model. None for new
        """
        BaseModel.__init__(self, "UNETModel", batch_size)
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
        ul = Convolution3D(1, 3, 3, 3, activation=Relu, border_mode="same")(x)
        ul = Reshape([self.sequence_size, self.img_size // 2, self.img_size // 2])(ul)

        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), border_mode='valid')(x)

        x = Convolution3D(256, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Convolution3D(256, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=2)(x)

        x = Reshape([(self.sequence_size // 2) * 256, 80, 80])(x)
        x = Deconvolution2D((self.sequence_size // 2) * 256, 3, 3,
                            [self.batch_size, (self.sequence_size // 2) * 256, 160, 160],
                            border_mode='same',
                            subsample=(2, 2),
                            activation=Relu)(x)
        """Merge the U-Layer back into the deconvolution"""
        x = merge([x, ul], mode="concat", concat_axis=1)

        x = Convolution2D(512, 3, 3, activation=Relu, border_mode="same")(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(64, 3, 3, activation=Relu, border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization(axis=1)(x)
        x = Convolution2D(1, 3, 3, activation="sigmoid", border_mode="same")(x)
        model = Model(input=inputs, output=x)
        return model
