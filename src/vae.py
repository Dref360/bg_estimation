'''
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6129
'''
import tensorflow as tf
from keras import backend as K
from keras import objectives
from keras.layers import Convolution2D, Convolution3D, MaxPooling3D, Deconvolution2D
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model

from src.base_model import BaseModel, Relu


class VAE(BaseModel):
    def __init__(self, sequence_size, img_size=321, batch_size=1, weight_file=None):
        self.output_size = 118
        BaseModel.__init__(self, "VAE", batch_size)
        self.sequence_size = sequence_size
        self.img_size = img_size
        self.build_model(loss=self.vae_loss)
        self.output_size = self.model.get_output_shape_at(-1)[-1]
        if weight_file:
            self.model.load_weights(weight_file)

    def sampling(self, args):
        latent_dim = 2
        epsilon_std = 0.01
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, latent_dim),
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        self.output_size = 118
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.output_size * self.output_size * objectives.binary_crossentropy(x, x_decoded_mean)
        z_exp = K.exp(self.z_log_var)
        z_exp = tf.select(tf.is_nan(z_exp), -1 * tf.ones_like(z_exp), z_exp)
        z_exp = tf.select(tf.is_inf(z_exp), tf.ones_like(z_exp), z_exp)
        self.z_log_var = tf.select(tf.is_nan(self.z_log_var), -1 * tf.ones_like(self.z_log_var), self.z_log_var)
        self.z_log_var = tf.select(tf.is_inf(self.z_log_var), tf.ones_like(self.z_log_var), self.z_log_var)

        self.z_mean = tf.select(tf.is_nan(self.z_mean), -1 * tf.ones_like(self.z_mean), self.z_mean)
        self.z_mean = tf.select(tf.is_inf(self.z_mean), tf.ones_like(self.z_mean), self.z_mean)

        kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - z_exp, axis=-1)
        return xent_loss + kl_loss

    def _build_model(self):
        latent_dim = 2
        intermediate_dim = 128
        epsilon_std = 0.01
        nb_conv = 3
        nb_filters = 64
        inputs = Input(shape=(3, self.sequence_size, self.img_size, self.img_size))
        x = Convolution3D(32, 3, 3, 3, activation=Relu, border_mode="same")(inputs)
        x = Convolution3D(64, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = MaxPooling3D()(x)
        x = Convolution3D(128, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = Convolution3D(32, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = MaxPooling3D()(x)
        x = Convolution3D(16, 3, 3, 3, activation=Relu, border_mode="same")(x)
        x = Convolution3D(8, 3, 3, 3, activation=Relu, border_mode="same")(x)
        flat = Flatten()(x)
        hidden = Dense(intermediate_dim, activation='relu')(flat)
        self.z_mean = Dense(latent_dim)(hidden)
        self.z_log_var = Dense(latent_dim)(hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(nb_filters * 29 * 29, activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (self.batch_size, nb_filters, 29, 29)
        else:
            output_shape = (self.batch_size, 29, 29, nb_filters)

        decoder_reshape = Reshape(output_shape[1:])

        decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                           output_shape,
                                           border_mode='same',
                                           subsample=(1, 1),
                                           activation='relu')

        decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                           output_shape,
                                           border_mode='same',
                                           subsample=(1, 1),
                                           activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (self.batch_size, nb_filters, 59, 59)
        else:
            output_shape = (self.batch_size, 59, 59, nb_filters)
        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 3, 3,
                                                  output_shape,
                                                  border_mode='valid',
                                                  subsample=(2, 2),
                                                  activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (self.batch_size, nb_filters, 119, 119)
        else:
            output_shape = (self.batch_size, 119, 119, nb_filters)
        decoder_deconv_4_upsamp = Deconvolution2D(nb_filters, 3, 3,
                                                  output_shape,
                                                  border_mode='valid',
                                                  subsample=(2, 2),
                                                  activation='relu')

        decoder_mean_squash = Convolution2D(1, 2, 2,
                                            border_mode='valid',
                                            activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        deconv_3_decoded = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_relu = decoder_deconv_4_upsamp(deconv_3_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
        vae = Model(input=inputs, output=x_decoded_mean_squash)
        return vae
