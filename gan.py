"""
Generative Adversarial Network (GAN) implementation
Date: 02/10/2020

References:
Model architecture built on top of that from:
Michel Kana, Ph.D.
https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
"""

# TODO - must resize input images manually to match output shape of generator.
# Find a better way to do this!
# TODO - fix multiples on output shape for certain x_shapes.

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import \
Dense, Dropout, Conv2D, Flatten, BatchNormalization, Conv2DTranspose, \
Reshape, Input, GlobalAveragePooling2D, Activation

import tensorflow.keras.activations as activations
from tensorflow.keras.layers import LeakyReLU, Layer

from tensorflow.keras.optimizers import Adam

from gan_warnings import ModelResetWarning

from functools import reduce
import inspect

from math import ceil


class GAN:
    def __init__(self, x_shape, kernal_size,
                num_blocks=4, min_filters=16, latent_dims=100, strides=2, lr=2e-4,
                verbose=True):
        """
        Arguments:
        - x_shape - shape of a single x sample (excl. batch dimension)
        - kernal_size - kernal_size for convolutions
        - num_blocks - number of convolutional layer blocks (optional)
        - min_filters - number of starting convolutional filters (optional)
        - latent_dims - number of latent dimensions for Generator (optional)
        - strides - strides for convolutions (optional)
        - lr - learning rate (optional)
        """
        assert len(x_shape) == 3

        self.x_shape = x_shape
        self.kernal_size = kernal_size
        self.num_blocks = num_blocks
        self.min_filters = min_filters
        self.latent_dims = latent_dims
        self.strides = strides
        self.lr = lr

        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()
        self.combined = self.create_combined(verbose=verbose)

    def reset_models(self, model="both"):
        if model == "generator":
            self.generator = self.create_generator()
            raise ModelResetWarning("Generator did reset")
        elif model == "discriminator":
            self.discriminator = self.create_discriminator()
            raise ModelResetWarning("Discriminator did reset")
        else:
            self.generator = self.create_generator()
            self.discriminator = self.create_discriminator()
            raise ModelResetWarning("Generator and Discriminator did reset")

        self.combined = self.create_combined()
        raise ModelResetWarning("Combined model did reset")

    @staticmethod
    def add_activation(x, activation):
        if inspect.isfunction(activation):
            x = Activation(activation)(x)
        elif inspect.isclass(activation):
            if issubclass(activation, Layer):
                x = activation()(x)
        else:
            raise Exception(f"Invalid activation type {type(activation)}.")
        return x

    def create_discriminator(self, compile=True):

        def add_block(x, filters, activation=LeakyReLU, input_shape=None):
            if input_shape is not None:
                x = Conv2D(filters, self.kernal_size, padding="same",
                    input_shape=input_shape, data_format="channels_last")(x)
            else:
                x = Conv2D(filters, self.kernal_size, padding="same")(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, self.kernal_size, strides=2, padding="same")(x)
            x = self.add_activation(x, activation)
            x = Dropout(0.25)(x)
            return x

        inp = Input(shape=self.x_shape)

        x = add_block(inp, self.min_filters, input_shape=self.x_shape)
        for i in range(1, self.num_blocks):
            x = add_block(x, self.min_filters * (2**i))

        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation=activations.sigmoid)(x)

        discriminator = Model(inputs=inp, outputs=x, name="discriminator")

        if compile:
            discriminator.compile(loss="binary_crossentropy",
                optimizer=Adam(lr=self.lr), metrics=["mae"])

        return discriminator

    def create_generator(self):

        def add_block(x, filters, activation=LeakyReLU, input_shape=None):
            if input_shape is not None:
                x = Conv2DTranspose(filters, self.kernal_size, strides=self.strides,
                    padding="same", input_shape=input_shape,
                    data_format="channels_last")(x)
            else:
                x = Conv2DTranspose(filters, self.kernal_size, strides=self.strides,
                    padding="same")(x)
            x = BatchNormalization()(x)
            x = self.add_activation(x, activation)
            return x

        inp = Input(shape=(self.latent_dims, ))

        conv_output_sizes = self.conv_output_shape(self.x_shape[:-1], self.strides,
                            self.num_blocks)
        num_filters = self.min_filters * (2**(self.num_blocks-1))

        input_vol = reduce(lambda x, y: x*y, conv_output_sizes) * num_filters
        x = Dense(input_vol)(inp)
        x = BatchNormalization()(x)

        input_shape = (*conv_output_sizes, num_filters)
        x = Reshape(target_shape=input_shape)(x)

        for i in range(self.num_blocks-2, -1, -1):
            x = add_block(x, self.min_filters * (2**i))

        x = add_block(x, self.min_filters)

        # Project result to shape which can be handled by Discriminator
        # TODO - Introduces too many parameters!
        # x = Flatten()(x)
        # x = Dense(reduce(lambda x,y: x*y, self.x_shape))(x)
        # x = BatchNormalization()(x)
        # x = Activation(activations.tanh)(x)
        # x = Reshape(target_shape=self.x_shape)(x)

        # Add channels (e.g. RGB)
        x = Conv2D(self.x_shape[-1], self.kernal_size,
            padding="same", activation=activations.tanh)(x)

        return Model(inputs=inp, outputs=x, name="generator")

    @staticmethod
    def conv_output_shape(inp_sizes, stride, num_blocks):
        inp_sizes = list(inp_sizes)
        for _ in range(num_blocks):
            for i, size in enumerate(inp_sizes):
                inp_sizes[i] = ceil(float(size)/float(stride))

        return inp_sizes

    def create_combined(self, verbose=True):
        combined = Sequential(name="combined")
        combined.add(self.generator)
        combined.add(self.discriminator)

        self.discriminator.trainable = False

        combined.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=self.lr), metrics=['mae'])

        if verbose:
            print(self.generator.summary(), "\n")
            print(self.discriminator.summary(), "\n")
            print(combined.summary(), "\n")

        return combined

    def fit(self):
        # TODO
        pass
