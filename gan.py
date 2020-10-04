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
from tensorflow.keras.losses import BinaryCrossentropy

from gan_warnings import ModelResetWarning
from history import History

from functools import reduce
import inspect, os, pickle
from uuid import uuid4

from math import ceil

from tqdm.auto import trange
from matplotlib import pyplot as plt
import numpy as np

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
        - lr - learning rate (optional). (gen_lr, disc_lr) or int.
        """
        assert len(x_shape) == 3

        self.x_shape = x_shape
        self.kernal_size = kernal_size
        self.num_blocks = num_blocks
        self.min_filters = min_filters
        self.latent_dims = latent_dims
        self.strides = strides

        if isinstance(lr, float):
            self.gen_lr = lr
            self.disc_lr = lr
        else:
            assert hasattr(lr, "__len__")
            assert len(lr) == 2
            self.gen_lr, self.disc_lr = lr

        self.history = None

        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()
        self.combined = self.create_combined(verbose=verbose)

    def reset_models(self):
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.combined = self.create_combined()
        raise ModelResetWarning("All models did reset.")

    @staticmethod
    def add_activation(x, activation):
        if inspect.isfunction(activation):
            x = Activation(activation)(x)
        elif inspect.isclass(activation):
            if issubclass(activation, Layer):
                x = activation()(x)
        else:
            raise Exception(f"Invalid activation type: {type(activation)}.")
        return x

    def create_discriminator(self):

        def add_block(x, filters, activation=LeakyReLU, input_shape=None):
            if input_shape is not None:
                x = Conv2D(filters, self.kernal_size, padding="same",
                    input_shape=input_shape, data_format="channels_last")(x)
            else:
                x = Conv2D(filters, self.kernal_size, padding="same")(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, self.kernal_size, strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = self.add_activation(x, activation)
            x = Dropout(0.25)(x)
            return x

        inp = Input(shape=self.x_shape)

        x = add_block(inp, self.min_filters, input_shape=self.x_shape)
        for i in range(1, self.num_blocks):
            x = add_block(x, self.min_filters * (2**i))

        x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.4)(x)
        x = Dense(1)(x)

        discriminator = Model(inputs=inp, outputs=x, name="discriminator")

        discriminator.compile(loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(lr=self.disc_lr), metrics=["mae"])

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
            optimizer=Adam(lr=self.gen_lr), metrics=['mae'])

        if verbose:
            print(self.generator.summary(), "\n")
            print(self.discriminator.summary(), "\n")
            print(combined.summary(), "\n")

        return combined

    def train(self, real_train, num_epochs, batch_size,
            disc_updates=1, gen_updates=1, show_imgs=True, save_imgs=True,
            labels=(0, 1)):
        """
        Arguments:
        - real_train - preprocessed training examples of real data.
        Shape (num_samples, spatial_dim, spatial_dim, channels) (np array)
        - num_epochs - number of epochs to run (int)
        - batch_size - (int)
        - disc_updates - number of batch updates to perform per step for the
        Discriminator before switching to the Generator (int)
        - gen_updates - number of batch updates to perform per step for the
        Generator before switching to the Discriminator (int)
        - show_imgs - generate and display progress images (once per epoch) (bool)
        - save_imgs - save images in run directory (bool)
        - labels - (negative label, positive label)

        Outputs:
        - Updates weights of GAN instance
        - Rewrites run history of GAN instance
        - Creates new directory to save history pickle and progress images
        """
        assert self.x_shape == real_train.shape[1:]
        spatial_dim = self.x_shape[0]

        runs_root_dir = "Training_Runs"
        if not os.path.isdir(runs_root_dir):
            os.mkdir(runs_root_dir)

        run_id = uuid4()
        run_dir = f"{runs_root_dir}/{str(run_id)}"
        os.mkdir(run_dir)

        self.history = History(run_id)

        total_real = len(real_train)
        steps = int(total_real/batch_size)
        half_batch1 = int(batch_size/2)
        half_batch2 = int(batch_size - half_batch1)

        for epoch in range(num_epochs):
            pbar = trange(steps)
            for step in pbar:

                # Train Discriminator
                disc_loss = 0
                for _ in range(disc_updates):

                    # BATCH_NORM:

                    random_real_indxs = np.random.choice(total_real, batch_size)

                    disc_loss += 0.5 * self.discriminator.train_on_batch(real_train[random_real_indxs],
                                                    np.zeros([batch_size, 1])+labels[1])[1]

                    random_seed = np.random.randn(batch_size, self.latent_dims)

                    disc_loss += 0.5 * self.discriminator.train_on_batch(self.generator.predict(random_seed),
                                                    np.zeros([batch_size, 1])+labels[0])[1]

                    # MIXING BATCH:

                    # random_seed = np.random.randn(half_batch2, self.latent_dims)
                    #
                    # random_real_indxs = np.random.choice(total_real, half_batch1)
                    # batch_data = np.concatenate((real_train[random_real_indxs],
                    #                             self.generator.predict(random_seed)))
                    #
                    # discrim_labels = np.concatenate((np.zeros([half_batch1, 1])+labels[1],
                    #                                 np.zeros([half_batch2, 1])+labels[0]))
                    #
                    # shuffle_indxs = np.random.permutation(batch_size)
                    # discrim_labels = discrim_labels[shuffle_indxs]
                    # batch_x = batch_data[shuffle_indxs]
                    #
                    # disc_loss += self.discriminator.train_on_batch(batch_x, discrim_labels)[1]

                # Train Generator
                gen_loss = 0
                for _ in range(gen_updates):
                    # Create new images (separate from those used to train discriminator)
                    random_seed = np.random.randn(batch_size, self.latent_dims)
                    gen_loss += self.combined.train_on_batch(random_seed,
                                    np.zeros([batch_size, 1])+labels[1])[1]

                disc_loss /= disc_updates
                gen_loss /= gen_updates
                self.history.disc_loss.append(disc_loss)
                self.history.gen_loss.append(gen_loss)

                pbar.set_postfix({"disc_loss": disc_loss, "gen_loss": gen_loss})

            if (show_imgs or save_imgs):
                random_seed = np.random.randn(1, self.latent_dims)
                fake_img = self.generator.predict(random_seed).reshape(spatial_dim, spatial_dim)
                plt.imshow(fake_img)
                if save_imgs:
                    plt.savefig(f"{run_dir}/img_epoch{epoch+1}.png")
                if not show_imgs:
                    plt.clf()
                else:
                    plt.show()

        pickle.dump(self.history, open(f"{run_dir}/history.p", "wb"))
