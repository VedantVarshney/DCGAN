"""
Deep Convolutional Generative Adversarial Network (GAN) with additional features
for improved stability and convergence
Date: 02/10/2020

References:
Michel Kana, Ph.D.
https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
"""

# TODO - must resize input images manually to match output shape of generator.
# Find a better way to do this!
# TODO - fix multiples on output shape for certain x_shapes.

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import \
Dense, Dropout, Conv2D, Flatten, BatchNormalization, Conv2DTranspose, \
Reshape, Input, GlobalAveragePooling2D, Activation, Lambda, Concatenate, \
LeakyReLU, Layer

import tensorflow.keras.activations as activations

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import backend as kback

from gan_warnings import ModelResetWarning
from history import History

from functools import reduce
import inspect, os, pickle, types
from contextlib import redirect_stdout
from uuid import uuid4

from tqdm.auto import trange
from matplotlib import pyplot as plt
import numpy as np

class GAN:
    def __init__(self, x_shape, kernal_size,
                num_blocks=2, min_filters_discr=64, min_filters_gen=256,
                latent_dims=100, strides=2, lr=1e-4, beta_1=0.45, verbose=True):
        """
        Arguments:
        - x_shape - shape of a single x sample (excl. batch dimension)
        - kernal_size - kernal_size for convolutions
        - num_blocks - number of convolutional layer blocks (optional)
        - min_filters - number of starting convolutional filters (optional)
        - latent_dims - number of latent dimensions for Generator (optional)
        - strides - strides for convolutions (optional)
        - lr - learning rate (optional). (gen_lr, disc_lr) or int.
        - beta_1 - beta_1 param for all Adam optimisers
        """
        assert len(x_shape) == 3

        for spatial_dim in x_shape[:-1]:
            if not (spatial_dim/strides**num_blocks).is_integer():
                raise Exception("Num_blocks incompatible with spatial dimension.\
                Resize input or change num_blocks.")

        self.x_shape = x_shape
        self.kernal_size = kernal_size
        self.num_blocks = num_blocks
        self.min_filters_discr = min_filters_discr
        self.min_filters_gen = min_filters_gen
        self.latent_dims = latent_dims
        self.strides = strides
        self.beta_1 = beta_1

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
        self.combined = self.create_combined()

        if verbose:
            self.print_summary()

    @classmethod
    def load(cls, gan_fpath, model_dir=None, verbose=True):
        model = pickle.load(open(gan_fpath, "rb"))
        if model_dir is None:
            model.reset_models()
        else:
            model.load_model(model_dir, verbose=verbose)
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        # Prevent models and history from being pickled
        unpicklable_attribs = ["discriminator", "generator", "combined", "history"]
        for param in unpicklable_attribs:
            state[param] = None
        return state

    def __setstate__(self, state):
        # Do not need to validate attributes as would have been checked in init
        # of previous instance
        self.__dict__.update(state)

    def save(self, gan_fpath, model_dir="model"):
        # Save parameters
        pickle.dump(self, open(gan_fpath, "wb"))

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # Save models in preferred format (not pickle) along with history
        self.discriminator.trainable = False
        self.combined.save(f"{model_dir}/combined")
        self.discriminator.trainable = True
        self.generator.save(f"{model_dir}/generator")
        self.discriminator.save(f"{model_dir}/discriminator")
        self.discriminator.trainable = False

        pickle.dump(self.history, open(f"{model_dir}/history.p", "wb"))

    def load_model(self, load_dir, verbose=True):
        # load models and associated history
        # Usually called manually after a pickle load of all other attributes
        self.discriminator = load_model(f"{load_dir}/discriminator")
        self.discriminator.trainable = False
        self.generator = load_model(f"{load_dir}/generator")
        self.combined = load_model(f"{load_dir}/combined")
        self.history = pickle.load(open(f"{load_dir}/history.p", "rb"))
        if verbose:
            self.print_summary()

        # TODO - check whether existing parameters and attributes agree with those
        # in the loaded model.

    def __repr__(self):
        return str(self.__dict__)

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

        def add_block(x, filters, activation=LeakyReLU, input_shape=None,
                    dropout=0.3):
            if input_shape is not None:
                x = Conv2D(filters, self.kernal_size, strides=self.strides, padding="same",
                    input_shape=input_shape, data_format="channels_last")(x)
            else:
                x = Conv2D(filters, self.kernal_size, strides=self.strides, padding="same")(x)

            x = self.add_activation(x, activation)
            x = Dropout(dropout)(x)
            return x

        def add_minibatch_discrimination(x, num_kernals=5, kernal_dim=3):
            vol = int(num_kernals * kernal_dim)
            x_sub = Dense(vol, use_bias=False)(x)
            x_sub = Reshape((num_kernals, kernal_dim))(x_sub)
            x_sub = Lambda(mbatch_discriminate)(x_sub)
            concat = Concatenate(axis=1)([x, x_sub])
            return concat

        def mbatch_discriminate(x_sub):
            x_sub = kback.expand_dims(x_sub, 3) - \
                kback.expand_dims(kback.permute_dimensions(x_sub, [1, 2, 0]), 0)
            x_sub = kback.sum(kback.abs(x_sub), axis=2)
            x_sub = kback.sum(kback.exp(-x_sub), axis=2)
            return x_sub

        inp = Input(shape=self.x_shape)

        x = add_block(inp, self.min_filters_discr, input_shape=self.x_shape)
        for i in range(1, self.num_blocks):
            x = add_block(x, int(self.min_filters_discr * (self.strides**i)))

        x = GlobalAveragePooling2D()(x)

        # Minibatch discrimination to prevent modal collapse
        concat = add_minibatch_discrimination(x)

        concat = Dense(1)(concat)

        discriminator = Model(inputs=inp, outputs=concat, name="discriminator")

        discriminator.compile(loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(lr=self.disc_lr, beta_1=self.beta_1), metrics=["mae"])

        return discriminator

    def create_generator(self):

        def add_block(x, filters, activation=LeakyReLU, strides=self.strides):
            x = Conv2DTranspose(filters, self.kernal_size, strides=strides,
                padding="same", use_bias=False)(x)
            x = BatchNormalization()(x)
            x = self.add_activation(x, activation)
            return x

        inp = Input(shape=(self.latent_dims, ))

        # TODO - maybe change number of filters increasing expon. with stride.
        num_filters = int(self.min_filters_gen * (self.strides**(self.num_blocks-1)))

        disc_output_spatial_dim = [int(dim/self.strides**self.num_blocks) \
            for dim in self.x_shape[:-1]]

        # Double inner dimension for first 3D tensor (relative to the last
        # 3D tensor in the discriminator)
        input_vol = int(np.prod(disc_output_spatial_dim) * num_filters*2)

        x = Dense(input_vol)(inp)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        input_shape = (*disc_output_spatial_dim, num_filters*2)
        x = Reshape(target_shape=input_shape)(x)

        # Initial convolution with no change in shape
        x =  add_block(x, num_filters, strides=(1,1))

        for i in range(1, self.num_blocks):
            x = add_block(x, int(num_filters/(2**i)))

        # Add channels (e.g. RGB)
        x = Conv2DTranspose(self.x_shape[-1], self.kernal_size,
            padding="same", strides=self.strides, use_bias=False, activation="tanh")(x)

        return Model(inputs=inp, outputs=x, name="generator")

    def create_combined(self):
        combined = Sequential(name="combined")
        combined.add(self.generator)
        combined.add(self.discriminator)

        self.discriminator.trainable = False

        combined.compile(loss=BinaryCrossentropy(from_logits=True),
            optimizer=Adam(lr=self.gen_lr, beta_1=self.beta_1), metrics=['mae'])

        return combined

    def print_summary(self):
        self.generator.summary()
        self.discriminator.summary()
        self.combined.summary()

    def generate_imgs(self, return_imgs=False, cmap=None,
        postproc_func=None):
        num = 4
        fake_imgs = self.generator.predict(np.random.randn(num, self.latent_dims))

        if postproc_func is not None:
            assert isinstance(postproc_func, types.FunctionType)
            fake_imgs = postproc_func(fake_imgs)
            assert isinstance(fake_imgs, np.ndarray)
            assert len(fake_imgs) == 4

        fig = self.create_imgs_fig(fake_imgs, 2, 2, cmap=cmap)

        if return_imgs:
            return fig, fake_imgs
        else:
            return fig

    @staticmethod
    def create_imgs_fig(imgs, rows, columns, cmap=None):
        assert len(imgs) == int(rows*columns)
        if imgs.shape[-1] == 1:
            # flatten channels dimension if 1
            imgs = imgs[:,:,:,0]
        fig = plt.figure(figsize=(8, 8))
        for i, img in enumerate(imgs):
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img, cmap=cmap)
            plt.axis("off")
        return fig

    def train(self, real_train, num_epochs, batch_size,
            disc_updates=1, gen_updates=1, show_imgs=True, save_imgs=True,
            labels=(0, 1), cmap=None, total_real=None, progress_frac=1,
            postproc_func=None):
        """
        Arguments:
        - real_train - Either nparray of real samples.
        Shape (num_samples, spatial_dim, spatial_dim, channels)
        OR a batch generator. Must then provide total_real.
        - num_epochs - number of epochs to run (int)
        - batch_size - (int). Must match that of batch generator (if used).
        - disc_updates - number of batch updates to perform per step for the
        Discriminator before switching to the Generator (int)
        - gen_updates - number of batch updates to perform per step for the
        Generator before switching to the Discriminator (int)
        - show_imgs - generate and display progress images (once per epoch) (bool)
        - save_imgs - save images in run directory (bool)
        - labels - (negative label, positive label) e.g. (0, 0.9) for one-sided,
        positive label smoothing.
        - cmap - cmap for plt.imshow.
        - total_real - must provide if generator passed as real_train.
        - progress_frac - show/save progress images every {progress_frac} epoch.
        E.g. every 1 epoch, every 0.25 epochs etc.
        - postproc_func - postprocess images before display/save
        - epoch_start - manually enter an epoch start if cannot be retrieved

        Outputs:
        - Updates weights of GAN instance
        - Rewrites run history of GAN instance
        - Creates new directory to save history pickle and progress images
        - Displays progressbars and progress images
        """
        if not isinstance(real_train, types.GeneratorType):
            assert isinstance(real_train, np.ndarray)
            assert self.x_shape == real_train.shape[1:]
            total_real = len(real_train)
        else:
            # TODO - also check shape agreement if generator was passed
            # must provide num of total samples if generator was passed
            real_gen = real_train

        for i in (batch_size, disc_updates, gen_updates, total_real):
            assert type(i) is int
            assert i != 0

        spatial_dims = self.x_shape[:-1]

        runs_root_dir = "Training_Runs"
        if not os.path.isdir(runs_root_dir):
            os.mkdir(runs_root_dir)

        run_id = uuid4()
        run_dir = f"{runs_root_dir}/{str(run_id)}"
        os.mkdir(run_dir)

        # Save readable model architecture summary for later reference
        with open(f"{run_dir}/model_summary.txt", "w") as f:
            with redirect_stdout(f):
                self.print_summary()

        if self.history is None:
            self.history = History(run_id)
        else:
            assert isinstance(self.history, History)

        # TODO - Warn np array dataset is truncated to be divisible by batch size.
        steps = int(total_real//batch_size)

        progress_steps = int(progress_frac * steps)
        if progress_steps == 0:
            progress_steps += 1

        def array_to_gen(random_real_indxs, steps):
            """
            Nested funct to wrap np array into a batch generator
            """
            while True:
                for i in range(steps):
                    yield real_train[random_real_indxs[i]], np.zeros([batch_size, 1])+labels[1]

        for epoch in range(num_epochs):
            pbar = trange(steps)

            random_real_indxs = np.random.permutation(int(batch_size*steps))\
                                .reshape(steps, batch_size)

            if isinstance(real_train, np.ndarray):
                real_gen = array_to_gen(random_real_indxs, steps)

            for step in pbar:
                # Train Discriminator
                disc_loss = 0
                for _ in range(disc_updates):
                    # Generator may take time to initialise (e.g. ImageDataGenerator)
                    # Cannot consume the generator in model.fit - need to train on BATCH
                    disc_loss += 0.5 * self.discriminator.train_on_batch(*next(real_gen))[1]

                    random_seed = np.random.randn(batch_size, self.latent_dims)

                    disc_loss += 0.5 * self.discriminator.train_on_batch(self.generator.predict(random_seed),
                                                    np.zeros([batch_size, 1])+labels[0])[1]

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

                pbar.set_postfix({"epoch": self.history.epoch, "disc_loss": disc_loss, "gen_loss": gen_loss})

                # HACK - investigate why step isn't already an int
                if (int(step+1)/progress_steps).is_integer():
                    if (show_imgs or save_imgs):
                        fig = self.generate_imgs(postproc_func=postproc_func, cmap=cmap)

                        if save_imgs:
                            plt.savefig(f"{run_dir}/img_epoch{self.history.epoch}_step{step+1}.png")
                        if  show_imgs:
                            plt.show()
                        else:
                            plt.clf()

            self.history.epoch += 1

        pickle.dump(self.history, open(f"{run_dir}/history.p", "wb"))
