from gan import GAN
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import os

import random
import numpy as np

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class TestModelSaveLoad(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.x_shape = (28, 28, 1)
        self.model = GAN(self.x_shape, kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)
        self.gan_fpath = "test_gan.p"
        self.model_dir = "test_model_dir"

        assert not os.path.isfile(self.gan_fpath)
        assert not os.path.isdir(self.model_dir)

    def tearDown(self):
        # TODO - dispose properly?
        del self.model
        os.remove(self.gan_fpath)
        os.rmdir(self.model_dir)
        clear_session()

    def test_model_save(self):
        discr_x = np.ones(shape=(1, *self.model.x_shape))
        comb_x = np.ones(shape=(1, self.latent_dims))
        y = np.ones(shape=(1, 1)))

        self.model.discriminator.train_on_batch(discr_x, y)
        self.model.combined.train_on_batch(comb_x, y)
        self.model.save(self.gan_fpath, model_dir=self.model_dir)

        model2 = GAN.load(self.gan_fpath, model_dir=self.model_dir)

        assert os.path.isfile(self.gan_fpath)
        assert os.path.isdir(self.model_dir)

        # TODO - finish!
