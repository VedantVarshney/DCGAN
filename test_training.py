from gan import GAN
import tensorflow as tf
from tensorflow.keras.backend import clear_session

from copy import deepcopy

import random
import numpy as np

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


class TestModelTraining(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        clear_session()
        self.model = GAN(x_shape=(28, 28, 1), kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)

    def tearDown(self):
        del self.model
        clear_session()

    def test_discriminator_trains(self):
        """
        Test whether a train step updates all trainable variables for the discriminator.
        """

        # HACK - run 10 training steps to 'semi-ensure' all params change
        data = (np.ones(shape=(10, *self.model.x_shape)), np.ones(shape=(10, 1)))

        assert self.model.discriminator.trainable == False
        # Temporarily set as trainable to extract trainable variables
        self.model.discriminator.trainable = True
        trainable_before = deepcopy(self.model.discriminator.trainable_variables)
        self.model.discriminator.trainable = False

        assert len(trainable_before) != 0

        # model compilation has already occured so can still train with
        # trainable set to False
        _ = self.model.discriminator.train_on_batch(*data)

        self.model.discriminator.trainable = True

        trainable_after = self.model.discriminator.trainable_variables

        self.assertEqual(len(trainable_before), len(trainable_after))

        for b, a in zip(trainable_before, trainable_after):
            assert (b.numpy() != a.numpy()).any()

        self.model.discriminator.trainable = False

    def test_generator_trains(self):
        """
        Test whether a train step updates all trainable variables for the generator.
        """
        # HACK - run 10 training steps to semi-ensure all params change
        data = (np.ones(shape=(10, self.model.latent_dims)), np.ones(shape=(10, 1)))

        trainable_before = deepcopy(self.model.generator.trainable_variables)
        assert len(trainable_before) != 0

        self.model.combined.train_on_batch(*data)

        trainable_after = self.model.generator.trainable_variables

        self.assertEqual(len(trainable_before), len(trainable_after))

        for b, a in zip(trainable_before, trainable_after):
            assert (b.numpy() != a.numpy()).any()

if __name__ == '__main__':
    tf.test.main()
