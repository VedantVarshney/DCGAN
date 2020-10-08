from gan import GAN
import tensorflow as tf
from tensorflow.keras.backend import clear_session

import random
import numpy as np

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class TestModelTraining(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.model = GAN(x_shape=(28, 28, 1), kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)

    def tearDown(self):
        del self.model
        clear_session()

    def test_discriminator_trains(self):
        """
        Test whether a train step updates all trainable variables for both models.
        """

        discr_data = (np.ones(shape=(1, *self.model.x_shape)), np.ones(shape=(1, 1)))

        trainable_before = self.model.discriminator.trainable_variables
        self.model.discriminator.train_on_batch(*discr_data)
        self.assertEqual(len(trainable_before), len(self.model.discriminator.trainable_variables))

        for b, a in zip(trainable_before, self.model.discriminator.trainable_variables):
            assert (b != a).any()

    # TODO - test generator variables train (through combined model)

if __name__ == '__main__':
    tf.test.main()
