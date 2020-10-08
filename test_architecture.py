from gan import GAN
import tensorflow as tf
from tensorflow.keras.backend import clear_session

import random
import numpy as np

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class TestModelOutputShapes(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.x_shape = (28, 28, 1)
        self.model = GAN(self.x_shape, kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)

    def tearDown(self):
        # TODO - dispose properly?
        del self.model
        clear_session()

    def test_model_shapes(self):
        model = GAN(x_shape=(28, 28, 1), kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)

        self.assertEqual((None, 28, 28, 1), model.generator.output_shape)
        self.assertEqual((None, 1), model.discriminator.output_shape)
        self.assertEqual((None, 1), model.combined.output_shape)
        self.assertEqual(model.generator.output_shape, model.discriminator.input_shape)


class TestModelTraining(tf.test.TestCase):

    def setUp(self):
        super().setUp()


if __name__ == '__main__':
    tf.test.main()
