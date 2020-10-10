from gan import GAN
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import os, shutil

import random
import numpy as np
import os

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class TestModelSaveLoad(tf.test.TestCase):

    def setUp(self):
        super().setUp()

        self.gan_fpath = "test_gan.p"
        self.model_dir = "test_model_dir"
        clear_session()

        assert not os.path.isfile(self.gan_fpath)
        assert not os.path.isdir(self.model_dir)

        self.x_shape = (28, 28, 1)
        self.model = GAN(self.x_shape, kernal_size=5, verbose=False,
            num_blocks=2, latent_dims=100, strides=2)

    def tearDown(self):
        # TODO - dispose properly?
        del self.model
        if os.path.isfile(self.gan_fpath):
            os.remove(self.gan_fpath)
        if os.path.isdir(self.model_dir):
            try:
                shutil.rmtree(self.model_dir)
            except OSError as e:
                print(f"Error deleting {dir_path}: {e.strerror}")

    def test_model_save(self):
        # HACK - run 10 training steps to semi-ensure all params change
        discr_x = np.ones(shape=(10, *self.model.x_shape))
        comb_x = np.ones(shape=(10, self.model.latent_dims))
        y = np.ones(shape=(10, 1))

        assert self.model.discriminator.trainable == False

        self.model.discriminator.train_on_batch(discr_x, y)
        self.model.combined.train_on_batch(comb_x, y)

        self.model.save(self.gan_fpath, model_dir=self.model_dir)

        # Check files were saved
        assert os.path.isfile(self.gan_fpath)
        assert os.path.isdir(self.model_dir)

        # Load saved model
        model2 = GAN.load(self.gan_fpath, model_dir=self.model_dir, verbose=False)

        assert model2.discriminator.trainable == False

        self.model.discriminator.trainable = True
        model2.discriminator.trainable = True

        assert len(self.model.discriminator.trainable_variables) > 0

        # Check discriminator variables are the same
        for (a, b) in zip(self.model.discriminator.trainable_variables, model2.discriminator.trainable_variables):
            assert (a.numpy() == b.numpy()).any()

        self.model.discriminator.trainable = False
        model2.discriminator.trainable = False

        assert len(self.model.generator.trainable_variables) > 0

        # Check all variables are the same
        for (a, b) in zip(self.model.generator.trainable_variables, model2.generator.trainable_variables):
            # Convert eager tensor to numpy
            assert (a.numpy() == b.numpy()).any()

        remove_params = ["discriminator", "generator", "combined"]
        # TODO - check this is a deep copy?
        model_dicts = (self.model.__dict__.copy(), model2.__dict__.copy())

        # Check two saved and loaded instance have same attributes
        self.assertEqual(*[list(dic.keys()) for dic in model_dicts])

        # Get attribute names (not incl. models)
        keys = [k for k in list(model_dicts[0].keys()) if k not in remove_params]

        # Check all non-model attributes are the same for saved and loaded model
        self.assertEqual(*[[model[key] for key in keys] for model in model_dicts])


if __name__ == '__main__':
    tf.test.main()
