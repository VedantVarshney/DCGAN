from gan import GAN

import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.backend import clear_session
from matplotlib import pyplot as plt
import numpy as np

import pickle

from skimage.transform import resize

from tqdm.auto import tqdm, trange


BATCH_SIZE = 30
assert BATCH_SIZE % 2 == 0

HALF_BATCH = int(BATCH_SIZE/2)

NUM_EPOCHS = 20

# Number of training batches before alternating the trainable model.
DISC_UPDATES = 1
GEN_UPDATES = 1

SPATIAL_DIM = 32
LATENT_DIMS = 50

TOTAL_REAL = int(60e3)
assert TOTAL_REAL % BATCH_SIZE == 0

steps = int(TOTAL_REAL / BATCH_SIZE)


def preprocess_real(save_only=False):
    (real_train, _) , (real_test, _) = mnist.load_data()

    real_train = np.asarray([resize(img, (SPATIAL_DIM, SPATIAL_DIM)) for img in real_train])
    real_test = np.asarray([resize(img, (SPATIAL_DIM, SPATIAL_DIM)) for img in real_test])

    real_train = (np.expand_dims(real_train, axis=-1)/127.5 - 1.).astype("float32")
    real_test = (np.expand_dims(real_test, axis=-1)/127.5 - 1.).astype("float32")

    if save_only:
        pickle.dump((real_train, real_test), open("mnist_train_tuple.p", "wb"))
    else:
        return real_train, real_test


def main(verbose=True):
    clear_session()

    # real_train, real_test = preprocess_real()
    # pickle.dump((real_train, real_test), open("mnist_train_tuple.p", "wb"))
    real_train, real_test = pickle.load(open("mnist_train_tuple.p", "rb"))

    gan = GAN(x_shape=real_train[0].shape, kernal_size=5,
        latent_dims=LATENT_DIMS, verbose=verbose)

    gan.train(real_train, 1, BATCH_SIZE)

    return gan

if __name__ == '__main__':
    main()
