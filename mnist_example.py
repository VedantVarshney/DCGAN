from gan import GAN

import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.backend import clear_session
from matplotlib import pyplot as plt
import numpy as np

import pickle

from skimage.transform import resize

from tqdm import tqdm, trange


BATCH_SIZE = 30
assert BATCH_SIZE % 2 == 0

HALF_BATCH = int(BATCH_SIZE/2)

NUM_EPOCHS = 5

# Number of training batches before alternating the trainable model.
DISC_UPDATES = 1
GEN_UPDATES = 1

SPATIAL_DIM = 32
LATENT_DIMS = 50

TOTAL_REAL = int(60e3)
assert TOTAL_REAL % BATCH_SIZE == 0

steps = int(TOTAL_REAL / BATCH_SIZE)


def preprocess_real():
    (real_train, _) , (real_test, _) = mnist.load_data()

    real_train = np.asarray([resize(img, (SPATIAL_DIM, SPATIAL_DIM)) for img in real_train])
    real_test = np.asarray([resize(img, (SPATIAL_DIM, SPATIAL_DIM)) for img in real_test])

    real_train = (np.expand_dims(real_train, axis=-1)/127.5 - 1.).astype("float32")
    real_test = (np.expand_dims(real_test, axis=-1)/127.5 - 1.).astype("float32")

    return real_train, real_test


def main():
    clear_session()

    # real_train, real_test = preprocess_real()
    # pickle.dump((real_train, real_test), open("mnist_train_tuple.p", "wb"))
    real_train, real_test = pickle.load(open("mnist_train_tuple.p", "rb"))

    gan = GAN(x_shape=real_train[0].shape, kernal_size=5,
        latent_dims=LATENT_DIMS)

    disc_loss_history = []
    gen_loss_history = []

    for epoch in range(NUM_EPOCHS):
        pbar = trange(steps)
        for step in pbar:

            # Train Discriminator
            disc_loss = 0
            for _ in range(DISC_UPDATES):
                random_seed = np.random.randn(HALF_BATCH, LATENT_DIMS)

                random_real_indxs = np.random.choice(TOTAL_REAL, HALF_BATCH)
                batch_data = np.concatenate((real_train[random_real_indxs],
                                            gan.generator.predict(random_seed)))

                discrim_labels = np.concatenate((np.ones([HALF_BATCH, 1]),
                                                np.zeros([HALF_BATCH, 1])))

                shuffle_indxs = np.random.permutation(BATCH_SIZE)
                discrim_labels = discrim_labels[shuffle_indxs]
                batch_x = batch_data[shuffle_indxs]

                disc_loss += gan.discriminator.train_on_batch(batch_x, discrim_labels)[1]

            # Train Generator
            gen_loss = 0
            for _ in range(GEN_UPDATES):
                # Create new images (separate from those used to train discriminator)
                random_seed = np.random.randn(BATCH_SIZE, LATENT_DIMS)
                gen_loss += gan.combined.train_on_batch(random_seed, np.ones([BATCH_SIZE, 1]))[1]

            disc_loss /= DISC_UPDATES
            gen_loss /= GEN_UPDATES
            disc_loss_history.append(disc_loss)
            gen_loss_history.append(gen_loss)

        pbar.set_postfix({"disc_loss": disc_loss, "gen_loss": gen_loss})


if __name__ == '__main__':
    main()
