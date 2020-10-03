
from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np

(x_train, y_train), (x_test, y_test) = load_data()

x_train = np.asarray([resize(img, (32, 32)) for img in x_train])

print(x_train.shape)

a = resize(x_train[0], (32, 32))

plt.imshow(a)
plt.show()

import pickle
real_train, real_test = pickle.load(open("mnist_train_tuple.p", "rb"))

plt.imshow(real_train[np.random.choice(len(real_train), 1)].reshape((32,32)))
