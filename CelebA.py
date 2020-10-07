import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob

IMGS_DIR = "../CelebFaces_A_Dataset/img_align_celeba"

def gen_real_img_batch(imgs_dir, target_size=(28, 28), batch_size=32, positive_label=1,
    channels=3):
    img_gen = image.ImageDataGenerator(rescale=1/255.)

    img_flow = img_gen.flow_from_directory(imgs_dir,
                class_mode="binary",
                target_size=target_size,
                batch_size=batch_size)

    img_flow.classes = np.zeros(len(img_flow.classes)) + positive_label
    img_flow.class_indices = {"real", positive_label}

    while True:
        yield next(img_flow)

    # TODO - crop image! Wrap next(img_flow) in a custom generator.

def main(imgs_dir):
    gen = gen_real_img_batch(imgs_dir)
    return gen

if __name__ == '__main__':
    main()
