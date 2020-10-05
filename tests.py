from gan import GAN


def test_model_architectures():
    """
    Inspect model summaries.
    Checklist:
        - All layers present
        - Generator reshape layer has the output shape as the last 3D layer
        in discriminator ( (None, 2, 2, 128) in this case).
        - Generator output has same shape as discriminator input
        - Discriminator has no trainable params
    """
    gan = GAN(x_shape=(28, 28, 1), kernal_size=5, verbose=True, num_blocks=2,
            strides=2)

if __name__ == '__main__':
    test_model_architectures()
