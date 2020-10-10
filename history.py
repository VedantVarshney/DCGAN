from matplotlib import pyplot as plt

class History:
    def __init__(self, run_id, epoch=1, disc_loss=[], gen_loss=[]):
        self.run_id = run_id
        self.epoch = epoch
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

    def plot_loss(self):
        plt.title(f"Run {str(self.run_id)}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.plot(self.disc_loss, label="Discriminator", alpha=0.8)
        plt.plot(self.gen_loss, label="Generator", alpha=0.8)
        plt.legend()

        plt.show()
