from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

class History:
    def __init__(self, run_id, disc_loss=[], gen_loss=[]):
        self.run_id = run_id
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

    def plot_loss(self):
        plt.title(f"Run {str(self.run_id)}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.plot(self.disc_loss, label="Discriminator")
        plt.plot(self.gen_loss, label="Generator")
        plt.show()
