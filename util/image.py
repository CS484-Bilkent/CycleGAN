import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def save_combined_image(fake_a, real_a, fake_b, real_b, epoch, i, run_name):
    # Select the first image in the batch for each tensor
    fake_a = fake_a[0] * 0.5 + 0.5
    real_a = real_a[0] * 0.5 + 0.5
    fake_b = fake_b[0] * 0.5 + 0.5
    real_b = real_b[0] * 0.5 + 0.5

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()

    images = [fake_a, real_a, fake_b, real_b]
    titles = ["Fake A", "Real A", "Fake B", "Real B"]

    for ax, img, title in zip(axs, images, titles):
        img = img.detach().cpu().numpy()  # Move to cpu
        # img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC for matplotlib

        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"runs/{run_name}/epoch_{epoch}_combined_{i}.png")
    plt.close(fig)
