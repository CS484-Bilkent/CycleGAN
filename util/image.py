import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def save_combined_image(real_a_to_b, real_a, real_b_to_a, real_b, epoch, i, run_name):
    # Select the first image in the batch for each tensor
    real_a_to_b = real_a_to_b[0]
    real_a = real_a[0]
    real_b_to_a = real_b_to_a[0]
    real_b = real_b[0]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()

    images = [real_a_to_b, real_a, real_b_to_a, real_b]
    titles = ["A -> B", "Real A", "B -> A", "Real B"]

    for ax, img, title in zip(axs, images, titles):
        img = img.detach().cpu().numpy()  # Move to cpu
        img = ((img - img.min()) / (img.max() - img.min())).astype(
            np.float32
        )  # Normalize to [0, 1] and ensure dtype is float
        img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC for matplotlib

        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"runs/{run_name}/epoch_{epoch}_combined_{i}.png")
    plt.close(fig)


def save_cycle_consistent_images(real_a, fake_b, rec_a, real_b, fake_a, rec_b, epoch, i, run_name):
    # Select the first image in the batch for each tensor
    real_a = real_a[0]
    fake_b = fake_b[0]
    rec_a = rec_a[0]
    real_b = real_b[0]
    fake_a = fake_a[0]
    rec_b = rec_b[0]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    # images = [real_a, real_a_to_b, real_a_to_b_to_a, real_b, real_b_to_a, real_b_to_a_to_b]
    images = [real_a, fake_b, rec_a, real_b, fake_a, rec_b]
    titles = ["Real A", "A -> B", "A -> B -> A", "Real B", "B -> A", "B -> A -> B"]

    for ax, img, title in zip(axs, images, titles):
        img = img.detach().cpu().numpy()  # Move to CPU
        img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)  # Normalize to [0, 1]
        img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC for matplotlib

        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"runs/{run_name}/epoch_{epoch}_cycle_consistent_{i}.png")
    plt.close(fig)
