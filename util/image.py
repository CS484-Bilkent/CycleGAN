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
