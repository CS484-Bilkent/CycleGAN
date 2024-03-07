# Try out person riding a horse dataset if we can find any. Currently that cannot be converted into zebras.
#
# Some failure cases are caused by the distribution charac- teristics of the training datasets.
# For example, our method has got confused in the horse → zebra example (Figure 17, right),
# because our model was trained on the wild horse and zebra synsets of ImageNet,
# which does not contain images of a person riding a horse or zebra.

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

# The images were scaled to 256 × 256 pixels.
# The training set size of each class: 939 (horse), 1177 (zebra), 996 (apple), and 1020 (orange).

transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),  # usual values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # imagenet values
        transforms.ToTensor(),
    ]
)

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


class ABDataset(Dataset):
    def __init__(self, root_a, root_b, transform=None) -> None:
        super().__init__()
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        self.root_a_images = os.listdir(root_a)
        self.root_b_images = os.listdir(root_b)

        # We discarded the rest, there is no mention of this in the paper.
        self.length_dataset = min(len(self.root_a_images), len(self.root_b_images))

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_a = self.root_a_images[index]
        image_b = self.root_b_images[index]

        image_a = os.path.join(self.root_a, image_a)
        image_b = os.path.join(self.root_b, image_b)

        image_a = Image.open(image_a).convert("RGB")
        image_b = Image.open(image_b).convert("RGB")

        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        return image_a, image_b
