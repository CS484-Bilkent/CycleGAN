from models.discriminator import Discriminator
from config import DEVICE


def train(): ...


def main():
    disc_a = Discriminator(in_channels=3).to(DEVICE)  # RGB
    disc_b = Discriminator(in_channels=3).to(DEVICE)  # RGB


if __name__ == "__main__":
    main()
