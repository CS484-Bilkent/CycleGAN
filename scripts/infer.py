import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from config.args import parser
from util.log import log
from models.generator import Generator
from models.discriminator import Discriminator
from dataset.dataset import ABDataset
from dataset.dataset import img_transform as transform
from config.args import parser
from torch.utils.data import DataLoader
from util.image import *
from tqdm import tqdm


def load_latest_checkpoint(checkpoints_dir, checkpoint_name=None):

    if checkpoint_name:
        return os.path.join(checkpoints_dir, checkpoint_name)

    checkpoints = [file for file in os.listdir(checkpoints_dir) if file.startswith("checkpoint_epoch")]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))

    return os.path.join(checkpoints_dir, latest_checkpoint)


def main(args):
    checkpoint_path = load_latest_checkpoint(args.checkpoints_dir, args.checkpoint)

    generator_A = Generator(img_channels=3, num_residuals=9).to(args.device)
    generator_B = Generator(img_channels=3, num_residuals=9).to(args.device)
    discriminator_A = Discriminator(in_channels=3).to(args.device)
    discriminator_B = Discriminator(in_channels=3).to(args.device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)

        generator_A.load_state_dict(checkpoint["gen_a_state_dict"])
        generator_B.load_state_dict(checkpoint["gen_b_state_dict"])
        discriminator_A.load_state_dict(checkpoint["disc_a_state_dict"])
        discriminator_B.load_state_dict(checkpoint["disc_b_state_dict"])

    else:
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    generator_A.eval()
    generator_B.eval()
    discriminator_A.eval()
    discriminator_B.eval()

    dataset = ABDataset(root_a=args.test_dir + "/testA", root_b=args.test_dir + "/testB", transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    tqdm_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    for i, (real_a, real_b) in tqdm_loop:
        real_a = real_a.to(args.device)
        real_b = real_b.to(args.device)

        # generate fake ones
        fake_b = generator_A(real_a)  # A to B
        fake_a = generator_B(real_b)  # B to A

        # regenerate for cycle consistency
        rec_a = generator_B(fake_b)  # Fake B to A
        rec_b = generator_A(fake_a)  # Fake A to B

        save_cycle_consistent_images(real_a, fake_b, rec_a, real_b, fake_a, rec_b, 0, i, f"test/{args.run_name}")

        if args.test_limit and i >= args.test_limit:
            break


if __name__ == "__main__":
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--test-limit", type=int, required=False)

    args = parser.parse_args()

    os.makedirs(f"runs/test/{args.run_name}", exist_ok=True)
    log("Using args:", args)

    main(args)
