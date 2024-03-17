import os
import argparse
import torch
from models.generator import Generator
from models.discriminator import Discriminator
from dataset.dataset import ABDataset
from dataset.dataset import img_transform as transform
from config.args import parser
from torch.utils.data import DataLoader
from util.image import save_combined_image



def load_latest_checkpoint(checkpoints_dir):
    checkpoints = [file for file in os.listdir(checkpoints_dir) if file.startswith("checkpoint_epoch")]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoints_dir, latest_checkpoint)

def main(args):
    checkpoint_path = args.checkpoint if args.checkpoint else load_latest_checkpoint("checkpoints")

    generator_A = Generator(img_channels=3, num_residuals=9).to(args.device)
    generator_B = Generator(img_channels=3, num_residuals=9).to(args.device)
    discriminator_A = Discriminator(in_channels=3).to(args.device) 
    discriminator_B = Discriminator(in_channels=3).to(args.device) 

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        generator_A.load_state_dict(checkpoint["generator_A_state_dict"])
        generator_B.load_state_dict(checkpoint["generator_B_state_dict"])
        discriminator_A.load_state_dict(checkpoint["discriminator_A_state_dict"])
        discriminator_B.load_state_dict(checkpoint["discriminator_B_state_dict"])

    else:
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    generator_A.eval()
    generator_B.eval()
    discriminator_A.eval()
    discriminator_B.eval()

    dataset = ABDataset(root_a=args.test_dir + "/testA", root_b=args.test_dir + "/testB", transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i,batch in enumerate(data_loader):
        real_a, real_b = batch['A'], batch['B']

        fake_b = generator_B(real_a)
        fake_a = generator_A(real_b)
        save_combined_image(fake_b, real_a, fake_a, real_b, 0, i, "test")

        if args.test_limit and i > args.test_limit:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform image translation using CycleGAN.")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()
    main(args)
