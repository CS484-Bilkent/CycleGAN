import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importing the required libraries

from models.discriminator import Discriminator
from models.generator import Generator
from dataset.dataset import ABDataset
from dataset.dataset import inv_normalize
from dataset.dataset import img_transform as transform
from config.args import parser
from util.log import log
from util.image import save_combined_image
from util.plot import plot_loss
from tqdm import tqdm
import itertools
from torch.utils.data import DataLoader
from collections import deque

import torch
import torch.nn as nn
from torch import optim


def main(args):
    disc_a = Discriminator(in_channels=3).to(args.device)  # RGB
    disc_b = Discriminator(in_channels=3).to(args.device)  # RGB

    gen_a = Generator(img_channels=3, num_residuals=9).to(args.device)  # RGB
    gen_b = Generator(img_channels=3, num_residuals=9).to(args.device)  # RGB

    optimizer_G = torch.optim.Adam(
        itertools.chain(gen_a.parameters(), gen_b.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(disc_a.parameters(), disc_b.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    # Taken from their code
    G_scaler = torch.cuda.amp.GradScaler()
    D_scaler = torch.cuda.amp.GradScaler()

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    # Usual stuff
    dataset = ABDataset(root_a=args.train_dir + "/trainA", root_b=args.train_dir + "/trainB", transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # maybe we can add an argument to specify checkpoint idk
    if args.load_checkpoints:
        checkpoint_dir = "checkpoints"

        latest_checkpoint_path = max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")],
            key=os.path.getctime,
        )

        if os.path.exists(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path)

            disc_a.load_state_dict(checkpoint["disc_a_state_dict"])
            disc_b.load_state_dict(checkpoint["disc_b_state_dict"])
            gen_a.load_state_dict(checkpoint["gen_a_state_dict"])
            gen_b.load_state_dict(checkpoint["gen_b_state_dict"])

        log("Loading Checkpoints - ", latest_checkpoint_path)

    disc_losses = deque(maxlen=1000)
    gen_losses = deque(maxlen=1000)
    for epoch in range(args.num_epochs):
        log("epoch", epoch + 1, "/", args.num_epochs)

        tqdm_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

        for i, (real_a, real_b) in tqdm_loop:
            real_a = real_a.to(args.device)
            real_b = real_b.to(args.device)

            with torch.cuda.amp.autocast():  # F16 Training?
                # Discriminator B
                fake_b = gen_b(real_a)
                disc_b_real = disc_b(real_b)
                disc_b_fake = disc_b(fake_b.detach())

                disc_b_real_loss = MSE(disc_b_real, torch.ones_like(disc_b_real))
                disc_b_fake_loss = MSE(disc_b_fake, torch.zeros_like(disc_b_fake))

                disc_b_loss = disc_b_real_loss + disc_b_fake_loss

                # Discriminator A
                fake_a = gen_a(real_b)
                disc_a_real = disc_a(real_a)
                disc_a_fake = disc_a(fake_a.detach())

                disc_a_real_loss = MSE(disc_a_real, torch.ones_like(disc_a_real))
                disc_a_fake_loss = MSE(disc_a_fake, torch.zeros_like(disc_a_fake))

                disc_a_loss = disc_a_real_loss + disc_a_fake_loss

                disc_loss = (
                    disc_a_loss + disc_b_loss
                ) / 2  # total loss here (paper mentions /2, so I just use it). Though in theory, it should give the same result without /2.

                disc_losses.append(disc_loss.item())

                optimizer_D.zero_grad()
                D_scaler.scale(disc_loss).backward()
                D_scaler.step(optimizer_D)
                D_scaler.update()

            with torch.cuda.amp.autocast():
                # Generators

                # Adversarial Loss
                disc_b_fake = disc_b(fake_b)
                disc_a_fake = disc_a(fake_a)
                gen_loss_a = MSE(disc_a_fake, torch.ones_like(disc_a_fake))
                gen_loss_b = MSE(disc_b_fake, torch.ones_like(disc_b_fake))

                # Cycle Loss
                cycle_a = gen_a(fake_b)
                cycle_b = gen_b(fake_a)
                cycle_loss_a = L1(real_a, cycle_a)
                cycle_loss_b = L1(real_b, cycle_b)

                # Identity Loss
                identity_a = gen_a(real_a)
                identity_b = gen_b(real_b)
                identity_loss_a = L1(real_a, identity_a)
                identity_loss_b = L1(real_b, identity_b)

                gen_loss = (
                    gen_loss_a
                    + gen_loss_b
                    + cycle_loss_a * args.lambda_cycle
                    + cycle_loss_b * args.lambda_cycle
                    + identity_loss_a * args.lambda_identity
                    + identity_loss_b * args.lambda_identity
                )

                gen_losses.append(gen_loss.item())

                # Usual stuff
                optimizer_G.zero_grad()
                G_scaler.scale(gen_loss).backward()
                G_scaler.step(optimizer_G)
                G_scaler.update()

            if i % 100 == 0:
                save_combined_image(gen_b(real_a), real_a, gen_a(real_b), real_b, epoch, i, args.run_name)
                plot_loss(disc_losses, gen_losses, f"epoch_{epoch}_i_{i}", args)

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            checkpoint_dir = "checkpoints"

            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            checkpoint = {
                "disc_a_state_dict": disc_a.state_dict(),
                "disc_b_state_dict": disc_b.state_dict(),
                "gen_a_state_dict": gen_a.state_dict(),
                "gen_b_state_dict": gen_b.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)
            log(f"Saving Checkpoint at epoch {epoch + 1}: {checkpoint_path}")


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()
    log("Using args:", args)
    os.makedirs(f"runs/{args.run_name}", exist_ok=False)
    os.makedirs(f"results/{args.run_name}", exist_ok=False)
    main(args)
