import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importing the required libraries

from models.discriminator import Discriminator
from models.generator import Generator
from dataset.dataset import ABDataset
from dataset.dataset import inv_normalize
from config.args import parser
from util.log import log
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import optim
from torchvision.utils import save_image


def train(disc_a, disc_b, gen_a, gen_b, opt_disc_a, opt_disc_b, opt_gen_a, opt_gen_b, L1, MSE, data_loader, args):
    tqdm_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    for i, (real_a, real_b) in tqdm_loop:
        real_a = real_a.to(args.device)
        real_b = real_b.to(args.device)

        # Discriminators
        fake_b = gen_a(real_a)
        disc_a_real = disc_a(real_a)
        disc_a_fake = disc_a(fake_b.detach())

        disc_b_loss = MSE(disc_a_real, torch.ones_like(disc_a_real)) + MSE(disc_a_fake, torch.zeros_like(disc_a_fake))

        fake_a = gen_b(real_b)
        disc_b_real = disc_a(real_b)
        disc_b_fake = disc_a(fake_a.detach())

        disc_a_loss = MSE(disc_b_real, torch.ones_like(disc_b_real)) + MSE(disc_b_fake, torch.zeros_like(disc_b_fake))

        disc_loss = disc_a_loss + disc_b_loss  # total loss here

        # Usual stuff
        opt_disc_a.zero_grad()
        opt_disc_b.zero_grad()
        disc_loss.backward()
        opt_disc_a.step()
        opt_disc_b.step()

        # Generators
        # Adversarial Loss
        disc_a_fake = disc_a(fake_a)
        disc_b_fake = disc_b(fake_b)
        generator_loss_a = MSE(disc_a_fake, torch.ones_like(disc_a_fake))
        generator_loss_b = MSE(disc_b_fake, torch.ones_like(disc_b_fake))

        # Cycle Loss
        cycle_a = gen_a(fake_b)
        cycle_b = gen_b(fake_a)
        cycle_a_loss = L1(real_a, cycle_a)
        cycle_b_loss = L1(real_b, cycle_b)

        # Identity Loss
        identity_a = gen_a(real_a)
        identity_b = gen_b(real_b)
        identity_a_loss = L1(real_a, identity_a)
        identity_b_loss = L1(real_b, identity_b)

        gen_loss = (
            generator_loss_a
            + generator_loss_b
            + cycle_a_loss * args.lambda_cycle
            + cycle_b_loss * args.lambda_cycle
            + identity_a_loss * args.lambda_identity
            + identity_b_loss * args.lambda_identity
        )

        # Usual stuff
        opt_gen_a.zero_grad()
        opt_gen_b.zero_grad()
        gen_loss.backward()
        opt_gen_a.step()
        opt_gen_b.step()

        if i % 200 == 0:
            save_image(inv_normalize(fake_a), f"runs/{args.run_name}/saved_images/fake_a_{i}.png")
            save_image(inv_normalize(fake_b), f"runs/{args.run_name}/saved_images/fake_b_{i}.png")


def main(args):
    disc_a = Discriminator(in_channels=3).to(args.device)  # RGB
    disc_b = Discriminator(in_channels=3).to(args.device)  # RGB

    gen_a = Generator(in_channels=3, num_residuals=9).to(args.device)  # RGB
    gen_b = Generator(in_channels=3, num_residuals=9).to(args.device)  # RGB

    opt_disc_a = optim.Adam(
        disc_a.parameters(),
        lr=args.learning_rate,
        # betas=(0.5, 0.999),
    )

    opt_disc_b = optim.Adam(
        disc_b.parameters(),
        lr=args.learning_rate,
        # betas=(0.5, 0.999),
    )

    opt_gen_a = optim.Adam(
        gen_a.parameters(),
        lr=args.learning_rate,
        # betas=(0.5, 0.999),
    )

    opt_gen_b = optim.Adam(
        gen_b.parameters(),
        lr=args.learning_rate,
        # betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    # Usual stuff
    dataset = ABDataset(root_a=args.train_dir + "/trainA", root_b=args.train_dir + "/trainB")
    data_loader = dataset.loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.load_checkpoints:
        # ...
        log("Loading Checkpoints...")

    for epoch in range(args.num_epochs):
        train(disc_a, disc_b, gen_a, gen_b, opt_disc_a, opt_disc_b, opt_gen_a, opt_gen_b, L1, MSE, data_loader, args)

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            # ...
            log("Saving Checkpoints...")


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
