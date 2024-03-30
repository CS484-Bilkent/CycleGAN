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


def linear_decay(epoch):
    if epoch < 100:
        return 1
    elif epoch < 200:
        return 1 - (epoch - 100) / 100
    else:
        print("Kill the training, weights are not updated after 200 epochs.")
        return 0  # Keep the learning rate at 0 after 200 epochs


"""
Horse => A
Zebra => B
"""


def main(args):
    disc_B = Discriminator(in_channels=3).to(args.device)
    disc_A = Discriminator(in_channels=3).to(args.device)

    gen_B = Generator(img_channels=3, num_residuals=9).to(args.device)
    gen_A = Generator(img_channels=3, num_residuals=9).to(args.device)

    opt_disc_A = optim.Adam(
        disc_A.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    opt_disc_B = optim.Adam(
        disc_B.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        itertools.chain(gen_A.parameters(), gen_B.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    # Taken from their code
    g_scaler = torch.cuda.amp.GradScaler()
    d_A_scaler = torch.cuda.amp.GradScaler()
    d_B_scaler = torch.cuda.amp.GradScaler()

    # Update LR
    g_scheduler = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: linear_decay(epoch) * args.learning_rate)
    d_A_scheduler = optim.lr_scheduler.LambdaLR(
        opt_disc_A, lr_lambda=lambda epoch: linear_decay(epoch) * args.learning_rate
    )
    d_B_scheduler = optim.lr_scheduler.LambdaLR(
        opt_disc_B, lr_lambda=lambda epoch: linear_decay(epoch) * args.learning_rate
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    # Usual stuff
    dataset = ABDataset(root_a=args.train_dir + "/trainA", root_b=args.train_dir + "/trainB", transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # maybe we can add an argument to specify checkpoint idk
    if args.load_checkpoints:
        checkpoint_dir = args.load_checkpoints_path

        latest_checkpoint_path = max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")],
            key=os.path.getctime,
        )

        if os.path.exists(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path)

            disc_A.load_state_dict(checkpoint["disc_a_state_dict"])
            disc_B.load_state_dict(checkpoint["disc_b_state_dict"])
            gen_A.load_state_dict(checkpoint["gen_a_state_dict"])
            gen_B.load_state_dict(checkpoint["gen_b_state_dict"])

        log("Loading Checkpoints - ", latest_checkpoint_path)

    disc_losses = deque(maxlen=1000)
    gen_losses = deque(maxlen=1000)

    for epoch in range(args.num_epochs):
        log("epoch", epoch + 1, "/", args.num_epochs)

        tqdm_loop = tqdm(data_loader, total=len(data_loader), leave=False)

        for idx, (A, B) in enumerate(tqdm_loop):
            B = B.to(args.device)
            A = A.to(args.device)

            # Train Discriminators H and Z
            with torch.cuda.amp.autocast():
                fake_A = gen_A(B)
                D_A_real = disc_A(A)
                D_A_fake = disc_A(fake_A.detach())
                D_A_real_loss = MSE(D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = MSE(D_A_fake, torch.zeros_like(D_A_fake))
                D_A_loss = D_A_real_loss + D_A_fake_loss

                fake_B = gen_B(A)
                D_B_real = disc_B(B)
                D_B_fake = disc_B(fake_B.detach())
                D_B_real_loss = MSE(D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = MSE(D_B_fake, torch.zeros_like(D_B_fake))
                D_B_loss = D_B_real_loss + D_B_fake_loss

                # put it togethor
                D_loss = (D_A_loss + D_B_loss) / 2

                disc_losses.append(D_loss.item())

            opt_disc_A.zero_grad()
            d_A_scaler.scale(D_A_loss).backward()
            d_A_scaler.step(opt_disc_A)
            d_A_scaler.update()

            opt_disc_B.zero_grad()
            d_B_scaler.scale(D_B_loss).backward()
            d_B_scaler.step(opt_disc_B)
            d_B_scaler.update()

            # Train Generators H and Z
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_A_fake = disc_A(fake_A)
                D_Z_fake = disc_B(fake_B)
                loss_G_A = MSE(D_A_fake, torch.ones_like(D_A_fake))
                loss_G_Z = MSE(D_Z_fake, torch.ones_like(D_Z_fake))

                # cycle loss
                cycle_B = gen_B(fake_A)
                cycle_A = gen_A(fake_B)
                cycle_B_loss = L1(B, cycle_B)
                cycle_A_loss = L1(A, cycle_A)

                identity_B = gen_B(B)
                identity_A = gen_A(A)
                identity_B_loss = L1(B, identity_B)
                identity_A_loss = L1(A, identity_A)

                # add all togethor
                G_loss = (
                    loss_G_Z
                    + loss_G_A
                    + cycle_B_loss * args.lambda_cycle
                    + cycle_A_loss * args.lambda_cycle
                    + identity_A_loss * args.lambda_identity
                    + identity_B_loss * args.lambda_identity
                )

                gen_losses.append(G_loss.item())

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 500 == 0 or idx == len(tqdm_loop) - 1:
                save_combined_image(
                    fake_A * 0.5 + 0.5,  # Denormalize
                    B * 0.5 + 0.5,  # Denormalize
                    fake_B * 0.5 + 0.5,  # Denormalize
                    A * 0.5 + 0.5,  # Denormalize
                    epoch,
                    idx,
                    args.run_name,
                )
                plot_loss(disc_losses, gen_losses, f"epoch_{epoch}_i_{idx}", args)

        d_A_scheduler.step()  # Update the learning rate after optimizer update
        d_B_scheduler.step()  # Update the learning rate after optimizer update
        g_scheduler.step()  # Update the learning rate after optimizer update

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:

            checkpoint_path = os.path.join(f"checkpoints/{args.run_name}", f"checkpoint_epoch_{epoch + 1}.pth")
            checkpoint = {
                "disc_a_state_dict": disc_A.state_dict(),
                "disc_b_state_dict": disc_B.state_dict(),
                "gen_a_state_dict": gen_A.state_dict(),
                "gen_b_state_dict": gen_B.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)
            log(f"Saving Checkpoint at epoch {epoch + 1}: {checkpoint_path}")


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()
    log("Using args:", args)
    os.makedirs(f"runs/{args.run_name}", exist_ok=False)
    os.makedirs(f"results/{args.run_name}", exist_ok=False)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    main(args)
