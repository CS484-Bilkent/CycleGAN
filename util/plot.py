import matplotlib.pyplot as plt


def plot_loss(disc_losses, gen_losses, file_name, args=None):

    time_steps = range(1, len(disc_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(time_steps, disc_losses, label="Discriminator Loss", color="blue", marker="o")
    plt.plot(time_steps, gen_losses, label="Generator Loss", color="red", marker="x")

    plt.title("Discriminator and Generator Loss Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.grid(True)

    plt.savefig(f"results/{args.run_name}_{file_name}.png")
