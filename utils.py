import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pathlib import Path

def save_checkpoint(generator, discriminator, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), out_dir / "generator.pt")
    torch.save(discriminator.state_dict(), out_dir / "discriminator.pt")


def plot_losses(gen_losses, disc_losses):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses)
    plt.title("Generator loss")
    plt.subplot(1, 2, 2)
    plt.plot(disc_losses)
    plt.title("Discriminator loss")
    plt.tight_layout()
    plt.show()


def show_fake_grid(tensor_batch, nrow: int = 8):
    grid = vutils.make_grid(tensor_batch, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
