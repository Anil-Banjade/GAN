import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import build_dataloader
from models import Generator, Discriminator
from utils import plot_losses, show_fake_grid, save_checkpoint


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataloader = build_dataloader(batch_size=args.batch_size, data_root=args.data_root)
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for images, _ in progress:
            images = images.to(device)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(images)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            progress.set_postfix({"d_loss": d_loss.item(), "g_loss": g_loss.item()})

        g_losses.append(g_epoch_loss / len(dataloader))
        d_losses.append(d_epoch_loss / len(dataloader))

        if (epoch + 1) % args.sample_every == 0:
            show_fake_grid(fake_images[: min(16, fake_images.size(0))].cpu())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_checkpoint(generator, discriminator, args.output_dir)
    plot_losses(g_losses, d_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    train(args)
