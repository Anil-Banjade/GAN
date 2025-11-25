import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from models import Generator


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator = Generator(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    noise = torch.randn(args.num_samples, args.latent_dim, device=device)
    with torch.no_grad():
        samples = generator(noise).cpu()

    fig, axes = plt.subplots(1, args.num_samples, figsize=(args.num_samples * 1.5, 2))
    for idx, ax in enumerate(axes):
        ax.imshow(samples[idx, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./artifacts/generator.pt")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
