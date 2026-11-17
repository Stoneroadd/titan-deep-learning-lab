"""Run inference on a reproducible synthetic example."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import SurfaceCNN
from synthetic_data import LABELS, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--index", type=int, default=0, help="Index within a regenerated synthetic sample set")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; run train.py first")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    if tuple(labels) != LABELS:
        parser.error("checkpoint class labels do not match this version")
    images, targets = generate_dataset(samples_per_class=max(1, (args.index + 1) // len(LABELS) + 1),
                                       image_size=checkpoint["image_size"], seed=args.seed)
    if args.index < 0 or args.index >= len(images):
        parser.error(f"index must be from 0 to {len(images) - 1}")
    model = SurfaceCNN(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(images[args.index:args.index + 1]))
        probabilities = torch.softmax(logits, dim=1)[0]
    prediction = int(probabilities.argmax())
    print(f"sample_index: {args.index}  seed: {args.seed}")
    print(f"synthetic_label: {labels[int(targets[args.index])]}  predicted: {labels[prediction]}")
    print("scores (uncalibrated softmax; not real-world probabilities):")
    for label, score in zip(labels, probabilities.tolist()):
        print(f"  {label:10s} {score:.3f}")


if __name__ == "__main__":
    main()
