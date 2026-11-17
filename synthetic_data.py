"""Procedural grayscale images for an educational deep-learning exercise."""
from __future__ import annotations

import numpy as np

LABELS = ("intact", "crack", "stain", "edge_chip")


def generate_dataset(samples_per_class: int = 200, image_size: int = 32,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return NCHW float32 images and integer labels, without external files."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")
    if image_size < 16:
        raise ValueError("image_size must be >= 16")

    rng = np.random.default_rng(seed)
    axis = np.linspace(-1, 1, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label in range(len(LABELS)):
        for _ in range(samples_per_class):
            # Base material has smooth illumination and low-amplitude texture.
            image = 0.66 + 0.07 * xx + 0.035 * yy
            image = image + rng.normal(0, 0.035, size=(image_size, image_size))
            if label == 1:  # Narrow, slightly curved dark fracture.
                slope = rng.uniform(-0.28, 0.28)
                curve = rng.uniform(0.04, 0.17) * np.sin(xx * rng.uniform(2.0, 4.0))
                distance = np.abs(yy - slope * xx - curve - rng.uniform(-0.12, 0.12))
                image -= (distance < rng.uniform(0.025, 0.055)) * rng.uniform(0.40, 0.62)
            elif label == 2:  # Diffuse irregular discoloration.
                cx, cy = rng.uniform(-0.5, 0.5, size=2)
                sx, sy = rng.uniform(0.18, 0.48, size=2)
                blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
                image -= blob * rng.uniform(0.20, 0.43)
            elif label == 3:  # Missing material at one randomly chosen edge.
                edge = int(rng.integers(4))
                reach = rng.uniform(0.35, 0.78)
                if edge == 0:
                    chip = (xx < -1 + reach * (1 - np.abs(yy)))
                elif edge == 1:
                    chip = (xx > 1 - reach * (1 - np.abs(yy)))
                elif edge == 2:
                    chip = (yy < -1 + reach * (1 - np.abs(xx)))
                else:
                    chip = (yy > 1 - reach * (1 - np.abs(xx)))
                image[chip] -= rng.uniform(0.32, 0.55)
            image = np.clip(image, 0, 1).astype(np.float32)
            images.append(image[None, :, :])
            labels.append(label)

    order = rng.permutation(len(labels))
    x = np.stack(images)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    return x, y
