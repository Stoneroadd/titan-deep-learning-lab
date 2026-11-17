"""Train and evaluate a small CNN on the locally generated demo dataset."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import SurfaceCNN
from synthetic_data import LABELS, generate_dataset


def metrics_from_predictions(targets: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        precision = tp / max(1, int(matrix[:, index].sum()))
        recall = tp / max(1, int(matrix[index, :].sum()))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": int(matrix[index, :].sum())}
    return {"accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
            "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
            "per_class": per_class, "confusion_matrix": matrix.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = generate_dataset(args.samples_per_class, seed=args.seed)
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    validation_size = max(len(LABELS), int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    train_data, validation_data = random_split(
        dataset, [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    model = SurfaceCNN(len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history: list[dict] = []
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)

        model.eval()
        targets: list[int] = []
        predictions: list[int] = []
        with torch.inference_mode():
            for batch_images, batch_labels in validation_loader:
                logits = model(batch_images.to(device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(batch_labels.tolist())
        scores = metrics_from_predictions(np.asarray(targets), np.asarray(predictions))
        row = {"epoch": epoch, "train_loss": total_loss / training_size,
               "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"]}
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={row['train_loss']:.4f}  "
              f"val_accuracy={scores['accuracy']:.3f}  macro_f1={scores['macro_f1']:.3f}")
        if scores["accuracy"] > best_accuracy:
            best_accuracy = scores["accuracy"]
            torch.save({"state_dict": model.state_dict(), "labels": list(LABELS),
                        "image_size": 32, "seed": args.seed}, args.output / "best_model.pt")
            best_scores = scores

    payload = {"dataset": "procedural synthetic grayscale patterns", "seed": args.seed,
               "samples_per_class": args.samples_per_class, "train_count": training_size,
               "validation_count": validation_size, "device": str(device), "history": history,
               "best_validation": best_scores}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *LABELS])
        for label, row in zip(LABELS, best_scores["confusion_matrix"]):
            writer.writerow([label, *row])
    print(f"Saved checkpoint and reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()
