"""Train a lightweight real-versus-AI-image classifier."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_image_detector.config import load_config
from ai_image_detector.data import are_binary_real_fake_classes, build_loader
from ai_image_detector.model import create_model




def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_total, labels, predictions = 0.0, [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_total += criterion(logits, targets).item() * len(targets)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return loss_total / len(loader.dataset), accuracy_score(labels, predictions), f1_score(labels, predictions, pos_label=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    device = choose_device(train_cfg["device"])
    train_loader, classes = build_loader(data_cfg["train_dir"], data_cfg["batch_size"], data_cfg["image_size"], data_cfg["num_workers"], True, train_cfg["augment"])
    val_loader, val_classes = build_loader(data_cfg["val_dir"], data_cfg["batch_size"], data_cfg["image_size"], data_cfg["num_workers"])
    if not are_binary_real_fake_classes(classes) or val_classes != classes:
        raise ValueError("Each split must contain matching fake/real class folders (case-insensitive).")

    model = create_model(**model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train(); running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * len(targets)
        val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device)
        metrics = {"epoch": epoch, "train_loss": running_loss / len(train_loader.dataset), "val_loss": val_loss, "val_accuracy": val_accuracy, "val_f1_fake": val_f1}
        history.append(metrics)
        print(metrics)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "architecture": model_cfg["architecture"], "dropout": model_cfg["dropout"], "classes": classes, "image_size": data_cfg["image_size"]}, output_dir / "best_model.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best model to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
