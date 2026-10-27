"""Evaluate a checkpoint on a seen or unseen generator test split."""
import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_image_detector.data import build_loader
from ai_image_detector.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    loader, classes = build_loader(args.data_dir, args.batch_size, checkpoint["image_size"], 2)
    if classes != checkpoint["classes"]:
        raise ValueError(f"Expected classes {checkpoint['classes']}; found {classes}")
    model = create_model(checkpoint["architecture"], pretrained=False, dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    labels, predictions, fake_scores = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            labels.extend(targets.tolist()); predictions.extend(probabilities.argmax(1).cpu().tolist()); fake_scores.extend(probabilities[:, 0].cpu().tolist())
    metrics = {"data_dir": args.data_dir, "accuracy": accuracy_score(labels, predictions), "f1_fake": f1_score(labels, predictions, pos_label=0), "roc_auc_fake": roc_auc_score([label == 0 for label in labels], fake_scores), "confusion_matrix": confusion_matrix(labels, predictions).tolist(), "report": classification_report(labels, predictions, target_names=classes, output_dict=True)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
