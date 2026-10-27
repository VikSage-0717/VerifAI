# Commands

Run all commands from the project root.

```powershell
# Baseline experiment
python scripts/train.py --config configs/baseline.yaml

# Evaluate on the generator included during training
python scripts/evaluate.py --checkpoint models/baseline/best_model.pt --data-dir data/test_seen --output outputs/seen_metrics.json

# Evaluate on a generator never included during training
python scripts/evaluate.py --checkpoint models/baseline/best_model.pt --data-dir data/test_unseen/stylegan --output outputs/stylegan_metrics.json

# Generate a visual explanation for a prediction
python scripts/gradcam.py --checkpoint models/baseline/best_model.pt --image path/to/image.jpg --output outputs/gradcam_example.png
```
