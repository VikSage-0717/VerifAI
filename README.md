# AI-Generated Image Detector

A lightweight deep-learning project for classifying images as real or AI-generated, evaluating cross-generator generalization, and explaining predictions with Grad-CAM.

## Project layout

```text
data/                 # Downloaded datasets (not committed)
models/               # Saved checkpoints (not committed)
outputs/              # Metrics, plots, and Grad-CAM images (not committed)
src/ai_image_detector/# Reusable training and inference code
scripts/              # Command-line entry points
app/                  # Streamlit web application
configs/              # Experiment configuration files
```

## Quick start

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Put data into the structure documented in `data/README.md`.

4. Train the baseline model:

   ```powershell
   python scripts/train.py --config configs/cifake_baseline.yaml
   ```

5. Evaluate a saved model:

   ```powershell
   python scripts/evaluate.py --checkpoint models/baseline/best_model.pt --data-dir data/test_seen
   ```

6. Launch the app:

   ```powershell
   streamlit run app/streamlit_app.py
   ```

## Research protocol

Train on real images plus fakes from one generator. Evaluate both on the same generator (`test_seen`) and on separate, never-seen generators (`test_unseen/<generator>`). Report the accuracy and F1-score drop between these settings.

This is a research prototype: a prediction score is evidence, not proof that an image is AI-generated.

## Included dataset

The CIFAKE dataset is included locally at `data/raw/cifake/`, retaining its original `train/FAKE`, `train/REAL`, `test/FAKE`, and `test/REAL` structure. Use `configs/cifake_baseline.yaml` to train with it. CIFAKE is useful for establishing a working baseline, but it cannot evaluate cross-generator generalization by itself; add a generator-labelled dataset such as GenImage for that research question.
