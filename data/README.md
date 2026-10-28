# Dataset layout

`torchvision.datasets.ImageFolder` assigns labels alphabetically. Use these exact class names so that `fake` is label `0` and `real` is label `1`:

```text
data/
  train/
    fake/     # Fakes from the single training generator, e.g. Stable Diffusion
    real/
  val/
    fake/
    real/
  test_seen/
    fake/     # Same generator used for training
    real/
  test_unseen/
    stylegan/
      fake/
      real/
    generator_b/
      fake/
      real/
```

For every split, make the number of real and fake images roughly equal. Do not let duplicate or near-duplicate images appear in both training and testing. Record the dataset source, generator, image count, and license in `data/dataset_manifest.csv`.

## Included CIFAKE data

CIFAKE has been downloaded unchanged to `data/raw/cifake/`. It uses uppercase folder names and its original split:

```text
data/raw/cifake/
  train/FAKE/  # 50,000 latent-diffusion images
  train/REAL/  # 50,000 CIFAR-10 images
  test/FAKE/   # 10,000 latent-diffusion images
  test/REAL/   # 10,000 CIFAR-10 images
```

Use this source directly with `configs/cifake_baseline.yaml`. It is intentionally kept separate from the `data/train` layout used by future GenImage experiments.
