"""Create a Grad-CAM heatmap for one image prediction."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_image_detector.model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/gradcam.png")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["architecture"], pretrained=False, dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    if checkpoint["architecture"] == "mobilenet_v3_small":
        target_layers = [model.features[-1]]
    elif checkpoint["architecture"] == "resnet18":
        target_layers = [model.layer4[-1]]
    else:
        raise ValueError("No Grad-CAM target layer registered for this architecture.")

    original = Image.open(args.image).convert("RGB")
    resized = original.resize((checkpoint["image_size"], checkpoint["image_size"]))
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(resized).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).item()
    with GradCAM(model=model, target_layers=target_layers) as cam:
        heatmap = cam(input_tensor=tensor, targets=None)[0]
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Prediction: {checkpoint['classes'][prediction]}. Grad-CAM saved to {output}")


if __name__ == "__main__":
    main()
