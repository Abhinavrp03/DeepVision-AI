import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_PATH = "deepvision_artifacts.pt"

artifacts = torch.load(
    ARTIFACT_PATH, map_location="cpu", weights_only=True
)
normal_features = artifacts["normal_features"].to(DEVICE)
patch_bank = artifacts["patch_bank"].to(DEVICE)
threshold = float(artifacts["threshold"])

weights = models.MobileNet_V2_Weights.DEFAULT
backbone = models.mobilenet_v2(weights=weights).features.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def inspect(image):
    if image is None:
        return None, None, "Please upload an image."

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    fmap = backbone(tensor)
    vector = F.normalize(
        F.adaptive_avg_pool2d(fmap, 1).flatten(1), dim=1
    ).cpu()

    score = torch.cdist(vector, normal_features).min().item()
    result = "Potential anomaly" if score > threshold else "Likely normal"

    patches = F.normalize(
        fmap.flatten(2).transpose(1, 2), dim=-1
    ).squeeze(0)

    distances = torch.cdist(patches, patch_bank)
    patch_scores = distances.min(dim=1).values

    h, w = fmap.shape[-2:]
    heatmap = F.interpolate(
        patch_scores.reshape(1, 1, h, w),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy()

    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min() + 1e-8
    )

    original = image.resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Anomaly map")
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    details = (
        f"Result: {result}\n"
        f"Anomaly score: {score:.4f}\n"
        f"Prototype threshold: {threshold:.4f}\n\n"
        "Experimental research prototype. Results are not "
        "validated for industrial decisions."
    )
    return original, fig, details

demo = gr.Interface(
    fn=inspect,
    inputs=gr.Image(type="pil", label="Upload bottle image"),
    outputs=[
        gr.Image(label="Input"),
        gr.Plot(label="Anomaly heatmap"),
        gr.Textbox(label="Inspection result", lines=5)
    ],
    title="DeepVision AI",
    description="Experimental industrial visual anomaly detection."
)

demo.launch()
