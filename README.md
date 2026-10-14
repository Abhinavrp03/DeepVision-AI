# DeepVision AI — Industrial Visual Anomaly Detection

DeepVision AI is an experimental computer-vision prototype that uses
pretrained MobileNetV2 features and nearest-neighbor distance scoring
to identify potentially anomalous bottle images.

## Features
- Pretrained CNN feature extraction with PyTorch
- Normal-image reference feature bank
- Image-level anomaly scoring
- Gradio image-upload interface
- Experimental spatial anomaly heatmap visualization

## Evaluation
Preliminary image-level ROC-AUC: 0.9992 on 83 MVTec AD bottle test images
(20 normal and 63 defective).

This result is specific to this experiment. Reproduce the evaluation and
validate the threshold before relying on it for other products or conditions.

## Run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `deepvision_artifacts.pt` is in the same directory as `app.py`.
3. Run: `python app.py`

The first run may download pretrained MobileNetV2 weights.

## Dataset
MVTec Anomaly Detection (MVTec AD), bottle category:
https://www.mvtec.com/company/research/datasets/mvtec-ad

The dataset has license restrictions, including non-commercial use.
Review the original license before using the dataset or derived materials.
The dataset itself is not included in this repository.

## Limitations
This is a research prototype, not a certified industrial inspection system.
The anomaly threshold is heuristic, and heatmaps are exploratory rather than
validated defect segmentation. Performance on other products is unknown.
