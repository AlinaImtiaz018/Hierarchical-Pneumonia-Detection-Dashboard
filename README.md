# Hierarchical Pneumonia Detection using Chest X-Rays

This project implements a **Hierarchical Deep Learning Pipeline** for detecting pneumonia from chest X-ray images using **DenseNet121**.

The system performs classification in two stages:

<img width="1396" height="778" align="center" alt="classification" src="https://github.com/user-attachments/assets/f892d486-8a1a-4dbe-bc75-a11a1c584c7c" />

Instead of directly predicting 3 classes simultaneously, the model first determines:

<table align="center" border="0">
  <tr>
    <td align="center">
      <img width="200" height="200" src="https://github.com/user-attachments/assets/ebc5830d-34ff-45dc-8f15-95a13d5c484c"/><br/>
      <sub>Normal</sub>
    </td>
    <td align="center">
      <img width="200" height="200" src="https://github.com/user-attachments/assets/cd073fbb-e1f1-4690-be18-b8be79962b0d"/><br/>
      <sub>Bacterial</sub>
    </td>
    <td align="center">
      <img width="200" height="200" src="https://github.com/user-attachments/assets/eff082ea-62cb-4398-bbf3-da30acd09325"/><br/>
      <sub>Viral</sub>
    </td>
  </tr>
</table>

<p align="center">
  <b>Figure 1: Chest X-ray classification examples</b>
</p>

1. NORMAL vs PNEUMONIA
2. If pneumonia is detected, a second model classifies:

   * VIRAL pneumonia
   * BACTERIAL pneumonia

This hierarchical approach is more clinically meaningful and improves interpretability.

---

# Dataset Structure

Place all chest X-ray images inside:

```text
chest_xray_dataset/
```

with the following structure:

```text
chest_xray_dataset/
├── NORMAL/
├── VIRUS/
└── BACTERIA/
```

Example:

```text
NORMAL/
    normal_1.jpg
    normal_2.jpg

VIRUS/
    virus_1.jpg
    virus_2.jpg

BACTERIA/
    bacteria_1.jpg
    bacteria_2.jpg
```

---

# Dataset Statistics

| Class    | Images |
| -------- | -----: |
| NORMAL   |   1583 |
| VIRUS    |   1493 |
| BACTERIA |   2780 |

The dataset imbalance is handled using:

* WeightedRandomSampler
* Weighted CrossEntropyLoss
* Data augmentation

---

# Project Structure

```text
Pneumonia_Project/
│
├── chest_xray_dataset/
│   ├── NORMAL/
│   ├── VIRUS/
│   └── BACTERIA/
│
├── data_stage1/
│   ├── train/
│   ├── val/
│   └── test/
│
├── data_stage2/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── stage1.pth
│   └── stage2.pth
│
├── results/
│   ├── stage1/
│   └── stage2/
│
├── test_images/
│
├── split_dataset.py
├── utils.py
├── train_stage1.py
├── train_stage2.py
├── inference.py
├── evaluate.py
├── visualize_results.py
├── visualization.py
├── gradcam.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Hierarchical Classification Pipeline

## Stage 1 — Binary Classification

DenseNet121 model:

```text
NORMAL vs PNEUMONIA
```

Where:

```text
PNEUMONIA = VIRUS + BACTERIA
```

Output:

| Label | Meaning   |
| ----- | --------- |
| 0     | NORMAL    |
| 1     | PNEUMONIA |

---

## Stage 2 — Pneumonia Subtype Classification

Executed only if Stage 1 predicts pneumonia.

DenseNet121 model:

```text
VIRUS vs BACTERIA
```

Output:

| Label | Meaning   |
| ----- | --------- |
| 0     | VIRAL     |
| 1     | BACTERIAL |

---

# Why DenseNet121?

DenseNet121 is widely used in medical imaging because it provides:

* Strong feature reuse
* Efficient gradient flow
* Better performance on small medical datasets
* Strong chest X-ray performance
* Better Grad-CAM visualizations

Inspired by:

* CheXNet
* Medical imaging literature

---

# Features

Implemented features include:

* Hierarchical classification pipeline
* DenseNet121 pretrained backbone
* WeightedRandomSampler
* Weighted CrossEntropyLoss
* Data augmentation
* Early stopping
* Model checkpoint saving
* Evaluation metrics
* Confusion matrix generation
* ROC curve generation
* Grad-CAM explainability
* GPU support
* Training history visualization

---

# Data Augmentation

The following augmentations are applied during training:

* Random horizontal flip
* Small rotation
* Small affine translation
* Brightness/contrast jitter

Medical precautions:

* No vertical flips
* No aggressive cropping
* No extreme rotations

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd Hierarchical-Pneumonia-Detection-using-Chest-X-Rays
```

---

## 2. Create Virtual Environment

Windows:

```bash
python -m venv pneumonia-env
```

---

## 3. Activate Environment

Windows:

```bash
pneumonia-env\Scripts\activate
```

If successful:

```text
(pneumonia-env)
```

appears in the terminal.

---

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install torch torchvision scikit-learn matplotlib pillow tqdm grad-cam
```

---

# Running the Project

---

## Step 1 — Split Dataset

Run:

```bash
python split_dataset.py
```

This creates:

```text
data_stage1/
data_stage2/
```

with train/validation/test splits.

---

## Step 2 — Train Stage 1 Model

Binary classifier:

```text
NORMAL vs PNEUMONIA
```

Run:

```bash
python train_stage1.py
```

Saved model:

```text
models/stage1.pth
```

---

## Step 3 — Train Stage 2 Model

Pneumonia subtype classifier:

```text
VIRUS vs BACTERIA
```

Run:

```bash
python train_stage2.py
```

Saved model:

```text
models/stage2.pth
```

---

## Step 4 — Evaluate Models

Run:

```bash
python evaluate.py
```

Generated outputs:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* ROC curve

Saved inside:

```text
results/
```

---

## Step 5 — Visualize Training Curves

Run:

```bash
python visualize_results.py
```

Outputs:

* Loss curves
* Accuracy curves

---

## Step 6 — Run Hierarchical Inference

Place a test image inside:

```text
test_images/
```

Run:

```bash
python inference.py
```

Example output:

```text
Prediction: BACTERIAL
```

or:

```text
Prediction: NORMAL
```

---

# Grad-CAM Explainability

Grad-CAM visualizes which lung regions influenced the prediction.

Run:

```bash
python gradcam.py
```

Expected behavior:

## Viral Pneumonia
<img width="758" height="202" align="center" alt="viral" src="https://github.com/user-attachments/assets/cc7cda7d-ac30-40f7-889f-b2fa85acf85f" />

* Diffuse bilateral opacities
* Widespread interstitial patterns

## Bacterial Pneumonia
<img width="762" height="201" align="center" alt="bacterial" src="https://github.com/user-attachments/assets/65d2dcad-b738-492f-ab8c-98a6e66e530c" />

* Localized lobar consolidation
* Focal opacity regions

---

# Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* AUC Score

Medical AI systems should not rely on accuracy alone.

---

# GPU Support

CUDA is automatically detected:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Verify GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

# Important Medical AI Notes

Always inspect Grad-CAM outputs.

Deep learning models may incorrectly learn shortcuts such as:

* Image borders
* Embedded text
* Scanner artifacts
* Hospital markers

instead of actual pathology.

---

# Future Improvements

Potential future extensions:

* Vision Transformers (ViT)
* Swin Transformers
* Ensemble learning
* Multi-label disease detection
* Lung segmentation
* Attention mechanisms
* Clinical report generation
* FastAPI deployment
* Docker deployment
* SHAP explainability
* Monte-Carlo Dropout uncertainty estimation

---

# References

DenseNet:

```text
Huang et al. — Densely Connected Convolutional Networks
```

CheXNet:

```text
Rajpurkar et al. — CheXNet: Radiologist-Level Pneumonia Detection
```

---

# License

This project is licensed under [LICENSE](LICENSE), intended for educational and research purposes only.

It is NOT approved for clinical diagnosis.
