# Chest X-Ray AI — Hierarchical Pneumonia Detection Dashboard

This repository contains an interactive **Streamlit application** that demonstrates a two-stage DenseNet121 pipeline for chest X-ray analysis. The app provides a user-friendly interface for exploring the dataset, reviewing training and evaluation results, and running live hierarchical predictions (normal vs pneumonia, then viral vs bacterial).

The purpose of this repository is to make the project accessible through an interactive web application without requiring users to execute notebooks or write code.

---

## Features

* Interactive hierarchical pneumonia detection demonstrator
* Dataset exploration with class balance and sample radiographs
* Methodology overview (architecture, preprocessing, training setup)
* Training curves and recorded experiment history
* Evaluation dashboard (metrics, classification reports, confusion matrices, ROC curves)
* Live inference with stage-wise probability breakdown
* Optional Grad-CAM visualization (when `grad-cam` is installed)
* Dark-themed, responsive Streamlit user interface

---

## Project Structure

```text
Chest-XRay-AI/
|
├── data/
|   ├── data_stage1/          # Stage 1 splits (NORMAL / PNEUMONIA)
|   └── data_stage2/          # Stage 2 splits (VIRUS / BACTERIA)
├── models/
|   ├── stage1.pth            # DenseNet121 screening checkpoint
|   └── stage2.pth            # DenseNet121 subtype checkpoint
├── results/
|   ├── stage1/               # history.json, metrics, reports, summary
|   └── stage2/
├── chest_xray_dataset/       # Source labels: NORMAL, VIRUS, BACTERIA
├── test_images/              # Bundled sample radiographs for the demo
├── app.py
├── CITATION.cff
├── CONDUCT.md
├── LICENSE                    # Main Streamlit dashboard (or equivalent entrypoint)
├── README.md
└── requirements.txt
```

Paths can be adjusted in the app; defaults point at the folders above relative to the project root.

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd Chest-XRay-AI
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (for Grad-CAM heatmaps in the demo):

```bash
pip install grad-cam
```

Ensure the model checkpoints (`models/stage1.pth`, `models/stage2.pth`) and result artifacts under `results/` are present if you want training and evaluation pages fully populated.

---

## Running the Application

Launch Streamlit (replace `app.py` with your actual entrypoint filename if different):

```bash
streamlit run app.py
```

The application will open in your default browser; if it does not, copy the local URL shown in the terminal after launch.

---

## Application Pages

| Page | Description |
|------|-------------|
| **Overview** | Project summary, dataset counts, model performance snapshot, decision pipeline |
| **Dataset explorer** | Class distribution charts, split tables, representative training images |
| **Methodology** | Stage 1 / Stage 2 architecture, loss, sampling, preprocessing and augmentation |
| **Training results** | Loss and accuracy curves from saved experiment histories |
| **Evaluation** | Test metrics, classification reports, confusion matrices, ROC curves |
| **Interactive demo** | Upload or select a chest X-ray and run the full hierarchical prediction |

---

## Technologies Used

* Python
* Streamlit
* PyTorch / torchvision
* DenseNet121
* Pandas
* NumPy
* Plotly
* Pillow (PIL)
* pytorch-grad-cam (optional)

---

## Purpose

This repository serves as a portfolio demonstration of a hierarchical chest X-ray classifier. It makes it easy to explore the dataset, training behavior, evaluation evidence, and live model predictions through a single web interface—intended for educational and research use only, not clinical diagnosis.

---

## Citation

If you use this repository in your work or research, please cite it using the provided **CITATION.cff** file (if present) or reference this repository appropriately.

---

## License

This project is licensed under the MIT License. See the **LICENSE** file for details.
