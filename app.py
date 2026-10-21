"""Professional dashboard for the hierarchical chest X-ray classifier.

The dashboard reads the existing dataset, result, and checkpoint files only.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

st.set_page_config(page_title="Chest X-Ray Dashboard", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
DEFAULTS = {"data_stage1": ROOT/"data"/"data_stage1", 
            "data_stage2": ROOT/"data"/"data_stage2",
            "model_stage1": ROOT/"models"/"stage1.pth", 
            "model_stage2": ROOT/"models"/"stage2.pth",
            "results_stage1": ROOT/"results"/"stage1", 
            "results_stage2": ROOT/"results"/"stage2",
            "source_data": ROOT/"chest_xray_dataset",
            "test_images": ROOT/"test_images"}

S1, S2 = ["NORMAL", "PNEUMONIA"], ["VIRUS", "BACTERIA"]

SUFFIXES = {".jpg", ".jpeg", ".png"}

COLORS = {"NORMAL": "#E8BCB9", "PNEUMONIA": "#AE445A", "VIRUS": "#F39F5A", "BACTERIA": "#662549"}

def css() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root{
    --bg:#0a0a0a;
    --card:#171717;
    --card1:#1a1818;
    --card2:#59232e;
    --card3:#35101E;
    --border:#662549;
    --accent:#F39F5A;
    --text:#fcf1f0;
    --muted:#E8BCB9;
    --title-accent:#b52f5d;
    --warning:#F39F5A
    }
    html,body,[class*="css"]{
    font-family:Inter,sans-serif
    }
    .stApp{
    background:var(--bg);color:var(--text)
    }
    [data-testid="stSidebar"]{
    background:#54162B;border-right:1px solid #451952
    }
    .block-container{
    max-width:1440px;padding-top:2.1rem;padding-bottom:3rem
    }
    #MainMenu,footer{
    # visibility:hidden
    }
    header[data-testid="stHeader"]{
    visibility:visible;background:transparent
    }
    button[data-testid="stExpandSidebarButton"]{
    visibility:visible!important;color:var(--text)!important;background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:8px!important
    }
    button[data-testid="stExpandSidebarButton"]:hover{
    border-color:var(--accent)!important;color:var(--accent)!important
    }
    .eyebrow{
    color:var(--accent);font-size:.72rem;letter-spacing:.14em;font-weight:700;text-transform:uppercase;margin-bottom:.55rem
    }
    .hero{
    font-size:clamp(2rem,4vw,3.2rem);line-height:1.05;letter-spacing:-.05em;font-weight:800;color:var(--text);margin:0 0 .65rem
    }
    .page-kicker{
    color:var(--accent);font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.45rem
    }
    .page-title{
    color:var(--text);font-size:2.05rem;line-height:1.15;letter-spacing:-.035em;font-weight:750;margin:0
    }
    .title-accent{
    color:var(--title-accent)
    }
    .section .title-accent{
    color:#E8BCB9
    }
    .page-description{
    color:var(--muted);font-size:.96rem;line-height:1.55;margin:.55rem 0 0;max-width:860px
    }
    .section-rule{
    height:1px;background:linear-gradient(90deg,var(--accent) 0%,var(--border) 40%,rgba(102,37,73,0) 100%);margin:.7rem 0 1rem
    }
    .copy{
    max-width:850px;color:var(--muted);font-size:1.04rem;line-height:1.65
    }
    .metric,.panel{
    background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem;height:100%;box-sizing:border-box
    }
    .metric{
    min-height:128px
    }
    .metric:hover{
    border-color:var(--accent)
    }
    .label{
    color:var(--muted);font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase
    }
    .value{
    color:var(--text);font-size:1.85rem;line-height:1.2;font-weight:750;padding-top:.45rem
    }
    .detail{
    color:var(--muted);font-size:.8rem;padding-top:.38rem
    }
    .section{
    color:var(--text);font-size:1.28rem;font-weight:700;margin:2.25rem 0 .45rem
    }
    .note{color:var(--muted);margin:0 0 1rem;font-size:.9rem
    }
    .notice{
    border-left:3px solid var(--accent);background:var(--card1);border-radius:0 10px 10px 0;padding:.85rem 1rem;color:var(--muted);line-height:1.5
    }
    .warn{
    border-left-color:var(--warning);color:var(--text);background:rgba(243,159,90,.12)
    }
    .prediction{
    text-align:center;padding:2rem 1.2rem
    }
    .overview-feature{
    background:linear-gradient(110deg,#1a1818 0%,#24121d 100%);border:1px solid var(--border);border-radius:14px;padding:1.45rem 1.55rem;min-height:245px;box-sizing:border-box
    }
    .overview-feature-title{
    color:var(--text);font-size:1.15rem;font-weight:700;margin:.45rem 0
    }
    .overview-feature-copy{
    color:var(--muted);font-size:.9rem;line-height:1.6;margin:0
    }
    .overview-chip{
    display:inline-block;background:rgba(243,159,90,.13);border:1px solid rgba(243,159,90,.35);border-radius:999px;color:var(--accent);font-size:.73rem;font-weight:700;letter-spacing:.04em;padding:.3rem .65rem;text-transform:uppercase
    }
    .overview-spacer{
    height:1rem
    }
    .stButton>button{
    border-radius:9px;border:1px solid var(--border);background:var(--card2);color:var(--text);font-weight:600
    }
    .stButton>button:hover{
    border-color:var(--accent);color:var(--accent)
    }
    [data-baseweb="tab"]{
    color:var(--muted)!important;font-weight:600
    }
    [data-baseweb="tab"][aria-selected="true"]{
    color:var(--accent)!important
    }
    [data-testid="stFileUploader"]{
    background:var(--card);border:1px dashed var(--border);border-radius:12px;padding:.75rem
    }
    [data-testid="stDataFrame"]{
    --ag-background-color:#1a1818;--ag-header-background-color:#1a1818;--ag-odd-row-background-color:#1a1818;--ag-foreground-color:#E8BCB9;--ag-header-foreground-color:#E8BCB9;--ag-border-color:#662549;--ag-row-hover-color:#24121d;--gdg-bg-cell:#1a1818;--gdg-bg-cell-medium:#1a1818;--gdg-bg-header:#1a1818;--gdg-bg-header-has-focus:#1a1818;--gdg-bg-cell-selected:#24121d;--gdg-bg-cell-hovered:#24121d;--gdg-text-dark:#fcf1f0;--gdg-text-medium:#E8BCB9;--gdg-horizontal-border-color:#662549;--gdg-header-bottom-border-color:#662549;background:#1a1818;border:1px solid var(--border);border-radius:10px;overflow:hidden
    }
    [data-testid="stDataFrame"] [role="grid"],[data-testid="stDataFrame"] [role="columnheader"],[data-testid="stDataFrame"] .ag-root-wrapper,[data-testid="stDataFrame"] .ag-header,[data-testid="stDataFrame"] .ag-center-cols-viewport{
    background:#1a1818!important;color:var(--text)!important
    }
    [data-testid="stCode"],.stCodeBlock{
    background:var(--card3)!important;border:1px solid var(--border);border-radius:10px;overflow:hidden
    }
    [data-testid="stCode"] pre,.stCodeBlock pre{
    background:var(--card3)!important;color:var(--text)!important
    }
    </style>""", unsafe_allow_html=True)

def path(key: str) -> Path: return Path(st.session_state.get(f"path_{key}", DEFAULTS[key]))

def images(folder: Path) -> list[Path]:
    try: return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    except OSError: return []

def metric(label: str, value: str, detail: str = "") -> None:
    st.markdown(f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div><div class="detail">{html.escape(detail)}</div></div>', unsafe_allow_html=True)


def two_tone_title(title: str) -> str:
    """Keep the first word light and tint the remaining words with the theme pink."""
    first, separator, remainder = title.strip().partition(" ")
    first_html = html.escape(first)
    if not separator:
        return first_html
    return f'{first_html} <span class="title-accent">{html.escape(remainder)}</span>'


def heading(title: str, note: str = "") -> None:
    st.markdown(f'<div class="section">{two_tone_title(title)}</div><div class="section-rule"></div>', unsafe_allow_html=True)
    if note: st.markdown(f'<div class="note">{html.escape(note)}</div>', unsafe_allow_html=True)


def page_header(
    title: str,
    description: str,
    kicker: str = "Chest X-ray Interpretation Dashboard",
    show_rule: bool = True,
) -> None:
    rule = '<div class="section-rule"></div>' if show_rule else ""
    st.markdown(
        f'<div class="page-kicker">{html.escape(kicker)}</div>'
        f'<h1 class="page-title">{two_tone_title(title)}</h1>'
        f'<p class="page-description">{html.escape(description)}</p>'
        f'{rule}',
        unsafe_allow_html=True,
    )

def alert(text: str, warning: bool = False) -> None:
    st.markdown(f'<div class="notice {"warn" if warning else ""}">{text}</div>', unsafe_allow_html=True)

def read_json(file: Path) -> Optional[dict[str, Any]]:
    try:
        result = json.loads(file.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, json.JSONDecodeError): return None

def read_metrics(file: Path) -> dict[str, float]:
    result = {}
    try:
        for row in file.read_text(encoding="utf-8").splitlines():
            if ":" in row:
                key, value = row.split(":", 1); result[key.strip().lower()] = float(value.strip())
    except (OSError, ValueError): pass
    return result


def summary_metrics(results_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """Load the machine-readable evaluation artifact, with a legacy fallback."""
    summary = read_json(results_dir / "summary_metrics.json") or {}
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = read_metrics(results_dir / "metrics.txt")
    return summary, metrics

@st.cache_data(show_spinner=False)
def distribution(root: str, classes: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([{"Split": split.title(), "Class": cls, "Images": len(images(Path(root)/split/cls))}
                         for split in ("train", "val", "test") for cls in classes])


@st.cache_data(show_spinner=False)
def source_distribution(root: str) -> pd.DataFrame:
    classes = ("NORMAL", "VIRUS", "BACTERIA")
    return pd.DataFrame([{"Class": cls, "Images": len(images(Path(root) / cls))} for cls in classes])

@st.cache_resource(show_spinner="Loading DenseNet121 checkpoint…")
def load_model(filename: str, device: str):
    file = Path(filename)
    if not file.is_file(): return None
    try:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        saved = torch.load(file, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved)
        return model.to(device).eval()
    except (OSError, RuntimeError, KeyError): return None

def preprocess(image: Image.Image) -> torch.Tensor:
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(image.convert("RGB")).unsqueeze(0)

def inference(image: Image.Image, stage1: nn.Module, stage2: nn.Module, device: str):
    with torch.no_grad():
        tensor = preprocess(image).to(device); p1 = torch.softmax(stage1(tensor), dim=1)[0].cpu().numpy()
        details = {"Stage 1": dict(zip(S1, map(float, p1)))}
        if int(p1.argmax()) == 0: return "NORMAL", float(p1[0]), details
        p2 = torch.softmax(stage2(tensor), dim=1)[0].cpu().numpy(); details["Stage 2"] = dict(zip(S2, map(float, p2)))
        return ("VIRAL" if int(p2.argmax()) == 0 else "BACTERIAL"), float(p2.max()), details

def plot_style(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Inter","color":"#E8BCB9"},
        height=height, margin={"l":35,"r":150,"t":45,"b":35}, legend={"orientation":"v","x":1.02,"xanchor":"left","y":1,"yanchor":"top"},
        xaxis={"gridcolor":"#451952","zerolinecolor":"#451952"}, yaxis={"gridcolor":"#451952","zerolinecolor":"#451952"})
    return fig

def charts(history: dict[str, Any], title: str):
    keys = ("train_loss","val_loss","train_acc","val_acc")
    if not all(isinstance(history.get(k), list) and history[k] for k in keys): return None, None
    ep = list(range(1, len(history["train_loss"])+1))
    loss = go.Figure([go.Scatter(x=ep,y=history["train_loss"],name="Training",line={"color":"#F39F5A","width":2.5}),
                      go.Scatter(x=ep,y=history["val_loss"],name="Validation",line={"color":"#AE445A","width":2.5})])
    acc = go.Figure([go.Scatter(x=ep,y=np.array(history["train_acc"])*100,name="Training",line={"color":"#F39F5A","width":2.5}),
                     go.Scatter(x=ep,y=np.array(history["val_acc"])*100,name="Validation",line={"color":"#E8BCB9","width":2.5})])
    loss.update_layout(title=f"{title}: loss",xaxis_title="Epoch",yaxis_title="Cross-entropy loss")
    acc.update_layout(title=f"{title}: accuracy",xaxis_title="Epoch",yaxis_title="Accuracy (%)")
    return plot_style(loss), plot_style(acc)


def case_mix_chart(counts: dict[str, int]) -> Optional[go.Figure]:
    labels = ["NORMAL", "VIRUS", "BACTERIA"]
    values = [counts.get(label, 0) for label in labels]
    if not any(values):
        return None
    chart = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=.68,
        sort=False,
        marker={"colors": [COLORS[label] for label in labels], "line": {"color": "#171717", "width": 3}},
        textinfo="percent",
        textfont={"color": "#fcf1f0", "size": 12},
        hovertemplate="%{label}<br>%{value:,} cases (%{percent})<extra></extra>",
    ))
    chart.update_layout(
        title="Case mix",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#E8BCB9"},
        height=245,
        margin={"l": 5, "r": 5, "t": 42, "b": 5},
        annotations=[{"text": "Dataset<br>mix", "showarrow": False, "font": {"color": "#fcf1f0", "size": 14}}],
    )
    return chart

def render_sidebar() -> str:
    nav = [("Overview","overview"),("Dataset explorer","dataset"),("Methodology","methodology"),("Training results","training"),("Evaluation","evaluation"),("Interactive demo","demo")]
    with st.sidebar:
        st.markdown("<div style='padding:.7rem .4rem 1.2rem'><div style='font-size:2.1rem'>🫁</div><div style='font-size:1.18rem;font-weight:800;color:#E8BCB9'>Chest X-Ray App</div><div style='color:#E8BCB9;font-size:.77rem;line-height:1.5;margin-top:.25rem'>Hierarchical pneumonia detection<br>using DenseNet121</div></div>",unsafe_allow_html=True)
        choice = st.radio("Navigate",[name for name,_ in nav],label_visibility="collapsed")
        st.divider(); st.caption("SYSTEM STATUS")
        device = "CUDA" if torch.cuda.is_available() else "CPU"; s1,s2 = path("model_stage1").is_file(),path("model_stage2").is_file()
        s1_color, s2_color = ("#E8BCB9" if s1 else "#AE445A"), ("#E8BCB9" if s2 else "#AE445A")
        s1_text, s2_text = ("ready" if s1 else "checkpoint missing"), ("ready" if s2 else "checkpoint missing")
        st.markdown(f"<div style='font-size:.82rem;line-height:1.9;color:#E8BCB9'>Compute <span style='color:#F39F5A'>{device}</span><br>Stage 1 <span style='color:{s1_color}'>{s1_text}</span><br>Stage 2 <span style='color:{s2_color}'>{s2_text}</span></div>",unsafe_allow_html=True)
        st.divider(); st.caption("Educational and research use only — not a clinical diagnostic device.")
    return dict(nav)[choice]

def overview() -> None:
    page_header(
        "Hierarchical Pneumonia Detection",
        "This project explores a two-stage DenseNet121 workflow for chest X-rays: first screening images as normal or pneumonia, then separating pneumonia cases into viral and bacterial categories. Use this dashboard to examine the dataset, training behavior, evaluation evidence, and live model predictions.",
        "Machine learning Project Dashboard",
        show_rule=False,
    )
    _,m1 = summary_metrics(path("results_stage1")); _,m2 = summary_metrics(path("results_stage2"))
    history = read_json(path("results_stage1")/"history.json") or {}
    source = source_distribution(str(path("source_data")))
    counts = dict(zip(source["Class"], source["Images"])) if not source.empty else {}
    total_cases = sum(counts.values())
    def case_share(class_name: str) -> str:
        count = counts.get(class_name, 0)
        return f"{(count / total_cases) * 100:.1f}% of all cases" if total_cases else "Source dataset unavailable"
    heading("Dataset at a Glance", "Counts are calculated from \"chest_xray_dataset\" and updated when the configured source path changes.")
    cases = [
        ("Total X-ray cases", f"{total_cases:,}" if total_cases else "—", "All available categories"),
        ("Bacterial cases", f"{counts.get('BACTERIA', 0):,}" if total_cases else "—", case_share("BACTERIA")),
        ("Viral cases", f"{counts.get('VIRUS', 0):,}" if total_cases else "—", case_share("VIRUS")),
        ("Normal cases", f"{counts.get('NORMAL', 0):,}" if total_cases else "—", case_share("NORMAL")),
    ]
    for col, value in zip(st.columns(4), cases):
        with col: metric(*value)
    st.markdown("<div class='overview-spacer'></div>", unsafe_allow_html=True)
    summary_col, mix_col = st.columns(2)
    with summary_col:
        st.markdown(
            f"<div class='overview-feature'><span class='overview-chip'>Hierarchical workflow</span>"
            f"<div class='overview-feature-title'>Screen first. Subtype only when pneumonia is detected.</div>"
            f"<p class='overview-feature-copy'>The dataset contains <b>{total_cases:,}</b> radiographs. "
            f"Pneumonia cases are divided into <b>{counts.get('BACTERIA', 0):,} bacterial</b> and "
            f"<b>{counts.get('VIRUS', 0):,} viral</b> images, allowing the second model to focus on clinically relevant subtype prediction.</p></div>",
            unsafe_allow_html=True,
        )
    with mix_col:
        mix = case_mix_chart(counts)
        if mix: st.plotly_chart(mix, use_container_width=True)
    heading("Model Performance", "Scores are read from the most recent saved evaluation artifacts.")
    values = [("Stage 1 Accuracy",f"{m1['accuracy']*100:.1f}%" if "accuracy" in m1 else "—","Normal vs pneumonia"),
      ("Stage 1 F1",f"{m1.get('f1 score',0):.3f}" if m1 else "—","Evaluation artifact"),
      ("Stage 2 accuracy",f"{m2['accuracy']*100:.1f}%" if "accuracy" in m2 else "—","Viral vs bacterial"),
      ("Stage 1 epochs",str(len(history.get('train_loss',[]))) if history else "—","Recorded training history")]
    for col,value in zip(st.columns(4),values):
        with col: metric(*value)
    heading("Decision Pipeline","The second classifier runs only when Stage 1 detects pneumonia.")
    st.markdown("<div class='panel'><div style='display:flex;gap:1rem;align-items:center;flex-wrap:wrap'><div><div class='label'>Stage 1: Screening</div><b>NORMAL ↔ PNEUMONIA</b></div><div style='color:#F39F5A;font-size:1.5rem'>→</div><div><div class='label'>Stage 2: Subtype</div><b>VIRUS ↔ BACTERIA</b></div><div style='color:#F39F5A;font-size:1.5rem'>→</div><div><div class='label'>Final output</div><b style='color:#F39F5A'>NORMAL / VIRAL / BACTERIAL</b></div></div></div>",unsafe_allow_html=True)
    if history:
        heading("Training Snapshot","Stage 1 history saved by the original training script.")
        loss,acc = charts(history,"Stage 1"); left,right = st.columns(2)
        with left: st.plotly_chart(loss,use_container_width=True)
        with right: st.plotly_chart(acc,use_container_width=True)
    else: alert(f"No Stage 1 history found at <code>{path('results_stage1')/'history.json'}</code>.",True)

def show_dataset(root: Path, classes: list[str], name: str) -> None:
    if not root.is_dir():
        alert(f"Dataset directory unavailable: <code>{html.escape(str(root))}</code>. Run <code>split_dataset.py</code> or update settings.",True); return
    data = distribution(str(root),tuple(classes)); left,right = st.columns([1.45,1])
    with left:
        fig = px.bar(data,x="Split",y="Images",color="Class",barmode="group",color_discrete_map=COLORS,title=f"{name} class distribution")
        st.plotly_chart(plot_style(fig),use_container_width=True)
    with right:
        metric("Images across splits",f"{int(data['Images'].sum()):,}",name)
        st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
        st.dataframe(data.pivot(index="Split",columns="Class",values="Images").fillna(0).astype(int),use_container_width=True)
    heading("Representative Images","Samples are loaded from the training split; labels are read from the folder structure.")
    for col,cls in zip(st.columns(len(classes)),classes):
        with col:
            files=images(root/"train"/cls);st.caption(cls)
            if files: st.image(str(files[0]),use_container_width=True);st.caption(files[0].name)
            else: alert("No training image available.",True)

def dataset() -> None:
    page_header(
        "Dataset Explorer", 
        "Inspect class balance, train-validation-test splits, and representative chest X-rays from the configured dataset.", 
        show_rule=False,
    )
    a,b=st.tabs(["Stage 1: Normal vs Pneumonia","Stage 2: Viral vs Bacterial"])
    with a: show_dataset(path("data_stage1"),S1,"Stage 1")
    with b: show_dataset(path("data_stage2"),S2,"Stage 2")
    heading("Data Preparation")
    alert("Source labels are <b>NORMAL</b>, <b>VIRAL</b>, and <b>BACTERIA</b>. <br>Stage 1 merges viral and bacterial cases into PNEUMONIA.<br>Stage 2 uses the two pneumonia subclasses. Training uses 224×224 resizing, ImageNet normalization, class weighting, and weighted sampling.")

def methodology() -> None:
    page_header(
        "Model & Methodology", 
        "Review the two-stage DenseNet121 architecture, the label hierarchy, and the preprocessing choices used by the original training workflow.", 
        show_rule=True,
    )
    for col,(title,body) in zip(st.columns(2),[("Stage 1: Screening","Input: 224×224 RGB radiograph<br>Output: NORMAL (0) / PNEUMONIA (1)"),("Stage 2: Subtype","Input: 224×224 RGB pneumonia image<br>Output: VIRUS (0) / BACTERIA (1)")]):
        with col: st.markdown(f"<div class='panel'><div class='label'>{title}</div><h3 style='color:#E8BCB9'>DenseNet121</h3><p style='color:#E8BCB9'>{body}<br>Loss: weighted cross-entropy<br>Sampling: WeightedRandomSampler<br>Optimizer: Adam, lr=1e-5<br>Schedule: ReduceLROnPlateau + early stopping</p></div>",unsafe_allow_html=True)
    heading("Preprocessing and Augmentation")
    st.code("Resize(224, 224)\nRandomHorizontalFlip()\nRandomRotation(15)\nRandomAffine(translate=(0.08, 0.08), scale=(0.95, 1.05))\nColorJitter(brightness=0.15, contrast=0.15)\nToTensor() + ImageNet normalization")
    alert("No vertical flips or aggressive crops are used. The demo uses validation/inference normalization; it does not change the existing model architecture or methodology.")

def training() -> None:
    page_header(
        "Training Results", 
        "Explore recorded loss and accuracy curves from the experiment histories saved by the training scripts.",
          show_rule=False,
    )
    for tab,key,name in zip(st.tabs(["Stage 1","Stage 2"]),("results_stage1","results_stage2"),("Stage 1","Stage 2")):
        with tab:
            history=read_json(path(key)/"history.json")
            if not history: alert(f"History missing or unreadable at <code>{path(key)/'history.json'}</code>.",True);continue
            train_acc,val_acc=history.get("train_acc",[]),history.get("val_acc",[])
            facts=[("Recorded epochs",str(len(history.get("train_loss",[]))),"Actual training history"),("Best validation accuracy",f"{max(val_acc)*100:.2f}%" if val_acc else "—","Highest logged value"),("Final training accuracy",f"{train_acc[-1]*100:.2f}%" if train_acc else "—","Last logged epoch")]
            for col,fact in zip(st.columns(3),facts):
                with col:metric(*fact)
            loss,acc=charts(history,name);left,right=st.columns(2)
            with left:
                if loss:st.plotly_chart(loss,use_container_width=True)
            with right:
                if acc:st.plotly_chart(acc,use_container_width=True)


def classification_report_table(file: Path) -> pd.DataFrame:
    """Convert sklearn's persisted text report into a data table."""
    rows = []
    try:
        for line in file.read_text(encoding="utf-8").splitlines():
            values = line.rsplit(maxsplit=4)
            if len(values) != 5 or values[0].strip().lower() == "precision":
                continue
            label, precision, recall, f1, support = values
            rows.append({"Class": label.strip(), "Precision": float(precision),
                         "Recall": float(recall), "F1 score": float(f1),
                         "Support": int(support)})
    except (OSError, ValueError):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def confusion_matrix_chart(summary: dict[str, Any], name: str) -> Optional[go.Figure]:
    matrix, classes = summary.get("confusion_matrix"), summary.get("class_names")
    if not isinstance(matrix, list) or not isinstance(classes, list) or not matrix:
        return None
    chart = go.Figure(go.Heatmap(
        z=matrix, x=classes, y=classes, text=matrix, texttemplate="%{text}",
        textfont={"color": "#E8BCB9", "size": 16}, colorscale=[[0, "#35101E"], [0.5, "#662549"], [1, "#F39F5A"]],
        showscale=False, hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    chart.update_layout(title=f"{name} — confusion matrix", xaxis_title="Predicted label", yaxis_title="True label")
    chart.update_yaxes(autorange="reversed")
    return plot_style(chart, 390)


def roc_chart(summary: dict[str, Any], name: str) -> Optional[go.Figure]:
    curve = summary.get("roc_curve")
    if not isinstance(curve, dict) or not isinstance(curve.get("fpr"), list) or not isinstance(curve.get("tpr"), list):
        return None
    auc_value = curve.get("auc")
    label = f"Model (AUC = {float(auc_value):.4f})" if isinstance(auc_value, (int, float)) else "Model"
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=label, mode="lines", line={"color": "#F39F5A", "width": 3}))
    chart.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="No-skill baseline", mode="lines", line={"color": "#AE445A", "width": 2, "dash": "dash"}))
    chart.update_layout(title=f"{name} — ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    chart.update_xaxes(range=[0, 1]); chart.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    return plot_style(chart, 390)

def evaluation_content(results: Path, name: str) -> None:
    summary,m = summary_metrics(results)
    if m:
        facts=[("Accuracy",f"{m.get('accuracy',0)*100:.2f}%","Test split"),("Precision",f"{m.get('precision',0):.4f}","Positive class"),("Recall",f"{m.get('recall',0):.4f}","Positive class"),("F1 score",f"{m.get('f1 score',0):.4f}","Harmonic mean")]
        for col,fact in zip(st.columns(4),facts):
            with col:metric(*fact)
    else:alert(f"No metrics.txt found at <code>{results/'metrics.txt'}</code>.",True)
    report=results/"classification_report.txt"
    if report.is_file():
        heading("Classification Report", "Parsed from the classification_report.txt saved by evaluate.py.")
        table = classification_report_table(report)
        if not table.empty:
            st.dataframe(table.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1 score": "{:.2f}"}), use_container_width=True, hide_index=True)
        else: alert("The saved classification report could not be parsed into a table.", True)
    heading("Evaluation Visualizations", "Interactive Plotly charts generated from summary_metrics.json.")
    matrix, roc = confusion_matrix_chart(summary, name), roc_chart(summary, name)
    if matrix is None or roc is None:
        alert(f"Missing <code>{results / 'summary_metrics.json'}</code>. Run <code>python evaluate.py</code> for this stage to create the JSON summary and interactive charts.", True)
        return
    left, right = st.columns(2)
    with left: st.plotly_chart(matrix, use_container_width=True)
    with right: st.plotly_chart(roc, use_container_width=True)

def evaluation() -> None:
    page_header(
        "Evaluation", 
        "Review held-out test metrics, class-level reports, confusion matrices, and ROC curves from the saved evaluation artifacts.", 
        show_rule=False
    )
    a,b=st.tabs(["Stage 1 — Normal vs Pneumonia","Stage 2 — Viral vs Bacterial"])
    with a:evaluation_content(path("results_stage1"),"Stage 1")
    with b:evaluation_content(path("results_stage2"),"Stage 2")

def demo() -> None:
    page_header(
        "Interactive Prediction", 
        "Upload a chest X-ray or select a bundled test image to run the complete hierarchical prediction workflow.", 
        show_rule=True,
    )
    device="cuda" if torch.cuda.is_available() else "cpu";model1,model2=path("model_stage1"),path("model_stage2")
    if not model1.is_file() or not model2.is_file():alert(f"Both checkpoints are required: <code>{model1}</code> and <code>{model2}</code>.",True);return
    left,right=st.columns([1.3,1])
    with left:uploaded=st.file_uploader("Upload X-ray",type=["jpg","jpeg","png"])
    samples=images(path("test_images"))
    with right:chosen=st.selectbox("Or select a bundled test image",["Select a sample…"]+[p.name for p in samples])
    image: Optional[Image.Image]=None
    try:
        if uploaded is not None:image=Image.open(uploaded).convert("RGB")
        elif chosen!="Select a sample…":image=Image.open(path("test_images")/chosen).convert("RGB")
    except (OSError,ValueError):alert("The selected image could not be opened.",True);return
    if image is None:alert("Choose an image to begin. Inference runs locally with configured project checkpoints.");return
    s1,s2=load_model(str(model1),device),load_model(str(model2),device)
    if s1 is None or s2 is None:alert("A checkpoint exists but could not be loaded. Confirm it matches the project’s DenseNet121 checkpoint format.",True);return
    with st.spinner("Running the hierarchical model…"):label,confidence,details=inference(image,s1,s2,device)
    left,right=st.columns([1,1.05])
    with left:st.image(image,caption="Input radiograph",use_container_width=True)
    color="#E8BCB9" if label=="NORMAL" else "#F39F5A" if label=="VIRAL" else "#AE445A"
    with right:
        st.markdown(f"<div class='panel prediction'><div class='label'>Final prediction</div><div style='font-size:2.35rem;font-weight:800;color:{color};margin:.4rem 0'>{label}</div><div style='color:#E8BCB9'>Model confidence: {confidence*100:.1f}%</div></div>",unsafe_allow_html=True)
        rows=[{"Stage":stage,"Class":cls,"Probability":value} for stage,values in details.items() for cls,value in values.items()]
        st.plotly_chart(plot_style(px.bar(pd.DataFrame(rows),x="Probability",y="Class",color="Stage",orientation="h",range_x=[0,1],title="Stage-wise class probabilities",color_discrete_sequence=["#F39F5A","#E8BCB9"]),260),use_container_width=True)
    if GRADCAM_AVAILABLE:
        with st.expander("Generate Stage 1 Grad-CAM"):
            if st.button("Create attention heatmap"):
                try:
                    heatmap=GradCAM(model=s1,target_layers=[s1.features.denseblock4])(input_tensor=preprocess(image).to(device))[0]
                    source=np.asarray(image.resize((224,224))).astype(np.float32)/255
                    st.image(show_cam_on_image(source,heatmap,use_rgb=True),caption="Stage 1 Grad-CAM",use_container_width=True)
                except (RuntimeError,ValueError) as err:alert(f"Grad-CAM could not be generated: {html.escape(str(err))}",True)
    else:alert("Install <code>grad-cam</code> to enable optional Grad-CAM visualizations.")
    alert("For educational and research use only. This model output is not clinical advice or a medical diagnosis.",True)

def main() -> None:
    css()
    for key,value in DEFAULTS.items():st.session_state.setdefault(f"path_{key}",str(value))
    {"overview":overview,"dataset":dataset,"methodology":methodology,"training":training,"evaluation":evaluation,"demo":demo}[render_sidebar()]()
if __name__ == "__main__":main()
