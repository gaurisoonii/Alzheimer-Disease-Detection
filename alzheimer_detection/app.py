

import os
import io
import tempfile
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="Alzheimer's Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    BEST_MODEL_PATH, CLASSES, NUM_CLASSES,
    CLINICAL_FEATURES, CLINICAL_DIM, IMG_SIZE
)
from model import MultiModalFusionModel
from preprocessing import (
    resample_volume, skull_strip, normalise, crop_or_pad, load_nifti
)
from gradcam import GradCAM3D, visualise_gradcam, visualise_all_classes


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4A90D9;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .metric-box {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .pred-label {
        font-size: 2rem;
        font-weight: 700;
    }
    .pred-CN    { color: #4CAF50; }
    .pred-MCI   { color: #FF9800; }
    .pred-AD    { color: #F44336; }
    .disclaimer {
        background: #2a2a2a;
        border-left: 4px solid #FF9800;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiModalFusionModel().to(device)
    if os.path.exists(BEST_MODEL_PATH):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model, device
    st.error(f"Model checkpoint not found at {BEST_MODEL_PATH}. "
              "Please train the model first (python train.py).")
    return None, device


# ─── MRI loading helper ───────────────────────────────────────────────────────

def load_mri(uploaded_file) -> np.ndarray | None:
    suffix = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(
        suffix='.nii.gz' if 'nii' in suffix else '.npy',
        delete=False
    ) as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    try:
        if suffix.endswith('.npy'):
            volume = np.load(tmp_path).astype(np.float32)
            if volume.shape != IMG_SIZE:
                volume = crop_or_pad(volume, IMG_SIZE)
        else:
            data, affine = load_nifti(tmp_path)
            data   = resample_volume(data, affine, (1.5, 1.5, 1.5))
            data   = skull_strip(data)
            data   = normalise(data)
            volume = crop_or_pad(data, IMG_SIZE)
        return volume
    except Exception as e:
        st.error(f"Failed to load MRI: {e}")
        return None
    finally:
        os.unlink(tmp_path)


# ─── Confidence chart ─────────────────────────────────────────────────────────

def confidence_chart(probs: np.ndarray) -> go.Figure:
    colors = ['#4CAF50', '#FF9800', '#F44336']
    fig = go.Figure(go.Bar(
        x=CLASSES, y=(probs * 100).tolist(),
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title='Confidence (%)',
        yaxis_range=[0, 110],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    return fig


# ─── Slice viewer ─────────────────────────────────────────────────────────────

def show_slices(volume: np.ndarray, cam: np.ndarray | None = None):
    """Interactive slice viewer with optional CAM overlay."""
    D, H, W = volume.shape

    col1, col2, col3 = st.columns(3)
    d_idx = col1.slider("Axial slice",    0, D-1, D//2)
    h_idx = col2.slider("Coronal slice",  0, H-1, H//2)
    w_idx = col3.slider("Sagittal slice", 0, W-1, W//2)

    alpha = st.slider("CAM overlay intensity", 0.0, 1.0, 0.45) if cam is not None else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#111111')

    slices_mri = [volume[d_idx], volume[:, h_idx], volume[:, :, w_idx]]
    slices_cam = [cam[d_idx],    cam[:, h_idx],    cam[:, :, w_idx]] \
                  if cam is not None else [None, None, None]
    titles = ['Axial', 'Coronal', 'Sagittal']

    import matplotlib.cm as mpl_cm
    for ax, mri_sl, cam_sl, title in zip(axes, slices_mri, slices_cam, titles):
        if cam_sl is not None and alpha > 0:
            s = mri_sl.copy(); s -= s.min()
            if s.max() > 1e-8: s /= s.max()
            mri_rgb = np.stack([s, s, s], axis=-1)
            cam_rgb = mpl_cm.get_cmap('jet')(cam_sl)[:, :, :3]
            blended = np.clip((1 - alpha) * mri_rgb + alpha * cam_rgb, 0, 1)
            ax.imshow(blended, aspect='auto')
        else:
            ax.imshow(mri_sl, cmap='gray', aspect='auto')
        ax.set_title(title, color='white')
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── App layout ───────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header">🧠 Alzheimer\'s Detection AI</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-modal MRI + clinical feature fusion model</div>',
                unsafe_allow_html=True)

    model, device = load_model()

    # ── Sidebar: clinical features ──
    st.sidebar.header("Clinical features")
    st.sidebar.caption("Fill in patient details for best accuracy.")

    clinical_vec = np.zeros(CLINICAL_DIM, dtype=np.float32)
    defaults = {
        'AGE': (40, 100, 70),
        'PTGENDER': (0, 1, 1),
        'PTEDUCAT': (0, 20, 12),
        'MMSE': (0, 30, 27),
        'CDRSB': (0.0, 18.0, 0.5),
        'ADAS11': (0.0, 70.0, 10.0),
        'Hippocampus': (2000.0, 10000.0, 6500.0),
        'Entorhinal': (500.0, 5000.0, 2500.0),
        'APOE4': (0, 2, 0),
    }
    for i, feat in enumerate(CLINICAL_FEATURES):
        if feat in defaults:
            lo, hi, default = defaults[feat]
            step = 1 if isinstance(lo, int) else 0.1
            val = st.sidebar.slider(feat, lo, hi, default, step=step)
            clinical_vec[i] = float(val)

    # Normalise clinical roughly (mean-center)
    clinical_means = np.array([70, 1, 14, 27, 0.5, 10, 6500, 2500, 0.3],
                               dtype=np.float32)
    clinical_stds  = np.array([8,  0.5, 3, 3, 1.5, 8, 800, 500, 0.6],
                               dtype=np.float32)
    clinical_norm  = (clinical_vec - clinical_means) / (clinical_stds + 1e-6)

    # ── Main: upload + predict ──
    st.header("Upload MRI scan")
    uploaded = st.file_uploader(
        "Supported formats: .nii, .nii.gz (raw NIFTI) or .npy (preprocessed)",
        type=['nii', 'gz', 'npy']
    )

    if uploaded is None:
        st.info("Upload an MRI scan to begin. "
                "Download a sample from ADNI or OASIS-3.")
        st.markdown("""
**Quick start with ADNI:**
1. Register at [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Download a T1 3T MRI scan as NIFTI
3. Upload the .nii.gz file here

**Or use preprocessed .npy:**
Run `python preprocessing.py` on your scan first.
        """)
        return

    if model is None:
        return

    with st.spinner("Loading and preprocessing MRI..."):
        volume = load_mri(uploaded)

    if volume is None:
        return

    st.success(f"MRI loaded — shape: {volume.shape}")

    # ── Inference ──
    with st.spinner("Running inference..."):
        mri_t  = torch.tensor(volume, dtype=torch.float32
                              ).unsqueeze(0).unsqueeze(0).to(device)
        clin_t = torch.tensor(clinical_norm, dtype=torch.float32
                              ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(mri_t, clin_t)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred   = int(np.argmax(probs))

    # ── Results ──
    st.header("Prediction")
    col1, col2 = st.columns([1, 2])

    with col1:
        css_cls = f"pred-{CLASSES[pred]}"
        desc = {
            'CN':  "Cognitively normal — no signs of Alzheimer's disease.",
            'MCI': "Mild Cognitive Impairment — early warning stage, monitor closely.",
            'AD':  "Alzheimer's Disease — significant cognitive decline detected."
        }[CLASSES[pred]]
        st.markdown(f"""
<div class="metric-box">
    <div class="pred-label {css_cls}">{CLASSES[pred]}</div>
    <div style="font-size:1.4rem; color:#ccc">{probs[pred]*100:.1f}% confidence</div>
    <div style="color:#888; margin-top:0.5rem; font-size:0.9rem">{desc}</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(confidence_chart(probs), use_container_width=True)

    # ── Grad-CAM ──
    st.header("Grad-CAM — brain region analysis")
    st.caption("Highlights which brain regions most influenced the prediction.")

    with st.spinner("Computing Grad-CAM..."):
        mri_t_grad = torch.tensor(volume, dtype=torch.float32
                                  ).unsqueeze(0).unsqueeze(0).to(device)
        clin_t_grad = torch.tensor(clinical_norm, dtype=torch.float32
                                   ).unsqueeze(0).to(device)
        gcam = GradCAM3D(model)
        cam, _, _ = gcam(mri_t_grad, clin_t_grad)
        gcam.remove_hooks()

    show_slices(volume, cam)

    # ── All-class CAM ──
    with st.expander("View Grad-CAM for all classes"):
        st.caption("Which regions matter for each diagnosis class?")
        with st.spinner("Generating all-class CAM..."):
            fig_all = visualise_all_classes(
                volume,
                torch.tensor(clinical_norm, dtype=torch.float32),
                model, device
            )
            st.pyplot(fig_all)
            plt.close()

    # ── Disclaimer ──
    st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is for research and educational purposes only.
    It is not a medical device and should not be used as the sole basis for clinical decisions.
    Always consult a qualified neurologist for diagnosis.
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
