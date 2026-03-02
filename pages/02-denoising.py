"""
Streamlit demo page: Image Denoising with Autoencoders
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Denoising | Autoencoders Demo",
    page_icon="",
    layout="wide",
)

# ── Project imports ───────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading denoising model…")
def load_model():
    import tensorflow as tf
    from tensorflow import keras
    from huggingface_hub import hf_hub_download

    hf_repo_id = os.getenv("HF_REPO_ID", "gperdrizet/autoencoders")
    hf_token   = os.getenv("HF_TOKEN", None)
    model_name = "denoising_ae_sigma25.keras"

    local_path = project_root / "models" / model_name
    if local_path.exists():
        return keras.models.load_model(str(local_path))

    try:
        model_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"models/{model_name}",
            repo_type="model",
            token=hf_token,
        )
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


def preprocess_image(img: Image.Image, size: int = 128) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]


def add_noise(arr: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma / 255.0, arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0, 1)


def postprocess(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr[0], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" Image Denoising with Autoencoders")

st.markdown("""
A denoising autoencoder is trained on **pairs of (noisy, clean)** images.
It learns to push corrupted inputs back toward the manifold of clean images -
effectively learning what noise looks like and how to remove it.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header(" Settings")
    noise_level = st.slider(
        "Noise level (σ)",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Gaussian noise standard deviation on 0–255 scale. Model was trained at σ=25.",
    )
    if noise_level != 25:
        st.info(
            f" Model was trained at σ=25. "
            f"At σ={noise_level} you may see {'better' if noise_level < 25 else 'worse'} results."
        )
    st.divider()
    st.markdown("**Model**")
    st.caption("Denoising AE - latent dim 256")
    st.caption("Trained on DF2K_OST, σ=25 Gaussian noise")

# Load model
model = load_model()
if model is None:
    st.error("Model unavailable.")
    st.stop()

# ── Image input ───────────────────────────────────────────────────────────────
st.subheader("Upload an Image")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with col_sample:
    st.markdown("**Or use a sample:**")
    use_sample = st.button("Load sample image")

source_img = None

if uploaded is not None:
    source_img = Image.open(uploaded)
elif use_sample:
    from src.data_utils import load_df2k_ost
    with st.spinner("Loading sample…"):
        try:
            images = load_df2k_ost(
                image_size=128,
                max_images=5,
                cache_dir=project_root / "data" / "df2k_ost_128",
            )
            idx = np.random.randint(len(images))
            source_img = Image.fromarray((images[idx] * 255).astype(np.uint8))
        except Exception as e:
            st.warning(f"Could not load sample: {e}")

# ── Denoising & display ───────────────────────────────────────────────────────
if source_img is not None:
    st.divider()

    np.random.seed(42)  # reproducible noise
    clean_arr  = preprocess_image(source_img)
    noisy_arr  = add_noise(clean_arr, sigma=noise_level)

    with st.spinner("Denoising…"):
        denoised_arr = model.predict(noisy_arr, verbose=0)

    clean_img   = postprocess(clean_arr)
    noisy_img   = postprocess(noisy_arr)
    denoised_img = postprocess(denoised_arr)

    psnr_noisy    = psnr(clean_arr, noisy_arr)
    psnr_denoised = psnr(clean_arr, denoised_arr)
    improvement   = psnr_denoised - psnr_noisy

    # ── Three-column display ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    display_size = (384, 384)

    with col1:
        st.subheader("Clean Original")
        st.image(clean_img.resize(display_size, Image.NEAREST), use_container_width=True)
        st.caption("Original image (resized to 128×128)")

    with col2:
        st.subheader(f"Noisy (σ={noise_level})")
        st.image(noisy_img.resize(display_size, Image.NEAREST), use_container_width=True)
        st.caption(f"After adding Gaussian noise - PSNR: {psnr_noisy:.1f} dB")

    with col3:
        st.subheader("Denoised")
        st.image(denoised_img.resize(display_size, Image.NEAREST), use_container_width=True)
        st.caption(f"After autoencoder denoising - PSNR: {psnr_denoised:.1f} dB")

    # ── Metrics ────────────────────────────────────────────────────────────
    st.subheader("Quality Metrics")

    m1, m2, m3 = st.columns(3)
    m1.metric("Noisy PSNR",    f"{psnr_noisy:.1f} dB",    help="Quality after adding noise")
    m2.metric("Denoised PSNR", f"{psnr_denoised:.1f} dB", delta=f"+{improvement:.1f} dB", help="Quality after denoising")
    m3.metric("Improvement",   f"+{improvement:.1f} dB",  help="PSNR gained by denoising")

    if psnr_denoised >= 30:
        st.success(f"Excellent denoising result ({psnr_denoised:.1f} dB)!")
    elif psnr_denoised >= 28:
        st.success(f"Good denoising result ({psnr_denoised:.1f} dB - above target of 28 dB)")
    elif psnr_denoised >= 25:
        st.warning(f"Fair result ({psnr_denoised:.1f} dB). Model was trained at σ=25 - try that noise level.")
    else:
        st.error(f"Poor result ({psnr_denoised:.1f} dB). Very high noise levels exceed the model's training range.")

    # ── Noise level comparison chart ───────────────────────────────────────
    with st.expander(" Performance across noise levels"):
        import matplotlib.pyplot as plt

        test_arr = preprocess_image(source_img)
        sigmas, psnrs_noisy, psnrs_denoised = [], [], []

        np.random.seed(0)
        for s in range(5, 105, 10):
            n = add_noise(test_arr, sigma=s)
            d = np.clip(model.predict(n, verbose=0), 0, 1)
            sigmas.append(s)
            psnrs_noisy.append(psnr(test_arr, n))
            psnrs_denoised.append(psnr(test_arr, d))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sigmas, psnrs_noisy,    "o-", label="Noisy input",   color="tomato")
        ax.plot(sigmas, psnrs_denoised, "s-", label="Denoised",       color="steelblue")
        ax.axvline(25, color="gray", linestyle="--", alpha=0.7, label="Training noise (σ=25)")
        ax.axhline(28, color="green", linestyle=":", alpha=0.7, label="Target (28 dB)")
        ax.set_xlabel("Noise level (σ)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Denoising Quality vs Noise Level")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ── Download ───────────────────────────────────────────────────────────
    st.download_button(
        " Download denoised image",
        image_to_bytes(denoised_img),
        "denoised.png",
        "image/png",
    )

else:
    st.info("👆 Upload an image or load a sample to try denoising.")

    with st.expander("How does denoising work?"):
        st.markdown("""
**Training:**
The model sees thousands of `(noisy_image, clean_image)` pairs. For each pair,
it tries to reconstruct the clean image from the noisy input. Over time it learns:
- Which patterns are signal (consistent structure in clean images)
- Which patterns are noise (random, unpredictable)

**Inference:**
When given a new noisy image, the encoder maps it into the latent space. Because
the latent space was shaped by **clean** images during training, the decoder
produces a cleaner reconstruction - noise doesn't project cleanly onto the learned manifold.

**Why does it work beyond σ=25?**
The model generalises somewhat, but performance degrades for noise levels far outside
its training distribution. This is a feature, not a bug - it shows the model
learned something meaningful, not just memorisation.
        """)
