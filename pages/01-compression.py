"""
Streamlit demo page: Image Compression with Autoencoders
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
    page_title="Image Compression | Autoencoders Demo",
    page_icon="",
    layout="wide",
)

# ── Project imports ───────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading compression model…")
def load_model():
    """Download model from HuggingFace (cached after first load)."""
    import tensorflow as tf
    from tensorflow import keras
    from huggingface_hub import hf_hub_download

    hf_repo_id = os.getenv("COMPRESSION_AE_REPO", "gperdrizet/compression_autoencoder")
    hf_token   = os.getenv("HF_TOKEN", None)
    model_name = "compression_ae.keras"

    # Try local model first
    local_path = project_root / "models" / model_name
    if local_path.exists():
        return keras.models.load_model(str(local_path))

    # Download from HuggingFace
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


def preprocess_image(img: Image.Image, size: int = 256) -> np.ndarray:
    """Resize and normalise a PIL image to a (1, size, size, 3) float32 array."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis]  # add batch dimension


def postprocess(arr: np.ndarray) -> Image.Image:
    """Convert a (1, H, W, 3) float32 array back to a PIL Image."""
    arr = np.clip(arr[0], 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── PSNR helper (no skimage dependency in demo) ───────────────────────────────
def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" Image Compression with Autoencoders")

st.markdown("""
An autoencoder learns to compress your image into a compact representation, then
reconstructs it. All learned automatically from training data, no hand-crafted rules.
""")

st.divider()

# Sidebar controls
with st.sidebar:
    st.header(" Settings")
    st.markdown("**Model**")
    st.caption("Pre-trained on DF2K_OST (~13,800 images @ 256×256)")
    st.caption("Convolutional AE with 5 encoder/decoder blocks")

# Load model
model = load_model()

if model is None:
    st.error("Model unavailable. Make sure the model is trained and available.")
    st.stop()

# ── Image input ───────────────────────────────────────────────────────────────
st.subheader("Upload an Image")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with col_sample:
    st.markdown("**Or use a sample image:**")
    sample_choice = st.selectbox(
        "Sample images",
        ["None", "Sample from training set"],
        label_visibility="collapsed",
    )

# Determine source image
source_img = None

if uploaded is not None:
    source_img = Image.open(uploaded)
elif sample_choice != "None":
    # Load a sample from the cached dataset
    from src.data_utils import load_df2k_ost
    with st.spinner("Loading sample image…"):
        try:
            images = load_df2k_ost(
                split='train',
                max_images=5,
            )
            source_img = Image.fromarray((images[0] * 255).astype(np.uint8))
        except Exception as e:
            st.warning(f"Could not load sample image: {e}")

# ── Compression & display ─────────────────────────────────────────────────────
if source_img is not None:
    st.divider()

    input_arr = preprocess_image(source_img)
    with st.spinner("Compressing…"):
        reconstructed_arr = model.predict(input_arr, verbose=0)

    recon_img   = postprocess(reconstructed_arr)
    score_psnr  = psnr(input_arr, reconstructed_arr)

    # ── Side-by-side images ────────────────────────────────────────────────
    col_orig, col_recon = st.columns(2)

    with col_orig:
        st.subheader("Original (resized to 256×256)")
        st.image(source_img.resize((512, 512), Image.NEAREST), use_container_width=True)
        orig_bytes = image_to_bytes(source_img.resize((256, 256), Image.LANCZOS))
        st.download_button(" Download original", orig_bytes, "original.png", "image/png")

    with col_recon:
        st.subheader("Reconstructed")
        st.image(recon_img.resize((512, 512), Image.NEAREST), use_container_width=True)
        recon_bytes = image_to_bytes(recon_img)
        st.download_button(" Download reconstructed", recon_bytes, "reconstructed.png", "image/png")

    # ── Metrics ────────────────────────────────────────────────────────────
    st.subheader("Quality Metrics")

    st.metric("PSNR", f"{score_psnr:.1f} dB",  help="Peak Signal-to-Noise Ratio. Higher is better. >30 dB is excellent.")

    # Quality interpretation
    if score_psnr >= 35:
        st.success(f"Excellent quality ({score_psnr:.1f} dB) - very hard to see differences.")
    elif score_psnr >= 30:
        st.success(f"Good quality ({score_psnr:.1f} dB) - minor artefacts may be visible.")
    elif score_psnr >= 25:
        st.warning(f"Fair quality ({score_psnr:.1f} dB) - noticeable blurring, but structure is preserved.")
    else:
        st.error(f"Low quality ({score_psnr:.1f} dB) - significant loss at this compression.")

    # ── Pixel difference ───────────────────────────────────────────────────
    with st.expander("🔍 Pixel Difference Map"):
        import matplotlib.pyplot as plt

        orig_arr  = input_arr[0]
        diff      = np.abs(orig_arr - reconstructed_arr[0])
        diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_arr)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(reconstructed_arr[0])
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        im = axes[2].imshow(diff.mean(axis=2), cmap="hot", vmin=0)
        axes[2].set_title("Absolute Difference (mean over channels)")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046)

        st.pyplot(fig)
        plt.close(fig)

else:
    # Placeholder
    st.info("👆 Upload an image or select a sample to see compression in action.")

    with st.expander("How does it work?"):
        st.markdown("""
**1. Encoder** (compression):
The encoder passes the image through 5 convolutional layers, each halving the spatial
dimensions: 256→128→64→32→16→8. A final Dense layer compresses the 8×8×512 feature map
into just **4096 numbers**.

**2. Latent space** (the bottleneck):
These 4096 numbers encode the entire image. The network was forced to decide what to keep
and what to discard - learning that edges, textures, and colours matter; exact pixel
values don't.

**3. Decoder** (reconstruction):
The decoder uses transposed convolutions to rebuild the image from those 4096 numbers,
producing a visually similar 256×256×3 output.
        """)
