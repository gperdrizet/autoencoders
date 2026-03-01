"""
Autoencoders Demo — Main Landing Page
"""

import streamlit as st

st.set_page_config(
    page_title="Autoencoders Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🧠 Autoencoders: Learning to Compress Reality")
st.markdown("*An interactive survey of what autoencoders can do — for bootcamp students*")

st.divider()

# ── Concept Introduction ──────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.header("What is an Autoencoder?")
    st.markdown("""
An autoencoder is a neural network trained to **compress** its input into a compact
representation and then **reconstruct** the original from that compressed form.

```
  Input  ──►  Encoder  ──►  Latent Space  ──►  Decoder  ──►  Output
(49,152)         │              (128)               │         (49,152)
                 └─── learns what matters ──────────┘
```

The magic happens in the **latent space** (bottleneck):
- The network *must* learn which features are important to keep
- Everything else is discarded
- This forces the model to understand the structure of the data

Unlike traditional compression (JPEG, ZIP), an autoencoder **learns its own rules**
from the data — no hand-crafted algorithms required.
    """)

with col2:
    st.header("Why Should You Care?")
    st.markdown("""
Autoencoders are a foundational concept that powers:

| Application | What it learns |
|---|---|
| 🗜️ **Compression** | What visual features matter most |
| 🔇 **Denoising** | Signal vs noise in data |

| 🎨 **Generation** (VAE) | The distribution of data |
| 🔍 **Representation Learning** | Meaningful features without labels |

The same core idea — compress then reconstruct — applies across **images, audio,
time series, text**, and more.
    """)

st.divider()

# ── Demo Cards ────────────────────────────────────────────────────────────────
st.header("Interactive Demos")
st.markdown("Select a demo from the sidebar, or click a card below:")

card1, card2 = st.columns(2)

with card1:
    st.subheader("🗜️ Image Compression")
    st.markdown("""
**384× compression ratio** using a 128-dimensional latent space.

Upload any image and watch the autoencoder compress it to 128 numbers, then
reconstruct it. Compare quality against standard JPEG.

- **Architecture**: 4-block CNN encoder/decoder
- **Dataset**: DF2K_OST high-quality images
- **Latent dim**: 128 values from 49,152 inputs
    """)
    st.page_link("pages/01-compression.py", label="Try Compression Demo →", icon="🗜️")

with card2:
    st.subheader("🔇 Image Denoising")
    st.markdown("""
**Trained to remove Gaussian noise** (σ=25) from corrupted images.

Add noise to an image and let the autoencoder restore it. The model has learned
what clean images look like and pushes noisy inputs toward that space.

- **Architecture**: Same CNN, larger latent (256)
- **Noise level**: σ=25 Gaussian
- **Metric**: PSNR improvement in dB
    """)
    st.page_link("pages/02-denoising.py", label="Try Denoising Demo →", icon="🔇")

st.divider()

# ── Architecture Explainer ────────────────────────────────────────────────────
with st.expander("📐 Architecture Details", expanded=False):
    st.markdown("""
### Compression & Denoising Autoencoder (Convolutional)

```
Encoder:
  Input (128×128×3)
  → Conv2D(64,  3×3, stride=2)  → BatchNorm → LeakyReLU  → 64×64×64
  → Conv2D(128, 3×3, stride=2)  → BatchNorm → LeakyReLU  → 32×32×128
  → Conv2D(256, 3×3, stride=2)  → BatchNorm → LeakyReLU  → 16×16×256
  → Conv2D(512, 3×3, stride=2)  → BatchNorm → LeakyReLU  → 8×8×512
  → Flatten → Dense(latent_dim)                           → 128

Decoder (mirror image):
  Dense(8×8×512) → Reshape(8,8,512)
  → ConvTranspose(512, 3×3, stride=2) → BatchNorm → LeakyReLU  → 16×16×512
  → ConvTranspose(256, 3×3, stride=2) → BatchNorm → LeakyReLU  → 32×32×256
  → ConvTranspose(128, 3×3, stride=2) → BatchNorm → LeakyReLU  → 64×64×128
  → ConvTranspose(64,  3×3, stride=2) → BatchNorm → LeakyReLU  → 128×128×64
  → Conv2D(3, 3×3, sigmoid)                                     → 128×128×3
```

    """)

# ── Dataset Info ──────────────────────────────────────────────────────────────
with st.expander("📊 Dataset Information", expanded=False):
    st.markdown("""
### DF2K_OST (Image demos)
- **Source**: DIV2K high-resolution image dataset
- **Processing**: Resized to 128×128 using Lanczos resampling
- **Size**: 900 images
- **Hosted**: [HuggingFace — gperdrizet/autoencoders](https://huggingface.co/datasets/gperdrizet/autoencoders)
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with TensorFlow · Streamlit · DF2K_OST | "
    "Models hosted on [HuggingFace](https://huggingface.co/gperdrizet/autoencoders)"
)
