# Autoencoders demo - workspace instructions

## Project overview

Educational demonstration of autoencoders for AI/ML bootcamp students. Focus on **pedagogical clarity** over implementation complexity — show what autoencoders can do, not every architectural detail.

**Current scope** (post Phase 1-2):
- Image compression (DF2K_OST dataset, 256×256 input, latent_dim=512)
- Image denoising (Gaussian noise removal)
- No anomaly detection (removed in Phase 1)

**Source of truth**: [PROJECT_PLAN.md](../PROJECT_PLAN.md) defines all specifications, architecture decisions, and phase completion status.

---

## Code style

### General patterns
- **Docstrings**: Google style with `Args:` and `Returns:` sections
- **Type hints**: Not required but use in function signatures when helpful
- **Variable names**: Descriptive over terse (`latent_dim` not `ld`)
- **Line length**: ~88 chars (Black-style) but not enforced

### Python conventions
```python
# Good: Clear, documented utility function
def load_df2k_ost(split='train', max_images=None):
    """
    Load DF2K_OST dataset from HuggingFace Hub.
    
    Args:
        split: 'train' or 'validation'
        max_images: Maximum number of images to load (None = all)
    
    Returns:
        numpy array of shape (N, 256, 256, 3) with values in [0, 1]
    """
```

### Keras model building
- **Named layers**: Always use `name=` parameter for debugging and visualization
- **Return tuples**: Model builders return `(autoencoder, encoder, decoder)` for flexibility
- **Print summaries**: Include informative prints (compression ratio, architecture stats)

Example from [src/model_utils.py](../src/model_utils.py#L10-L80):
```python
x = layers.Conv2D(64, 3, strides=2, padding='same', name='enc_conv1')(x)
x = layers.BatchNormalization(name='enc_bn1')(x)
x = layers.LeakyReLU(0.2, name='enc_relu1')(x)
```

---

## Architecture

### Directory structure
```
src/
  model_utils.py    # build_compression_ae(), build_denoising_ae()
  data_utils.py     # load_df2k_ost(), add_gaussian_noise()
  metrics.py        # calculate_psnr(), calculate_ssim(), compute_metrics()

pages/
  01-compression.py    # Streamlit demo for compression
  02-denoising.py      # Streamlit demo for denoising

notebooks/
  01-compression.ipynb    # Training notebook with explanations
  02-denoising.ipynb      # Training notebook with explanations

models/               # .keras files (downloaded from HuggingFace)
data/                 # HuggingFace datasets cache (auto-created)
logs/                 # TensorBoard event files
```

### Component responsibilities

**`src/model_utils.py`**: Pure model architecture definitions
- Returns `(autoencoder, encoder, decoder)` for all builders
- No training logic, no data loading
- Prints architecture summary when called

**`src/data_utils.py`**: Dataset loading and preprocessing
- HuggingFace dataset integration
- Noise augmentation for denoising
- Returns numpy arrays normalized to [0, 1]

**`src/metrics.py`**: Evaluation functions
- PSNR, SSIM, MSE calculations
- Batch-aware processing
- Uses `scikit-image` for consistency

**`pages/*.py`**: Streamlit demo apps
- Self-contained (imports from `src/`)
- Model loading with `@st.cache_resource`
- HuggingFace download fallback if local model missing

**`notebooks/*.ipynb`**: Training workflows
- Markdown explanations between code cells
- `TRAIN_MODEL` flag to skip training and load pre-trained
- Save models locally and upload to HuggingFace

---

## Build and test

### Initial setup
```bash
# Install dependencies
pip install -r requirements.txt
```

Models and datasets download automatically from HuggingFace on first use. No configuration needed for students.

### Run Streamlit demos
```bash
streamlit run app.py
```
Opens on <http://localhost:8501>. Navigate demos via sidebar.

### Run training notebooks
```bash
jupyter notebook
```
Open `notebooks/01-compression.ipynb` or `notebooks/02-denoising.ipynb`.

Set `TRAIN_MODEL = False` at top of notebook to skip training and use pre-trained model.

### Manual model training (advanced)
Training happens in notebooks, but you can also run:
```python
from src.model_utils import build_compression_ae
from src.data_utils import load_df2k_ost

autoencoder, encoder, decoder = build_compression_ae(latent_dim=512, input_shape=(256, 256, 3))
train_images = load_df2k_ost(split='train')
# ...training loop...
autoencoder.save('models/compression_ae.keras')
```

---

## Project conventions

### HuggingFace integration
- **Hardcoded repository IDs** (no user configuration):
  - Dataset: `gperdrizet/DF2K_OST`
  - Models: `gperdrizet/compression_autoencoder`, `gperdrizet/denoising_autoencoder`
- **Automatic downloads**: Models/datasets download on first use and cache locally
- **No authentication required**: Students only need to download public assets

### Model file naming
Post Phase 1-2, use simple names without dimensions:
- `compression_ae.keras`
- `denoising_ae.keras`
- Not: `compression_ae_latent512.keras` (old convention)

### Image specifications
- **Input resolution**: 256×256×3 (changed from 128×128 in Phase 2)
- **Value range**: [0, 1] float32 (not uint8)
- **Color space**: RGB only
- **Resampling**: Use `Image.LANCZOS` for resizing

### Quality philosophy
> "Visual fidelity over compression ratio"

- **Primary evaluation**: Look at reconstructed images
- **Secondary metrics**: PSNR (>30dB guide) and SSIM (>0.90 guide)
- Don't chase metrics at the expense of visual quality
- See [PROJECT_PLAN.md](../PROJECT_PLAN.md#objective-3-complete-compression-and-denoising-autoencoders) for details

### Training configuration
Standard training setup (adjust as needed):
```python
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    keras.callbacks.ModelCheckpoint('models/best_model.keras', save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='logs/')
]

history = autoencoder.fit(
    train_images, train_images,
    validation_data=(val_images, val_images),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
```

### Streamlit page pattern
All demo pages follow this structure:
1. **Config**: `st.set_page_config()`
2. **Imports**: Add project root to `sys.path`, load `.env`
3. **Model loader**: `@st.cache_resource` function with HF download fallback
4. **Helper functions**: Preprocessing, postprocessing
5. **UI**: Title, instructions, file uploader, comparison views

Example from [pages/01-compression.py](../pages/01-compression.py#L30-L55)

---

## Integration points

### External dependencies
- **HuggingFace Hub**: Dataset and model storage
- **TensorFlow/Keras**: Model training and inference
- **Streamlit**: Web demo framework
- **scikit-image**: Metrics (PSNR, SSIM)

### Dataset: DF2K_OST
Combined dataset from DIV2K + Flickr2K + OST:
- ~13,874 images total
- 256×256 pixels (Lanczos resampled)
- 90/10 train/validation split
- Loaded via `datasets.load_dataset("gperdrizet/DF2K_OST")`

---

## Development notes

### When adding new features
1. Check [PROJECT_PLAN.md](../PROJECT_PLAN.md) for current phase and specifications
2. Keep pedagogical focus — students should understand *why* not just *how*
3. Add docstrings and comments explaining design decisions
4. Test both notebooks and Streamlit demos
5. Update README.md if user-facing changes

### When modifying models
1. Change architecture in `src/model_utils.py` first
2. Update corresponding notebook (`notebooks/*.ipynb`)
3. Retrain and evaluate visually before checking metrics
4. Upload new model to HuggingFace if satisfactory
5. Update Streamlit demo if inference code changes

### When debugging
- **TensorBoard logs**: `tensorboard --logdir logs/`
- **Model summaries**: Call `model.summary()` after building
- **Image values**: Always check `arr.min()`, `arr.max()`, `arr.dtype` — many bugs are normalization issues
- **Cache issues**: Delete `data/` and `.streamlit/cache/` if strange behavior

### Common pitfalls
- uint8 images not normalized leads to poor training
- Wrong input shape (128 vs 256) causes model mismatch
- Forgetting batch dimension causes inference errors
- JPEG artifacts in comparisons, use PNG for demos
- Metrics look good but visual quality poor, trust your eyes

---

## Quick reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run web demos | `streamlit run app.py` |
| Open training notebooks | `jupyter notebook` |
| View TensorBoard logs | `tensorboard --logdir logs/` |
| Check current phase | Read [PROJECT_PLAN.md](../PROJECT_PLAN.md) |
| Model architecture | See [src/model_utils.py](../src/model_utils.py) |
| Dataset loading | See [src/data_utils.py](../src/data_utils.py) |
