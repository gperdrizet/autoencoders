# Autoencoders demo - project plan

## Project overview

### Goal
Create an educational demonstration of autoencoders for bootcamp students, focusing on a **survey of applications** rather than depth of encoder/decoder implementation details. Show the power of learned latent representations across multiple use cases.

### Key philosophy
- **Pedagogical focus**: Help students understand the core concept of learned latent space representation
- **Application survey**: Show how autoencoders can be adapted to a range of problems
- **Diversity**: Use multiple data types across demonstrations, e.g. images, text, tabular
- **Interactivity**: Create demos that student's can actually use, i.e. student can upload their own images for compression, but are unlikely to have ECG traces to upload for analysis

---

## Technical Foundation

Training datasets and trained models should be made available to the ML community:

- Dataset: https://huggingface.co/datasets/gperdrizet/DF2K_OST
- Models:
   - https://huggingface.co/gperdrizet/compression_autoencoder
   - https://huggingface.co/gperdrizet/denoising_autoencoder

### Dataset: DF2K_OST

DF2K_OST is a combined image dataset commonly used for image superresolution benchmarks. It is composed of 3 datasets: DIV2K, Flickr2K and OST datasets.

- **Sources**: Raw data is available locally under `/data/raw/` but need to be preprocessed an uploaded to HuggingFace
- **Processing**: Resized to 256×256 pixels using Lanczos resampling
- **HuggingFace repository**: https://huggingface.co/datasets/gperdrizet/DF2K_OST

### Development Strategy
1. **Start simple**
2. **Get excellent performance first**, then expand if needed
3. **No premature optimization**: Skip connections, VAE, etc. only if basic approach fails

---

## MINIMAL WORKING STATE PLAN

**Target Scope**: Compression and Denoising demonstrations only  
**Status**: Draft - Created February 27, 2026

### Objective 1: Remove ECG Demo and Legacy Code

**Philosophy:** Clean removal without archiving. No backwards compatibility or version references.

#### 1.1 Delete Anomaly Detection Components
**Files to delete:**
- `pages/03-anomaly-detection.py`
- `notebooks/03-anomaly-detection.ipynb`
- Any anomaly detection models in `models/` directory

#### 1.2 Clean Up Web App
**File: `app.py`**
- Remove "03 - Anomaly Detection" navigation link
- Remove ECG or anomaly detection references from landing page content

#### 1.3 Clean Up Source Code
**File: `src/model_utils.py`**
- Remove `build_anomaly_detection_ae()` function
- Update module docstring (compression + denoising only)

**File: `src/data_utils.py`**
- Remove ECG5000, time-series, or anomaly detection functions
- Keep only image-related utilities
- Update module docstring

#### 1.4 Clean Up Requirements
**Files: `requirements.txt`, `requirements-cloud.txt`**
- Remove `tensorflow-datasets` (was only for ECG5000)
- Keep all image processing and core ML dependencies

---

### Objective 2: Complete the DF2K_OST Dataset

**Current raw data status:**
- DIV2K: 7.5GB available ✓
- OST: 4.1GB available ✓  
- Flickr2K: 667MB (download in progress)

**Target specifications:**
- Image resolution: **256×256 pixels**
- Resampling method: Lanczos
- Format: JPEG or PNG
- Repository: `gperdrizet/DF2K_OST` (credentials in `.env`)

#### 2.1 Verify Raw Data Completeness
**Expected counts:**
- DIV2K: 800 training + 100 validation = 900 images
- Flickr2K: 2,650 images
- OST: 10,324 images  
- **Total: ~13,874 images**

**Actions:**
1. Count actual images in each raw directory
2. Complete Flickr2K download if needed
3. Verify no corrupted files

#### 2.2 Create Dataset Processing Script
**Create: `scripts/build_df2k_ost_dataset.py` (one-time use, not in final product)**

Script should:
1. Load all images from `data/raw/DIV2K/`, `data/raw/Flickr2K/`, `data/raw/OST/`
2. Resize to 256×256 using Lanczos resampling
3. Create 90/10 train/validation split (stratified by source if desired)
4. Save in HuggingFace datasets format
5. Push to `gperdrizet/DF2K_OST` using token from `.env`
6. Generate dataset card with:
   - Source attribution (DIV2K, Flickr2K, OST)
   - Processing details (256×256 Lanczos)
   - Train/validation split info
   - License information
   - Usage examples

**Script can be deleted after successful upload**

#### 2.3 Integrate Dataset Loading into Source Code
**File: `src/data_utils.py`**

Add function to load dataset from HuggingFace:
```python
def load_df2k_ost_dataset(split='train'):
    """Load DF2K_OST dataset from HuggingFace Hub.
    
    Args:
        split: 'train' or 'validation'
        
    Returns:
        Dataset with 256×256 images
    """
    # Hardcode repo: 'gperdrizet/DF2K_OST'
    # No token needed for public dataset
    # Use datasets.load_dataset()
    # Return numpy arrays or PIL images
```

**Requirements:**
- Hardcode repository ID (no user input needed)
- No authentication required (public dataset)
- Automatic caching on first download
- Clear error messages if download fails

---

### Objective 3: Complete Compression and Denoising Autoencoders

**Core specifications:**
- Input resolution: **256×256×3** = 196,608 values
- Latent dimension: **≥256** (for ≤768× compression)
- Priority: **Visual fidelity over compression ratio**
- Evaluation: **Qualitative first** (images must look good), quantitative metrics secondary

#### 3.1 Compression Autoencoder

**Architecture decisions:**
- Start with `latent_dim=512` (384× compression)
- If reconstruction quality is poor, increase to 1024 or 2048
- Target: Reconstructed images look nearly identical to originals

**Training approach:**
1. **Update `notebooks/01-compression.ipynb`:**
   - Set `INPUT_SIZE = 256` (not 128)
   - Set `LATENT_DIM = 512` (starting point)
   - Train and evaluate visually
   - If quality is poor, increase latent_dim and retrain

2. **Qualitative evaluation priorities:**
   - No visible artifacts or blurring
   - Sharp edges preserved
   - Color accuracy maintained
   - Details in textures retained
   - **Visual inspection is the primary success criterion**

3. **Quantitative metrics (secondary):**
   - PSNR (guide: >30 dB is good, but trust your eyes)
   - SSIM (guide: >0.90 is good, but trust your eyes)
   - Don't optimize for metrics at expense of visual quality

4. **Model naming:**
   - Save as `models/compression_ae.keras` (simple, no dimension in filename)
   - Delete old latent128/latent512 models

**Training configuration:**
- Epochs: 50-100 (use early stopping)
- Batch size: 16-32 (depending on GPU memory)
- Learning rate: 1e-3 with ReduceLROnPlateau
- Callbacks: EarlyStopping (patience=10), ModelCheckpoint, ReduceLROnPlateau

#### 3.2 Denoising Autoencoder

**Architecture decisions:**
- Start with `latent_dim=512` (same as compression)
- Noise level: σ=25 Gaussian
- May need larger latent than compression for good denoising

**Training approach:**
1. **Update `notebooks/02-denoising.ipynb`:**
   - Set `INPUT_SIZE = 256`
   - Set `LATENT_DIM = 512` (starting point)
   - Generate noisy/clean pairs
   - Train and evaluate visually

2. **Qualitative evaluation priorities:**
   - Noise effectively removed
   - No excessive smoothing or detail loss
   - Natural appearance (not over-processed)
   - **Visual comparison with noisy input is primary criterion**

3. **Quantitative metrics (secondary):**
   - PSNR vs noisy input (guide: >25 dB improvement)
   - SSIM (guide: >0.85)
   - Metrics are guides, not targets

4. **Model naming:**
   - Save as `models/denoising_ae.keras` (simple, no sigma in filename)

**Training configuration:**
- Epochs: 50-100 (use early stopping)
- Batch size: 16-32
- Learning rate: 1e-3 with ReduceLROnPlateau
- Same callback strategy as compression

#### 3.3 Model Upload to HuggingFace Hub

**Target repositories:**
- Create/update: `gperdrizet/compression_autoencoder`
- Create/update: `gperdrizet/denoising_autoencoder`
- Use token from `.env` file

**For each model:**
1. Upload `.keras` file
2. Create model card with:
   - Architecture description
   - Input/output specs (256×256×3)
   - Training details
   - Example usage code
   - Sample inference results
3. Add config file if needed

**Code integration:**
- Update model loading in `src/model_utils.py` or relevant files
- Hardcode repository IDs (no user input)
- No authentication for downloading (public models)
- Auto-download on first use with caching

#### 3.4 Iterative Quality Improvement

**If initial results are poor:**

**Compression issues:**
- Blurry reconstructions → Increase latent_dim (768, 1024, 2048)
- Artifacts → Adjust architecture (more layers, different activation)
- Poor color → Check normalization, loss function
- **Priority: Visual quality over compression ratio**

**Denoising issues:**
- Incomplete noise removal → Increase latent_dim
- Over-smoothing → Reduce epochs, adjust loss
- Lost details → Try perceptual loss, adjust architecture

**Architecture modifications if needed:**
- Add more convolutional blocks
- Increase filter counts
- Adjust layer depths
- Consider residual connections (only if needed)

---

### Objective 4: Documentation Audit and Updates

**Core specifications to propagate:**
- Image resolution: 256×256×3
- Two demonstrations: compression and denoising only
- No anomaly detection references
- No unit testing

#### 4.1 Update README.md

**Remove:**
- All anomaly detection / ECG references
- References to "3 demonstrations"
- Old image dimensions (128×128)
- Testing instructions

**Update:**
1. **Project description:**
   - "Two demonstrations" of autoencoders
   - Compression and denoising only

2. **Dataset information:**
   - Image size: 256×256 pixels
   - Dataset: DF2K_OST (~13,874 images from DIV2K, Flickr2K, OST)
   - 90/10 train/validation split
   - HuggingFace: `gperdrizet/DF2K_OST`

3. **Demonstrations:**
   - **Compression:** 256×256×3 → latent_dim → 256×256×3
   - Include compression ratio (depends on final latent_dim)
   - Focus on visual quality description
   - **Denoising:** σ=25 Gaussian noise removal
   - Emphasize visual before/after comparison

4. **Models:**
   - Two models only: compression_ae.keras, denoising_ae.keras
   - HuggingFace repos: `gperdrizet/compression_autoencoder`, `gperdrizet/denoising_autoencoder`
   - Auto-download on first use

5. **Setup instructions:**
   - Verify commands work
   - Streamlit launch instructions
   - Remove test commands

#### 4.2 Update app.py Landing Page

**Content updates:**
- Remove "3 demonstrations" → "2 demonstrations"
- Update any introductory text about autoencoders
- Remove navigation to page 3
- Remove ECG/anomaly mentions

**Technical content:**
- Update image dimensions in examples
- Update compression ratios if mentioned
- Verify links to documentation

#### 4.3 Update Notebooks

**Both notebooks (`01-compression.ipynb`, `02-denoising.ipynb`):**

Update markdown cells to:
- Reference 256×256 input images
- Remove references to other demonstrations
- Update dataset loading documentation
- Update model architecture descriptions
- Update performance metrics (PSNR/SSIM as guides, not targets)
- Add emphasis on visual quality evaluation

**Code cells:**
- Update `INPUT_SIZE = 256`
- Update model paths (simple names: compression_ae.keras, denoising_ae.keras)
- Update dataset loading to use HuggingFace
- Verify all imports and dependencies

#### 4.4 Update Source Code Documentation

**File: `src/data_utils.py`**
- Module docstring: compression and denoising only
- Function docstrings: 256×256 image specifications
- Remove ECG/time-series references
- Add HuggingFace dataset loading function

**File: `src/model_utils.py`**
- Module docstring: two autoencoders (compression, denoising)
- Remove `build_anomaly_detection_ae()` function
- Update architecture documentation for 256×256 inputs
- Clear parameter descriptions

**File: `src/metrics.py`**
- Remove unused metrics if any
- Document PSNR and SSIM as evaluation guides
- Emphasize qualitative over quantitative

**File: `pages/*.py`**
- Update page titles and descriptions
- Reference correct input dimensions
- Update model loading paths

#### 4.5 Update PROJECT_PLAN.md

**Technical Foundation section:**
- Image size: 256×256
- Dataset: ~13,874 images (DIV2K + Flickr2K + OST)
- Two models only
- Remove anomaly autoencoder references

**Development Strategy:**
- Visual quality first approach
- Metrics as secondary validation
- Iterative improvement based on visual inspection

#### 4.6 Configuration Alignment

**Create consistency across all files:**

| Parameter | Value | Files to Check |
|-----------|-------|----------------|
| Image size | 256×256×3 | README, notebooks, src/, app.py, pages/ |
| Compression latent_dim | 512+ (TBD) | Notebook 01, src/model_utils.py, pages/01 |
| Denoising latent_dim | 512+ (TBD) | Notebook 02, src/model_utils.py, pages/02 |
| Noise sigma | 25 | Notebook 02, pages/02, README |
| Dataset source | gperdrizet/DF2K_OST | src/data_utils.py, notebooks, README |
| Model repos | gperdrizet/compression_autoencoder, gperdrizet/denoising_autoencoder | All model loading code, README |

**Verification checklist:**
- [ ] No references to 128×128 images
- [ ] No references to anomaly detection or ECG
- [ ] No references to 3 demonstrations
- [ ] All model paths use simple names (no dimensions in filename)
- [ ] All dataset loading uses HuggingFace repo
- [ ] No unit testing infrastructure
- [ ] Visual quality emphasized over metrics

---

## Execution Order

**Phase 1: Cleanup (1 day)**
1. Delete anomaly detection page, notebook
2. Remove anomaly detection code from source files
3. Remove unused dependencies

**Phase 2: Dataset (2-3 days)**
1. Verify raw data completeness (complete Flickr2K if needed)
2. Create and run dataset processing script
3. Upload to HuggingFace with dataset card
4. Integrate dataset loading into `src/data_utils.py`
5. Test dataset download and loading

**Phase 3: Model Training (3-7 days)**
1. Train compression autoencoder (start with latent_dim=512)
2. Evaluate visually - adjust latent_dim if needed
3. Train denoising autoencoder (start with latent_dim=512)
4. Evaluate visually - adjust latent_dim if needed
5. Upload both models to HuggingFace with model cards
6. Integrate auto-download into model loading code
7. Test models in web app

**Phase 4: Documentation (1-2 days)**
1. Update README.md
2. Update PROJECT_PLAN.md
3. Update all notebook markdown cells
4. Update source code docstrings
5. Update web app content
6. Verify configuration consistency

**Phase 5: End-to-End Validation (1 day)**
1. Fresh clone and setup
2. Run both notebooks end-to-end
3. Launch and test web app
4. Verify all models/datasets auto-download
5. Visual quality check on multiple test images

**Total estimated time: 8-14 days**

---

## Success Criteria

The minimal working state is achieved when:

1. **Code is clean:**
   - No anomaly detection references anywhere
   - Only compression and denoising code remains
   - No broken imports or unused dependencies

2. **Dataset works:**
   - DF2K_OST available on HuggingFace
   - Auto-downloads on first use
   - All images are 256×256×3
   - Train/validation split correct

3. **Models work:**
   - Compression model produces visually good reconstructions
   - Denoising model effectively removes noise
   - Both models auto-download from HuggingFace
   - Inference works in notebooks and web app

4. **Web app works:**
   - Two functional demos (compression, denoising)
   - Upload and process images correctly
   - Display results with visual comparisons
   - Metrics shown (but not emphasized)
   - No broken links or errors

5. **Notebooks work:**
   - Both notebooks run end-to-end without errors
   - Training completes successfully
   - Visual results look good
   - Models save correctly

6. **Documentation is accurate:**
   - README matches actual implementation
   - All specifications consistent (256×256, 2 demos, etc.)
   - Setup instructions work from scratch
   - No references to removed features

---

## Future Enhancements (Out of Scope for MVP)

Items explicitly deferred:

- Anomaly detection demonstration
- Additional autoencoder variants (VAE, β-VAE, etc.)
- Other data modalities (text, audio, tabular)
- Advanced features (latent interpolation, style transfer)
- Unit testing infrastructure
- Deployment automation
- Performance optimization (quantization, etc.)

---

**Plan Status:** Draft - Updated with stakeholder requirements  
**Created:** February 27, 2026  
**Next Steps:** Begin Phase 1 cleanup