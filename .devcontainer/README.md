# Dual DevContainer Configuration: GPU & CPU

This project supports both GPU and CPU-only development environments using VS Code Dev Containers.

## GPU Configuration
- Uses `gperdrizet/deeplearning-gpu` Docker image
- Requires Linux/WSL2, NVIDIA drivers ≥545, Docker with GPU support
- All dependencies (CUDA, cuDNN, PyTorch, TensorFlow) pre-installed

## CPU Configuration
- Uses `mcr.microsoft.com/devcontainers/python:3.10` Docker image
- Works on any machine (Mac, Windows, Linux)
- PyTorch and TensorFlow CPU-only

## How to Use
1. Open the project in VS Code
2. When prompted, select either:
   - **Autoencoders demo** (GPU)
   - **Autoencoders demo (CPU)**
3. Click "Reopen in Container"

## Ports
- 8501: Streamlit
- 6006: TensorBoard

## Customization
Both containers provide identical development environments except for hardware support.

## More Info
See [CIFAR10 DevContainer Guide](https://github.com/gperdrizet/CIFAR10/blob/main/.devcontainer/README.md) for reference.
