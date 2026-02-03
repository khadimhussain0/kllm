# Setup Guide

Detailed installation instructions for KLLM with Blackwell (RTX 50 series) and other NVIDIA GPUs.

## System Requirements

- Ubuntu 22.04+ or compatible Linux
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.x drivers installed
- Python 3.12

## Verified Working Configuration

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 (Linux 6.14) |
| GPU | RTX 5090 (32GB) |
| Driver | 580.126.09 |
| CUDA | 13.0 (driver) / 12.8 (toolkit) |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| vLLM | 0.15.0 |
| transformers | 4.57.3 |
| Unsloth | 2026.1.4 |

## Installation Steps

### 1. Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment

```bash
uv venv .venv --python=3.12 --seed
source .venv/bin/activate
```

### 3. Install PyTorch and vLLM

For **Blackwell / RTX 50 series** (CUDA 12.8):
```bash
uv pip install -U vllm --torch-backend=cu128
```

For **Ada / RTX 40 series** (CUDA 12.1):
```bash
uv pip install -U vllm --torch-backend=cu121
```

This installs PyTorch with the correct CUDA backend automatically.

### 4. Install Unsloth

```bash
uv pip install unsloth unsloth_zoo bitsandbytes
```

### 5. Pin transformers Version

**Critical**: vLLM 0.15.0 requires exactly transformers 4.57.3. Other versions cause import errors.

```bash
uv pip install --force-reinstall "transformers==4.57.3"
```

### 6. Install Project Dependencies

```bash
uv pip install -e .
```

### 7. Install System Dependencies (if needed)

If you see `Python.h: No such file or directory` during compilation:

```bash
sudo apt-get install -y python3.12-dev
```

## Running Training

### Foreground

```bash
source .venv/bin/activate
python scripts/train_qwen3.py --config configs/qwen3_14b.yaml
```

### Background (Recommended)

```bash
source .venv/bin/activate
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=disabled \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    python scripts/train_qwen3.py --config configs/qwen3_14b.yaml \
    > training.log 2>&1 &

# Monitor
tail -f training.log
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Better memory allocation on newer GPUs |
| `WANDB_MODE=disabled` | Disable W&B logging (optional) |
| `LD_LIBRARY_PATH` | Ensure CUDA libraries are found |
| `HF_TOKEN` | HuggingFace authentication (optional) |
| `WANDB_API_KEY` | W&B authentication (optional) |

## Dependency Resolution Notes

The ML ecosystem has strict version interdependencies:

```
vLLM 0.15.0
  └── requires transformers==4.57.3
  └── requires torch>=2.9.0

Unsloth 2026.1.4
  └── requires transformers>=4.45.0
  └── requires bitsandbytes

bitsandbytes
  └── requires matching CUDA version
```

Installing packages in the wrong order can cause version conflicts. Follow the steps above in order.

## Troubleshooting

### transformers version mismatch

```
ImportError: cannot import name 'X' from 'transformers'
```

Fix:
```bash
uv pip install --force-reinstall "transformers==4.57.3"
```

### CUDA not found

```
RuntimeError: CUDA error: no kernel image is available
```

Verify CUDA:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Compilation errors

```
fatal error: Python.h: No such file or directory
```

Fix:
```bash
sudo apt-get install -y python3.12-dev
```

### Permission errors

```
PermissionError: [Errno 13] Permission denied: 'models/...'
```

Fix:
```bash
sudo chown -R $(whoami):$(whoami) models/
```
