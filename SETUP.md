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

## GGUF Conversion with llama.cpp

To convert fine-tuned models to GGUF format for use with Ollama, LM Studio, or llama.cpp.

### 1. Clone llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
```

### 2. Install Python Dependencies

```bash
pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

Or install manually:
```bash
pip install numpy~=1.26.4 sentencepiece~=0.2.0 transformers>=4.57.1 gguf>=0.1.0 protobuf>=4.21.0
```

### 3. (Optional) Build llama.cpp for Inference

Only needed if you want to run models directly with llama.cpp:

```bash
cd ~/llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # Use -DGGML_CUDA=OFF for CPU-only
make -j$(nproc)
```

### 4. Convert Model to GGUF

```bash
cd ~/llama.cpp
python convert_hf_to_gguf.py /path/to/merged/model \
    --outfile /path/to/output.gguf \
    --outtype q8_0 \
    --verbose
```

**Quantization options (`--outtype`):**
| Type | Description | Size (20B model) |
|------|-------------|------------------|
| `f16` | 16-bit float (best quality) | ~40GB |
| `q8_0` | 8-bit quantized | ~21GB |
| `q4_k_m` | 4-bit quantized (recommended) | ~12GB |
| `q5_k_m` | 5-bit quantized | ~14GB |

### 5. Example: Convert GPT-OSS Fine-tuned Model

```bash
# Create output directory
mkdir -p models/gpt-oss-gguf

# Convert merged model to Q8_0 GGUF
cd ~/llama.cpp
python convert_hf_to_gguf.py /home/elunic/kllm/models/gpt-oss-20b-finetuned-merged \
    --outfile /home/elunic/kllm/models/gpt-oss-gguf/gpt-oss-20b-finetuned-q8_0.gguf \
    --outtype q8_0 \
    --verbose
```

### 6. Use with Ollama

Create a `Modelfile`:
```bash
cat > models/gpt-oss-gguf/Modelfile << 'EOF'
FROM ./gpt-oss-20b-finetuned-q8_0.gguf

TEMPLATE """<|start|>user<|message|>{{ .Prompt }}<|end|>
<|start|>assistant<|channel|>final<|message|>"""

PARAMETER stop "<|end|>"
PARAMETER stop "<|return|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF
```

Create and run the model:
```bash
cd models/gpt-oss-gguf
ollama create my-gpt-oss -f Modelfile
ollama run my-gpt-oss
```

### 7. Use with llama.cpp Directly

```bash
~/llama.cpp/build/bin/llama-cli \
    -m models/gpt-oss-gguf/gpt-oss-20b-finetuned-q8_0.gguf \
    -p "What is DNA?" \
    -n 512 \
    --temp 0.7
```

### Notes

- **Merged model required**: Convert the merged model (with LoRA weights merged), not the LoRA adapter directory
- **Conversion time**: ~3-5 minutes for a 20B model
- **Disk space**: Ensure enough space for both the source model and GGUF output
- **GPU not required**: Conversion runs on CPU
