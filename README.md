# KLLM

Fine-tune LLMs with LoRA/QLoRA on consumer hardware.

## Features

- **QLoRA Training** - Train 14B+ models on 16-32GB VRAM
- **Multiple Models** - Qwen3, GPT-OSS, Nemotron support
- **W&B Integration** - Built-in experiment tracking
- **HuggingFace Hub** - One-click model publishing

## Supported Models

| Model | Parameters | VRAM (QLoRA) |
|-------|------------|--------------|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14B | ~16GB |
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | 21B (MoE) | ~14GB |
| [Nemotron-3-Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30B (3B active) | ~20GB |

## Quick Start

### Prerequisites

- NVIDIA GPU (16GB+ VRAM)
- Python 3.12
- CUDA 12.x

### Installation

```bash
git clone https://github.com/khadimhussain0/kllm.git
cd kllm

# Create virtual environment
uv venv .venv --python=3.12 --seed
source .venv/bin/activate

# Install dependencies (CUDA 12.8 for RTX 50 series)
uv pip install -U vllm --torch-backend=cu128
uv pip install unsloth unsloth_zoo bitsandbytes
uv pip install --force-reinstall "transformers==4.57.3"
uv pip install -e .
```

For CUDA 12.1 (RTX 40 series), use `--torch-backend=cu121`.

See [SETUP.md](SETUP.md) for detailed installation and troubleshooting.

### Training

```bash
source .venv/bin/activate
python scripts/train_qwen3.py --config configs/qwen3_14b.yaml
```

Background training:
```bash
nohup python scripts/train_qwen3.py --config configs/qwen3_14b.yaml > training.log 2>&1 &
tail -f training.log
```

## Data Format

Alpaca format (JSONL):

```json
{"instruction": "What is X?", "input": "Context here", "output": "Answer here"}
```

Place data in `data/train.jsonl` and `data/eval.jsonl`.

## Configuration

See `configs/qwen3_14b.yaml`:

```yaml
model:
  name: "Qwen/Qwen3-14B"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.0

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  bf16: true
```

## Docker (Alternative)

```bash
docker pull unsloth/unsloth:latest
docker compose build
docker run --rm -it --gpus all -v $(pwd):/app -w /app kllm:latest \
  python scripts/train_qwen3.py --config configs/qwen3_14b.yaml
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Silent process death | Reduce batch size, check `dmesg \| grep oom` |
| `UnboundLocalError` with dropout | Set `lora_dropout: 0.0` |
| transformers import errors | `uv pip install --force-reinstall "transformers==4.57.3"` |
| Permission denied on save | `sudo chown -R $(whoami) models/` |
| Compilation errors | `sudo apt-get install python3.12-dev` |

See [SETUP.md](SETUP.md) for detailed troubleshooting.

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- [Unsloth](https://unsloth.ai/)
- [Hugging Face](https://huggingface.co/)
- [Weights & Biases](https://wandb.ai/)
