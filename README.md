# KLLM

Fine-tune state-of-the-art LLMs with LoRA/QLoRA on consumer hardware.

## Features

- **Multiple Model Support** - Qwen3, GPT-OSS, Nemotron, and more
- **Memory Efficient** - QLoRA enables training on GPUs with 16-32GB VRAM
- **Experiment Tracking** - Built-in Weights & Biases integration
- **Model Sharing** - One-click push to HuggingFace Hub
- **Docker Ready** - Uses prebuilt [unsloth/unsloth](https://hub.docker.com/r/unsloth/unsloth) image

## Supported Models

| Model | Parameters | Type | VRAM (QLoRA) |
|-------|------------|------|--------------|
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | 14B | Instruct | ~16GB |
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | 21B (MoE) | Reasoning | ~14GB |
| [Nemotron-3-Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30B (3B active) | Agentic | ~20GB |

## Quick Start

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Weights & Biases](https://wandb.ai/) account (optional)
- [HuggingFace](https://huggingface.co/) account (optional)

### Installation

```bash
git clone https://github.com/yourusername/kllm.git
cd kllm

cp .env.example .env
# Edit .env with your HF_TOKEN and WANDB_API_KEY

# Pull base image and build
docker pull unsloth/unsloth:latest
docker compose build
```

### Training

```bash
# Using Make
make train-qwen3

# Or with resource limits
docker run --rm -it \
  --gpus all \
  --cpus=8 \
  --memory=24g \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/workspace/.cache/huggingface \
  -e HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2) \
  -w /app \
  kllm:latest \
  python scripts/train_qwen3.py --config configs/qwen3_14b.yaml
```

### Evaluation

```bash
make eval MODEL=./models/qwen3-14b-finetuned BENCHMARK=general
```

## Data Format

KLLM uses Alpaca format (JSONL):

```json
{"instruction": "What is X?", "input": "Context here", "output": "Answer here"}
```

Place your data in:
- `data/train.jsonl`
- `data/eval.jsonl`

## Configuration

See `configs/qwen3_14b.yaml` for training options:

```yaml
model:
  name: "Qwen/Qwen3-14B"
  load_in_4bit: true

lora:
  r: 64
  lora_alpha: 128

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 2

data:
  train_file: "./data/train.jsonl"
  eval_file: "./data/eval.jsonl"

wandb:
  project: "kllm"
  tags: ["qwen3", "fine-tuning"]

hub:
  push_to_hub: false
  repo_id: "your-username/model-name"
```

## Makefile Commands

```bash
make build        # Build Docker image
make shell        # Interactive shell
make train-qwen3  # Train Qwen3-14B
make train-gpt    # Train GPT-OSS-20B
make train-nemo   # Train Nemotron-3-Nano
make tensorboard  # Start TensorBoard (localhost:6006)
make clean        # Remove containers
make lint         # Run linters
make format       # Format code
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://unsloth.ai/) - Fast fine-tuning framework
- [Hugging Face](https://huggingface.co/) - Model hub and transformers library
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Evaluation framework
