# Contributing to KLLM

## Development Setup

```bash
git clone https://github.com/yourusername/kllm.git
cd kllm

uv venv .venv --python=3.12 --seed
source .venv/bin/activate

uv pip install -U vllm --torch-backend=cu128
uv pip install unsloth unsloth_zoo bitsandbytes
uv pip install --force-reinstall "transformers==4.57.3"
uv pip install -e ".[dev]"
```

## Code Style

```bash
ruff format .
ruff check --fix .
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes and run linting
3. Submit PR with clear description

## Adding New Models

1. Create config in `configs/`
2. Add training script in `scripts/` if needed
3. Update README
4. Test on small dataset

## Configuration Notes

- `lora_dropout` must be `0.0`
- `use_gradient_checkpointing: "unsloth"` recommended
- `bf16: true` preferred for newer GPUs

## Reporting Issues

Include: Python/CUDA versions, GPU model, full traceback, steps to reproduce.
