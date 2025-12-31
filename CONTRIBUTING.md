# Contributing to KLLM

Thank you for your interest in contributing to KLLM!

## Development Setup

1. Fork and clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install ruff pytest
   ```

## Code Style

- Format code with `ruff format`
- Check linting with `ruff check`
- Use type hints for function signatures
- Write docstrings for public functions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run linting and tests
4. Submit a pull request with a clear description

## Adding New Models

To add support for a new model:

1. Create a config file in `configs/`
2. Add a training script in `scripts/` if needed
3. Update the README with model details
4. Test training on a small dataset

## Reporting Issues

When reporting bugs, please include:

- Python and CUDA versions
- GPU model and VRAM
- Full error traceback
- Steps to reproduce
