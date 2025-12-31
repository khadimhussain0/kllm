.PHONY: build shell train-qwen3 train-gpt train-nemo tensorboard clean lint format help

DOCKER_IMAGE := kllm
DOCKER_RUN := docker compose run --rm

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker compose build

shell: ## Start interactive shell
	$(DOCKER_RUN) kllm bash

train-qwen3: ## Train Qwen3-14B model
	$(DOCKER_RUN) kllm python scripts/train_qwen3.py --config configs/qwen3_14b.yaml

train-gpt: ## Train GPT-OSS-20B model
	$(DOCKER_RUN) kllm python scripts/train_gpt_oss.py --config configs/gpt_oss_20b.yaml

train-nemo: ## Train Nemotron-3-Nano model
	$(DOCKER_RUN) kllm python scripts/train_nemotron.py --config configs/nemotron_nano.yaml

eval: ## Run evaluation
	$(DOCKER_RUN) kllm python evaluation/run_eval.py --model $(MODEL) --benchmark $(BENCHMARK)

tensorboard: ## Start TensorBoard
	docker compose up tensorboard

clean: ## Remove containers and cache
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

lint: ## Run linters
	$(DOCKER_RUN) kllm ruff check scripts/ data/ evaluation/

format: ## Format code
	$(DOCKER_RUN) kllm ruff format scripts/ data/ evaluation/
