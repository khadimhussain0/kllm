#!/usr/bin/env python3
"""Dataset preparation utilities for different model formats."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.progress import track

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()

TEMPLATES = {
    "qwen3": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    },
}


def format_alpaca(example: dict, template: str = "qwen3") -> dict:
    """Convert Alpaca format to chat template."""
    tmpl = TEMPLATES[template]
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    text = tmpl["user"].format(content=user_content) + tmpl["assistant"].format(content=output)
    return {"text": text}


def format_sharegpt(example: dict, template: str = "qwen3") -> dict:
    """Convert ShareGPT format to chat template."""
    tmpl = TEMPLATES[template]
    conversations = example.get("conversations", [])

    text = ""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))

        if role in ["system"]:
            text += tmpl["system"].format(content=content)
        elif role in ["human", "user"]:
            text += tmpl["user"].format(content=content)
        elif role in ["gpt", "assistant"]:
            text += tmpl["assistant"].format(content=content)

    return {"text": text}


def format_reasoning(example: dict, template: str = "qwen3") -> dict:
    """Format for reasoning tasks with chain-of-thought."""
    tmpl = TEMPLATES[template]
    question = example.get("question", example.get("instruction", ""))
    reasoning = example.get("reasoning", example.get("chain_of_thought", ""))
    answer = example.get("answer", example.get("output", ""))

    response = f"<|thinking|>\n{reasoning}\n<|/thinking|>\n\n{answer}" if reasoning else answer
    text = tmpl["user"].format(content=question) + tmpl["assistant"].format(content=response)
    return {"text": text}


def format_tool_calling(example: dict, template: str = "qwen3") -> dict:
    """Format for tool/function calling tasks."""
    tmpl = TEMPLATES[template]
    tools = example.get("tools", [])
    query = example.get("query", example.get("instruction", ""))
    tool_calls = example.get("tool_calls", [])
    response = example.get("response", example.get("output", ""))

    system = "You are a helpful assistant."
    if tools:
        system += f"\n\nAvailable tools:\n{json.dumps(tools, indent=2)}"

    text = tmpl["system"].format(content=system)
    text += tmpl["user"].format(content=query)

    if tool_calls:
        text += tmpl["assistant"].format(content=json.dumps({"tool_calls": tool_calls}))
    text += tmpl["assistant"].format(content=response)

    return {"text": text}


def load_and_prepare(
    source: str,
    output_dir: str = "./data",
    template: str = "qwen3",
    format_type: str = "auto",
    train_split: float = 0.9,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[str, str]:
    """Load and prepare dataset for fine-tuning."""
    console.print(f"[bold]Loading: {source}[/bold]")

    if source.endswith((".jsonl", ".json")):
        with open(source) as f:
            data = [json.loads(line) for line in f] if source.endswith(".jsonl") else json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(source, split="train")

    console.print(f"Loaded {len(dataset)} examples")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    # Auto-detect format
    if format_type == "auto":
        sample = dataset[0]
        if "conversations" in sample:
            format_type = "sharegpt"
        elif "reasoning" in sample or "chain_of_thought" in sample:
            format_type = "reasoning"
        elif "tools" in sample or "tool_calls" in sample:
            format_type = "tool_calling"
        else:
            format_type = "alpaca"
        console.print(f"Detected format: {format_type}")

    formatters: dict[str, Callable] = {
        "alpaca": lambda x: format_alpaca(x, template),
        "sharegpt": lambda x: format_sharegpt(x, template),
        "reasoning": lambda x: format_reasoning(x, template),
        "tool_calling": lambda x: format_tool_calling(x, template),
    }

    formatted = [formatters[format_type](item) for item in track(dataset, description="Formatting...")]

    random.seed(seed)
    random.shuffle(formatted)

    split_idx = int(len(formatted) * train_split)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    console.print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    eval_file = output_path / "eval.jsonl"

    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(eval_file, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Saved to {output_dir}[/green]")
    return str(train_file), str(eval_file)


def create_samples(output_dir: str = "./data") -> None:
    """Create sample datasets for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_instructions": [
            {"instruction": "Explain quantum computing.", "output": "Quantum computing uses qubits..."},
            {"instruction": "Write a prime checker in Python.", "output": "```python\ndef is_prime(n):\n    ...```"},
        ],
        "sample_reasoning": [
            {"question": "If a train goes 120 miles in 2 hours, what's the speed?", "reasoning": "Speed = 120/2 = 60", "answer": "60 mph"},
        ],
        "sample_tool_calling": [
            {"tools": [{"name": "get_weather"}], "query": "Weather in Tokyo?", "tool_calls": [{"name": "get_weather", "args": {"location": "Tokyo"}}], "response": "22°C, sunny"},
        ],
    }

    for name, data in samples.items():
        with open(output_path / f"{name}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    console.print(f"[green]Created samples in {output_dir}[/green]")


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="Dataset preparation utilities")

    @app.command()
    def prepare(
        source: str,
        output_dir: str = "./data",
        template: str = "qwen3",
        format_type: str = "auto",
        train_split: float = 0.9,
        max_samples: int | None = None,
    ):
        """Prepare dataset for fine-tuning."""
        load_and_prepare(source, output_dir, template, format_type, train_split, max_samples)

    @app.command()
    def samples(output_dir: str = "./data"):
        """Create sample datasets."""
        create_samples(output_dir)

    app()
