#!/usr/bin/env python3
"""Evaluation pipeline for fine-tuned models."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

BENCHMARKS = {
    "general": ["mmlu", "hellaswag", "arc_challenge", "winogrande"],
    "reasoning": ["gsm8k", "math_algebra"],
    "code": ["humaneval", "mbpp"],
    "all": ["mmlu", "hellaswag", "arc_challenge", "gsm8k", "humaneval"],
}


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    output_dir: str,
    batch_size: int = 4,
    limit: int | None = None,
) -> dict:
    """Run lm-evaluation-harness benchmarks."""
    console.print(f"[bold]Running lm-eval[/bold]")
    console.print(f"Model: {model_path}")
    console.print(f"Tasks: {', '.join(tasks)}")

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        console.print(result.stdout)

        results_file = Path(output_dir) / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error: {e.stderr}[/red]")

    return {}


def evaluate_custom(model_path: str, test_file: str, output_dir: str) -> dict:
    """Run custom evaluation on JSONL test file."""
    console.print("[bold]Running custom evaluation[/bold]")

    if not os.path.exists(test_file):
        console.print(f"[yellow]Test file not found: {test_file}[/yellow]")
        return {}

    console.print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    results = {"total": len(test_data), "correct": 0, "examples": []}

    for item in track(test_data, description="Evaluating..."):
        prompt = item.get("prompt", item.get("instruction", ""))
        expected = item.get("expected", item.get("output", ""))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):].strip()

        is_correct = expected.lower().strip() in generated.lower()
        if is_correct:
            results["correct"] += 1

        results["examples"].append({
            "prompt": prompt[:100],
            "expected": expected[:100],
            "generated": generated[:100],
            "correct": is_correct,
        })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0

    output_file = Path(output_dir) / "custom_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]Accuracy: {results['accuracy']:.2%}[/green]")
    return results


def compare_models(
    base_model: str,
    finetuned_model: str,
    tasks: list[str],
    output_dir: str,
) -> dict:
    """Compare base vs fine-tuned model performance."""
    console.print("[bold green]Comparing models[/bold green]")

    console.print("\n[bold]Base model...[/bold]")
    base_results = run_lm_eval(base_model, tasks, f"{output_dir}/base", limit=100)

    console.print("\n[bold]Fine-tuned model...[/bold]")
    ft_results = run_lm_eval(finetuned_model, tasks, f"{output_dir}/finetuned", limit=100)

    table = Table(title="Model Comparison")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Base", style="yellow")
    table.add_column("Fine-tuned", style="green")
    table.add_column("Delta", style="magenta")

    comparison = {}
    if base_results and ft_results:
        for task in tasks:
            base_score = base_results.get("results", {}).get(task, {}).get("acc", 0)
            ft_score = ft_results.get("results", {}).get(task, {}).get("acc", 0)
            delta = ft_score - base_score

            table.add_row(task, f"{base_score:.2%}", f"{ft_score:.2%}", f"{delta:+.2%}")
            comparison[task] = {"base": base_score, "finetuned": ft_score, "delta": delta}

    console.print(table)

    with open(f"{output_dir}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main(
    model: str,
    base_model: str | None = None,
    benchmark: str = "general",
    tasks: list[str] | None = None,
    custom_test: str | None = None,
    output_dir: str = "./results/eval",
    batch_size: int = 4,
    limit: int | None = None,
) -> None:
    """Run evaluation pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    console.print("[bold]KLLM Evaluation[/bold]")
    console.print("=" * 60)

    eval_tasks = tasks or BENCHMARKS.get(benchmark, [])

    if base_model:
        compare_models(base_model, model, eval_tasks, output_dir)
    elif eval_tasks:
        run_lm_eval(model, eval_tasks, output_dir, batch_size, limit)

    if custom_test:
        evaluate_custom(model, custom_test, output_dir)

    console.print(f"\n[green]Results saved to: {output_dir}[/green]")


if __name__ == "__main__":
    import typer

    typer.run(main)
