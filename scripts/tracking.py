#!/usr/bin/env python3
"""Weights & Biases utilities for experiment tracking."""

from __future__ import annotations

import os
from typing import Any

import wandb
from rich.console import Console
from rich.table import Table

console = Console()


class ExperimentTracker:
    """Wrapper for W&B experiment tracking."""

    def __init__(
        self,
        project: str = "kllm",
        entity: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
    ):
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.run: wandb.Run | None = None

    def start(self, name: str | None = None) -> wandb.Run:
        """Initialize a new W&B run."""
        if not os.environ.get("WANDB_API_KEY"):
            console.print("[yellow]WANDB_API_KEY not set, running offline[/yellow]")
            os.environ["WANDB_MODE"] = "offline"

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=self.config,
            tags=self.tags,
        )

        if self.run.url:
            console.print(f"[green]W&B run: {self.run.url}[/green]")

        return self.run

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to current run."""
        if self.run:
            wandb.log(metrics, step=step)

    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "model",
        metadata: dict | None = None,
    ) -> None:
        """Log artifact to current run."""
        if not self.run:
            console.print("[red]No active run[/red]")
            return

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_dir(path)
        self.run.log_artifact(artifact)
        console.print(f"[green]Artifact logged: {name}[/green]")

    def finish(self) -> None:
        """End the current run."""
        if self.run:
            wandb.finish()
            console.print("[green]Run finished[/green]")


def compare_runs(
    project: str = "kllm",
    entity: str | None = None,
    limit: int = 10,
) -> None:
    """Display comparison of recent runs."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    try:
        runs = list(api.runs(path))[:limit]
    except Exception as e:
        console.print(f"[red]Error fetching runs: {e}[/red]")
        return

    table = Table(title=f"Recent runs in {project}")
    table.add_column("Name", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Loss", style="yellow")
    table.add_column("Created", style="blue")

    for run in runs:
        loss = run.summary.get("eval/loss", run.summary.get("train/loss", "N/A"))
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        table.add_row(run.name, run.state, str(loss), run.created_at[:10])

    console.print(table)


def download_artifact(
    artifact_name: str,
    project: str = "kllm",
    entity: str | None = None,
    version: str = "latest",
    output_dir: str = "./artifacts",
) -> str:
    """Download artifact from W&B."""
    api = wandb.Api()
    full_name = f"{entity}/{project}/{artifact_name}:{version}" if entity else f"{project}/{artifact_name}:{version}"

    console.print(f"Downloading {full_name}...")
    artifact = api.artifact(full_name)
    path = artifact.download(root=output_dir)
    console.print(f"[green]Downloaded to: {path}[/green]")
    return path


def create_sweep_config(model_name: str = "qwen3") -> dict:
    """Generate hyperparameter sweep configuration."""
    return {
        "method": "bayes",
        "metric": {"name": "eval/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4,
            },
            "lora_r": {"values": [16, 32, 64, 128]},
            "lora_alpha": {"values": [32, 64, 128, 256]},
            "per_device_train_batch_size": {"values": [1, 2, 4]},
            "gradient_accumulation_steps": {"values": [4, 8, 16]},
        },
        "name": f"{model_name}-sweep",
    }


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="W&B experiment tracking utilities")

    @app.command()
    def compare(project: str = "kllm", entity: str | None = None, limit: int = 10):
        """Compare recent runs."""
        compare_runs(project, entity, limit)

    @app.command()
    def download(
        artifact_name: str,
        project: str = "kllm",
        entity: str | None = None,
        version: str = "latest",
        output_dir: str = "./artifacts",
    ):
        """Download an artifact."""
        download_artifact(artifact_name, project, entity, version, output_dir)

    @app.command()
    def sweep(model: str = "qwen3"):
        """Generate sweep configuration."""
        config = create_sweep_config(model)
        console.print(config)

    app()
