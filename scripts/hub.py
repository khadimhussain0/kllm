#!/usr/bin/env python3
"""HuggingFace Hub utilities for model management."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import (
    HfApi,
    create_repo,
    login,
    snapshot_download,
    upload_folder,
    whoami,
)
from rich.console import Console
from rich.table import Table

console = Console()


def check_auth() -> bool:
    """Verify HuggingFace authentication status."""
    try:
        user = whoami()
        console.print(f"[green]Authenticated as: {user['name']}[/green]")
        return True
    except Exception:
        console.print("[red]Not authenticated[/red]")
        console.print("Set HF_TOKEN environment variable or run: huggingface-cli login")
        return False


def hf_login(token: str | None = None) -> None:
    """Authenticate with HuggingFace Hub."""
    token = token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        console.print("[green]Login successful[/green]")
    else:
        console.print("[yellow]No token provided, running interactive login...[/yellow]")
        login()


def upload_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload model",
) -> None:
    """Upload model directory to HuggingFace Hub."""
    if not check_auth():
        return

    path = Path(model_path)
    if not path.exists():
        console.print(f"[red]Path not found: {model_path}[/red]")
        return

    console.print(f"Creating repo: {repo_id}")
    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        console.print(f"[yellow]Note: {e}[/yellow]")

    console.print(f"Uploading from {model_path}...")
    upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        commit_message=commit_message,
        ignore_patterns=["*.pyc", "__pycache__", "*.log", "wandb/"],
    )

    console.print(f"[green]Done! https://huggingface.co/{repo_id}[/green]")


def create_model_card(
    repo_id: str,
    base_model: str,
    task: str = "text-generation",
    dataset: str = "custom",
) -> None:
    """Generate and upload a model card."""
    if not check_auth():
        return

    model_card = f"""---
license: mit
base_model: {base_model}
tags:
- fine-tuned
- fine-tuned
- {task}
datasets:
- {dataset}
pipeline_tag: {task}
---

# {repo_id.split('/')[-1]}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Your prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Training

- **Method**: QLoRA (4-bit quantization + LoRA)
- **Framework**: Unsloth + Hugging Face TRL

## License

MIT
"""

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    console.print(f"[green]Model card uploaded to {repo_id}[/green]")


def download_model(
    repo_id: str,
    local_dir: str = "./models",
    revision: str = "main",
) -> str:
    """Download model from HuggingFace Hub."""
    console.print(f"Downloading {repo_id}...")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, revision=revision)
    console.print(f"[green]Downloaded to: {path}[/green]")
    return path


def list_my_models() -> None:
    """List models owned by authenticated user."""
    if not check_auth():
        return

    user = whoami()
    api = HfApi()
    models = list(api.list_models(author=user["name"]))

    table = Table(title=f"Models by {user['name']}")
    table.add_column("Model", style="cyan")
    table.add_column("Downloads", style="green")
    table.add_column("Likes", style="yellow")

    for model in models:
        table.add_row(model.id, str(model.downloads), str(model.likes))

    console.print(table)


if __name__ == "__main__":
    import typer

    app = typer.Typer(help="HuggingFace Hub utilities")

    app.command("login")(hf_login)
    app.command("upload")(upload_model)
    app.command("card")(create_model_card)
    app.command("download")(download_model)
    app.command("models")(list_my_models)
    app.command("check")(check_auth)

    app()
