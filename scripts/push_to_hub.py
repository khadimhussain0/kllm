#!/usr/bin/env python3
"""Push model to HuggingFace Hub."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()


def main(
    model_path: str = typer.Argument(..., help="Path to model folder"),
    repo_name: str = typer.Option(None, "--name", "-n", help="Repository name (default: folder name)"),
    private: bool = typer.Option(False, "--private", "-p", help="Make repository private"),
) -> None:
    """Push a model folder to HuggingFace Hub."""
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")

    if not hf_token:
        print("Error: HF_TOKEN not found in .env")
        raise typer.Exit(1)

    if not hf_username:
        print("Error: HF_USERNAME not found in .env")
        raise typer.Exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        raise typer.Exit(1)

    # Use folder name if repo_name not specified
    if repo_name is None:
        repo_name = model_path.name

    repo_id = f"{hf_username}/{repo_name}"

    print(f"Logging in to HuggingFace...")
    login(token=hf_token)

    print(f"Pushing to: {repo_id}")
    print(f"From: {model_path}")

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # Upload folder
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message="Upload model",
    )

    print(f"\nDone! Model available at:")
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    typer.run(main)
