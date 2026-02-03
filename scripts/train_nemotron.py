#!/usr/bin/env python3
"""Fine-tune Nemotron-3-Nano for agentic/tool-use tasks using Unsloth with QLoRA."""

from __future__ import annotations

# Unsloth must be imported first to apply optimizations
from unsloth import FastLanguageModel

import json
import os
from typing import TYPE_CHECKING

import wandb
import yaml
from datasets import load_dataset
from huggingface_hub import login as hf_login
from transformers import TrainingArguments
from trl import SFTTrainer

if TYPE_CHECKING:
    from datasets import Dataset


def load_config(config_path: str | Path) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_integrations(config: dict, run_name: str) -> None:
    """Initialize HuggingFace and Weights & Biases connections."""
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        print("[HF] Authenticated")

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.init(
            project=config.get("wandb", {}).get("project", "kllm"),
            entity=config.get("wandb", {}).get("entity"),
            name=run_name,
            config=config,
            tags=config.get("wandb", {}).get("tags", ["nemotron", "agentic"]),
        )
        print(f"[W&B] Run: {wandb.run.url}")
    else:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="kllm", name=run_name, config=config, mode="offline")


def prepare_dataset(config: dict) -> dict[str, Dataset]:
    """Load agentic/tool-use dataset."""
    train_file = config["data"]["train_file"]
    eval_file = config["data"]["eval_file"]

    if os.path.exists(train_file):
        dataset = load_dataset(
            "json", data_files={"train": train_file, "eval": eval_file}
        )
        return {"train": dataset["train"], "eval": dataset["eval"]}

    print("Using sample dataset (glaive-function-calling)...")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    dataset = dataset.select(range(min(5000, len(dataset))))
    split = dataset.train_test_split(test_size=0.1)
    return {"train": split["train"], "eval": split["test"]}


def format_tool_calling_prompt(example: dict) -> dict:
    """Format example for tool/function calling."""
    if "system" in example and "chat" in example:
        system = example.get("system", "")
        chat = example.get("chat", "")
        text = f"<|im_start|>system\n{system}<|im_end|>\n{chat}"
        return {"text": text}

    if "tools" in example:
        tools = example.get("tools", [])
        query = example.get("query", example.get("instruction", ""))
        tool_calls = example.get("tool_calls", [])
        response = example.get("response", example.get("output", ""))

        tools_str = json.dumps(tools, indent=2)
        system = f"You are a helpful assistant.\n\nAvailable tools:\n{tools_str}"

        text = f"<|im_start|>system\n{system}<|im_end|>\n"
        text += f"<|im_start|>user\n{query}<|im_end|>\n"

        if tool_calls:
            tool_call_str = json.dumps({"tool_calls": tool_calls})
            text += f"<|im_start|>assistant\n{tool_call_str}<|im_end|>\n"
            text += f"<|im_start|>assistant\n{response}<|im_end|>"
        else:
            text += f"<|im_start|>assistant\n{response}<|im_end|>"

        return {"text": text}

    if "instruction" in example:
        instruction = example["instruction"]
        output = example.get("output", "")
        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        return {"text": text}

    return example


def main(
    config: str = "configs/nemotron_nano.yaml",
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> None:
    """Run fine-tuning pipeline for Nemotron agentic model."""
    config = load_config(config)

    if push_to_hub:
        config.setdefault("hub", {})["push_to_hub"] = True
        if hub_repo:
            config["hub"]["repo_id"] = hub_repo

    run_name = config["training"]["run_name"]

    print("=" * 60)
    print("KLLM - Nemotron-3-Nano Agentic Fine-Tuning")
    print("=" * 60)
    print("Note: Nemotron uses MoE (30B total, 3B active)")

    setup_integrations(config, run_name)

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    # Apply LoRA
    print("[2/5] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    print("[3/5] Loading dataset...")
    dataset = prepare_dataset(config)
    train_dataset = dataset["train"].map(format_tool_calling_prompt)
    eval_dataset = dataset["eval"].map(format_tool_calling_prompt)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Configure trainer
    print("[4/5] Configuring trainer...")
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        eval_strategy=training_config.get("eval_strategy", "no"),
        eval_steps=training_config.get("eval_steps"),
        optim=training_config["optim"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        report_to="none",
        run_name=run_name,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        packing=config["data"]["packing"],
        args=training_args,
    )

    # Train
    print("[5/5] Training...")
    print("-" * 60)
    trainer.train()

    # Save model
    output_dir = config["training"]["output_dir"]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    merged_dir = f"{output_dir}-merged"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # Log artifact
    if wandb.run and os.environ.get("WANDB_API_KEY"):
        artifact = wandb.Artifact(
            name=f"nemotron-{run_name}",
            type="model",
            metadata={"base_model": config["model"]["name"]},
        )
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)

    # Push to Hub
    if config.get("hub", {}).get("push_to_hub"):
        repo_id = config["hub"].get("repo_id")
        if repo_id:
            print(f"[HF] Pushing to {repo_id}...")
            model.push_to_hub(repo_id, private=config["hub"].get("private", False))
            tokenizer.push_to_hub(repo_id)

    print("=" * 60)
    print(f"Done! Model saved to {output_dir}")
    if wandb.run and wandb.run.url:
        print(f"W&B: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    import typer

    typer.run(main)
