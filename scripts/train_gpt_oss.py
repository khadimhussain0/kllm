#!/usr/bin/env python3
"""Fine-tune GPT-OSS reasoning models using Unsloth with QLoRA."""

from __future__ import annotations

# Unsloth must be imported first to apply optimizations
from unsloth import FastLanguageModel

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
            tags=config.get("wandb", {}).get("tags", ["gpt-oss", "reasoning"]),
        )
        print(f"[W&B] Run: {wandb.run.url}")
    else:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="kllm", name=run_name, config=config, mode="offline")


def prepare_dataset(config: dict) -> dict[str, Dataset]:
    """Load reasoning dataset."""
    train_file = config["data"]["train_file"]
    eval_file = config["data"]["eval_file"]

    if os.path.exists(train_file):
        dataset = load_dataset(
            "json", data_files={"train": train_file, "eval": eval_file}
        )
        return {"train": dataset["train"], "eval": dataset["eval"]}

    print("Using sample dataset (gsm8k)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    split = dataset.train_test_split(test_size=0.1)
    return {"train": split["train"], "eval": split["test"]}


def format_reasoning_prompt(example: dict) -> dict:
    """Format example with thinking tags for chain-of-thought."""
    if "question" in example:
        question = example["question"]
        answer = example.get("answer", "")
        final = answer.split("####")[-1].strip() if "####" in answer else answer

        text = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<|thinking|>\n{answer}\n<|/thinking|>\n\n"
            f"The answer is {final}<|im_end|>"
        )
        return {"text": text}

    if "instruction" in example:
        instruction = example["instruction"]
        reasoning = example.get("reasoning", "")
        output = example.get("output", "")

        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<|thinking|>\n{reasoning}\n<|/thinking|>\n\n"
            f"{output}<|im_end|>"
        )
        return {"text": text}

    return example


def main(
    config: str = "configs/gpt_oss_20b.yaml",
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> None:
    """Run fine-tuning pipeline for GPT-OSS reasoning model."""
    config = load_config(config)

    if push_to_hub:
        config.setdefault("hub", {})["push_to_hub"] = True
        if hub_repo:
            config["hub"]["repo_id"] = hub_repo

    run_name = config["training"]["run_name"]

    print("=" * 60)
    print("KLLM - GPT-OSS Reasoning Fine-Tuning")
    print("=" * 60)
    print("Note: GPT-OSS uses MXFP4 format - Unsloth handles this natively")

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
    train_dataset = dataset["train"].map(format_reasoning_prompt)
    eval_dataset = dataset["eval"].map(format_reasoning_prompt)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Configure trainer
    print("[4/5] Configuring trainer...")
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        optim=config["training"]["optim"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        report_to="wandb",
        run_name=run_name,
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
            name=f"gpt-oss-{run_name}",
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
