#!/usr/bin/env python3
"""Interactive chat with fine-tuned model."""

from __future__ import annotations

from unsloth import FastLanguageModel
import typer


def main(
    model_path: str = "./models/qwen3-14b-finetuned",
    max_tokens: int = 512,
) -> None:
    """Chat with your fine-tuned model."""
    print(f"Loading {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("\n" + "=" * 50)
    print("Chat with your fine-tuned model")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            question = input("\n> ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            prompt = (
                f"<|im_start|>system\n"
                f"You are a helpful assistant. Provide clear, well-formatted answers with proper capitalization and punctuation.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("assistant\n")[-1].strip()

            print(f"\n{response}")
        except KeyboardInterrupt:
            break

    print("\nBye!")


if __name__ == "__main__":
    typer.run(main)
