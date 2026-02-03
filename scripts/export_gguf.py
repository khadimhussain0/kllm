#!/usr/bin/env python3
"""Export fine-tuned model to GGUF format for Ollama/LM Studio."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()

QUANT_OPTIONS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def main(
    model_path: str = "./models/qwen3-14b-finetuned-merged",
    output_dir: str = "./models/gguf",
    quantization: str = "q4_k_m",
) -> None:
    """
    Export model to GGUF format.

    Quantization options:
      q4_k_m - 4-bit (recommended, smallest)
      q5_k_m - 5-bit (balanced)
      q8_0   - 8-bit (better quality)
      f16    - 16-bit (best quality, largest)
    """
    from unsloth import FastLanguageModel

    if quantization not in QUANT_OPTIONS:
        console.print(f"[red]Invalid quantization. Choose from: {QUANT_OPTIONS}[/red]")
        raise typer.Exit(1)

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading model from {model_path}...[/bold]")

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model_name = model_path.name

    console.print(f"[bold]Exporting to GGUF ({quantization})...[/bold]")
    console.print(f"Output: {output_dir}")

    model.save_pretrained_gguf(
        str(output_dir / model_name),
        tokenizer,
        quantization_method=quantization,
    )

    # Find generated GGUF file
    gguf_files = list(output_dir.glob("**/*.gguf"))

    console.print(f"\n[green]Done![/green]")

    if gguf_files:
        gguf_file = gguf_files[0]
        console.print(f"GGUF file: {gguf_file}")

        # Create Modelfile for Ollama
        modelfile_content = f"""FROM ./{gguf_file.name}

TEMPLATE \"\"\"<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
        modelfile_path = gguf_file.parent / "Modelfile"
        modelfile_path.write_text(modelfile_content)

        console.print(f"\n[bold cyan]Ollama:[/bold cyan]")
        console.print(f"  cd {gguf_file.parent}")
        console.print(f"  ollama create {model_name} -f Modelfile")
        console.print(f"  ollama run {model_name}")

        console.print(f"\n[bold cyan]LM Studio:[/bold cyan]")
        console.print(f"  Import: {gguf_file}")


if __name__ == "__main__":
    typer.run(main)
