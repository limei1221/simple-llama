import torch
import argparse
from transformers import AutoTokenizer
from model import LlamaForCausalLM, LlamaConfig
import json
import os
from typing import Optional, List, Union
import yaml


def load_model_and_tokenizer(
    model_path: str,
    device: Optional[str] = None,
) -> tuple[LlamaForCausalLM, AutoTokenizer]:
    """
    Load the trained model and tokenizer from the specified path.

    Args:
        model_path: Path to the directory containing the model and tokenizer
        device: Device to load the model on ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = LlamaConfig(**config_dict)

    # Initialize model
    model = LlamaForCausalLM(config, tokenizer=tokenizer)

    model_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(model_file) and model_file.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_file)

    model.load_state_dict(state_dict)

    # Move model to specified device
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_text(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    device: str = "cuda",
) -> List[str]:
    """
    Generate text from the model given a prompt.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to consider
        top_p: Cumulative probability threshold for nucleus sampling
        num_return_sequences: Number of sequences to generate
        device: Device to run generation on

    Returns:
        List of generated text sequences
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode outputs
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained Llama model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for text generation"
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML config file for generation parameters"
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            # Update args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Generate text
    print(f"\nGenerating text for prompt: {args.prompt}")
    generated_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        device=args.device or "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Print results
    print("\nGenerated text(s):")
    for i, text in enumerate(generated_texts, 1):
        print(f"\n{i}. {text}")


if __name__ == "__main__":
    main()
