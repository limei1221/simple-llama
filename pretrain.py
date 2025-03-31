from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from dataclasses import dataclass
import sys
import os
from transformers.trainer_callback import TrainerCallback
import torch
import json

from model import LlamaForCausalLM, LlamaConfig


@dataclass
class ScriptArguments:
    tokenizer_name: str
    dataset_name: str
    max_length: int


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_steps=1000):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.prompts = [
            "Once upon a time",
            "The future of artificial intelligence",
            "In the beginning",
        ]

    def on_log(self, args, state, control, logs=None, **kwargs):

        if state.global_step % self.eval_steps == 0:
            model = kwargs.get("model")
            if model is None:
                return

            model.eval()  # Set to evaluation mode
            print(f"\n=== Generating samples at step {state.global_step} ===")
            try:
                with torch.no_grad():  # Disable gradient calculation
                    for prompt in self.prompts:
                        # Move input tensors to the same device as the model
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}

                        outputs = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=script_args.max_length,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        generated_text = self.tokenizer.decode(
                            outputs[0], skip_special_tokens=True
                        )
                        print(f"\nPrompt: {prompt}")
                        print(f"Generated: {generated_text}")
            except Exception as e:
                print(f"Generation failed with error: {str(e)}")
                import traceback

                print(traceback.format_exc())
            finally:
                print("\n========================\n")
                model.train()  # Set back to training mode


def main(script_args: ScriptArguments, training_args: TrainingArguments):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

    if tokenizer.pad_token is None:
        # use the EOS token as the pad token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(training_args.output_dir)
    total_vocab_size = len(tokenizer)
    print(f"Total vocab size: {total_vocab_size}")

    # config = AutoConfig.from_pretrained(script_args.model_config)
    config = LlamaConfig(
        vocab_size=total_vocab_size,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
    )

    # Save config as json instead of using save_pretrained
    config_dict = (
        config.model_dump()
    )  # or config.dict() depending on your pydantic version
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    model = LlamaForCausalLM(config, tokenizer=tokenizer)

    # Only print on the main process
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_in_billions = trainable_params / 1_000_000_000
    print(f"Trainable parameters: {trainable_params_in_billions:.2f} billion")

    # Load your dataset
    dataset = load_dataset(
        script_args.dataset_name, name="default", split="train", streaming=True
    )

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=script_args.max_length,
        )
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize the generation callback
    generation_callback = GenerationCallback(
        tokenizer, eval_steps=training_args.eval_steps
    )

    # Initialize the Trainer with our custom trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=collator,
        callbacks=[generation_callback],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    # Make sure a yaml file is provided
    if len(sys.argv) < 2:
        raise ValueError("Please provide a yaml file with the training arguments")

    # Make sure the yaml file exists
    if not os.path.exists(sys.argv[1]):
        raise ValueError(f"The yaml file {sys.argv[1]} does not exist")

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_yaml_file(sys.argv[1])
    main(script_args, training_args)
