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
from accelerate import Accelerator
import sys
import os

from model import LlamaForCausalLM, LlamaConfig


@dataclass
class ScriptArguments:
    tokenizer_name: str
    dataset_name: str


def main(script_args: ScriptArguments, training_args: TrainingArguments):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

    # config = AutoConfig.from_pretrained(script_args.model_config)
    config = LlamaConfig(
    vocab_size=128256,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=12,
    num_attention_heads=16
)
    model = LlamaForCausalLM(config)

    if tokenizer.pad_token is None:
        # use the EOS token as the pad token
        tokenizer.pad_token = tokenizer.eos_token

    # Only print on the main process
    # Get the accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
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
            examples["text"], padding=False, truncation=True, max_length=512
        )
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=collator,
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
