"""AIMO Progress Prize 3 - Fine-tuning Script

Fine-tune math models using LoRA on competition-relevant data.
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Training configuration
CONFIG = {
    "base_model": "Qwen/Qwen2.5-Math-7B-Instruct",  # Start with 7B for testing
    "output_dir": "./checkpoints",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "max_seq_length": 4096,
    "warmup_ratio": 0.1,
}


def load_math_datasets():
    """Load and combine math training datasets."""
    datasets_to_load = [
        ("lighteval/MATH", "all"),  # MATH dataset
        ("gsm8k", "main"),  # Grade school math
    ]

    all_data = []

    for dataset_name, split in datasets_to_load:
        try:
            ds = load_dataset(dataset_name, split="train")
            print(f"Loaded {len(ds)} examples from {dataset_name}")
            all_data.extend(ds)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")

    return all_data


def format_training_example(example: dict) -> str:
    """Format a training example with chain-of-thought."""
    system = """You are an expert mathematician. Solve the problem step by step, showing all work. Give your final answer in \\boxed{answer} format."""

    if "problem" in example:
        problem = example["problem"]
    elif "question" in example:
        problem = example["question"]
    else:
        return None

    if "solution" in example:
        solution = example["solution"]
    elif "answer" in example:
        solution = f"The answer is \\boxed{{{example['answer']}}}"
    else:
        return None

    # Format as chat
    formatted = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
{solution}<|im_end|>"""

    return formatted


def prepare_dataset(tokenizer, max_length: int = 4096):
    """Prepare training dataset."""
    raw_data = load_math_datasets()

    formatted_examples = []
    for example in raw_data:
        formatted = format_training_example(example)
        if formatted:
            formatted_examples.append({"text": formatted})

    dataset = Dataset.from_list(formatted_examples)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def train():
    """Main training function."""
    print(f"Training config: {CONFIG}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = prepare_dataset(tokenizer, CONFIG["max_seq_length"])
    print(f"Training on {len(train_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {CONFIG['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])

    print("Training complete!")


if __name__ == "__main__":
    train()
