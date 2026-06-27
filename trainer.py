"""
LoRA / QLoRA Training Pipeline
Supports Mistral, Llama-3, Gemma with 4-bit quantization.
"""
from __future__ import annotations
import os, json, time, logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16                   # LoRA rank
    lora_alpha: int = 32          # scaling = lora_alpha / r
    lora_dropout: float = 0.05
    bias: str = "none"            # none | all | lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    dataset_name: str = ""
    output_dir: str = "./finetuned"
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    # Sequence
    max_seq_length: int = 1024
    packing: bool = True
    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class LoRATrainer:
    """
    Full LoRA/QLoRA fine-tuning pipeline using HuggingFace PEFT + TRL.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._check_deps()

    def _check_deps(self):
        missing = []
        for pkg in ["transformers", "peft", "trl", "datasets", "accelerate"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(f"pip install {' '.join(missing)} bitsandbytes")

    def load_model_and_tokenizer(self):
        """Load quantized model + tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self.config.model_name}...")

        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def apply_lora(self, model):
        """Wrap model with PEFT LoRA adapters."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from peft import TaskType

        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora.target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    def load_dataset(self):
        """Load and format the training dataset."""
        from datasets import load_dataset, Dataset
        if os.path.exists(self.config.dataset_name):
            # Local JSONL
            data = []
            with open(self.config.dataset_name) as f:
                for line in f:
                    data.append(json.loads(line))
            dataset = Dataset.from_list(data)
        else:
            dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"Dataset size: {len(dataset)} samples")
        return dataset

    def format_prompts(self, dataset, tokenizer):
        """Format dataset into instruction-following format."""
        def _format(sample):
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            output = sample.get("output", sample.get("response", ""))
            if inp:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return {"text": text}

        return dataset.map(_format)

    def train(self):
        """Full training loop."""
        from trl import SFTTrainer
        from transformers import TrainingArguments

        model, tokenizer = self.load_model_and_tokenizer()
        model = self.apply_lora(model)
        dataset = self.load_dataset()
        dataset = self.format_prompts(dataset, tokenizer)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="no",
            fp16=True,
            report_to="tensorboard",
            group_by_length=True,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
        )

        logger.info("🚀 Starting training...")
        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        logger.info(f"Training done in {elapsed/60:.1f} min")

        # Save
        trainer.model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Model saved to {self.config.output_dir}")
        return trainer
