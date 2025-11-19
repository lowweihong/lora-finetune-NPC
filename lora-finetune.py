import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from torch.distributed.elastic.multiprocessing.errors import record
import evaluate
import numpy as np
from transformers import EvalPrediction
import torch

import os
from huggingface_hub import login

# Get HF_TOKEN from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Please set it using: export HF_TOKEN='your_token_here'"
    )

# Login to HuggingFace
login(token=hf_token)

# Set CUDA_VISIBLE_DEVICES if not already set (optional, defaults to GPU 1)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Format for SFT: Add persona to system prompt
def format_example(example):
    system = f"You are {example['Name']}, {example['Biography']}. Respond in character with emotion: {example['Emotion']}."
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": example["Query"]},
            {"role": "assistant", "content": example["Response"].strip('"')}
        ]
    }

@record
def main():
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset("amaydle/npc-dialogue")
    
    train_dataset = dataset['train'].map(format_example)
    
    # formatted.to_json("train.json")  # For your config
    train_dataset.to_json(config.get('dataset_id_or_path','train_split.json'))

    # eval_dataset = dataset['test'].map(format_example)
    # eval_dataset.to_json(config.get('eval_dataset_id_or_path', 'dev_split.json'))
    
    # Quantization for efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        # attn_implementation=config.get("attn_implementation", "flash_attention_2"),
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)  # Prep for PEFT
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    # Load BERTScore metric (once, outside the function)
    bertscore = evaluate.load("bertscore")
    
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        # Convert to PyTorch tensors
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        
        # Shift for causal LM: predictions are logits shifted left
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode predictions and labels (ignore -100 masked tokens)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        label_texts = tokenizer.batch_decode(torch.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Compute BERTScore (semantic similarity; use F1 as main score)
        bert_results = bertscore.compute(predictions=pred_texts, references=label_texts, lang="en")
        
        # Optional: Add perplexity for comparison
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        perplexity = np.exp(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels).item())
        
        perf = {
            "bertscore_f1": np.mean(bert_results["f1"]),  # Average F1 score
            "bertscore_precision": np.mean(bert_results["precision"]),
            "bertscore_recall": np.mean(bert_results["recall"]),
            "perplexity": perplexity  # Optional built-in metric
        }
        print(perf)
        return perf

    # LoRA config
    peft_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"])  # Or "all-linear"
    )
    
    # Load dataset
    train_dataset = Dataset.from_json(config["dataset_id_or_path"])
    
    # Training args (switch to SFTConfig for assistant_only_loss support)
    args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        bf16=True,  # If supported
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        logging_steps=10,
        save_steps=500,
        # eval_strategy="steps",  # Always "steps" now (or "epoch" for per-epoch)
        # eval_steps=10,  # Eval every 500 steps (adjust to match save_steps or less frequent)
        push_to_hub=config.get("push_to_hub", False),
        completion_only_loss=True,  # Added here! Focuses loss on assistant tokens only
        # max_seq_length=config.get("max_seq_length", 1024),  # Optional: Re-add if you want explicit control
        packing=config.get("packing", True)  # Optional: Re-add if you want packing enabled
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        processing_class=tokenizer,  # Changed here
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,  # Add this!
        peft_config=peft_config,
        # max_seq_length=config.get("max_seq_length", 1024),
        # packing=config.get("packing", True),
        compute_metrics=compute_metrics,  # Add this!
        args=args
    )
    
    # Train
    trainer.train()
    
    # Save (adapters only)
    trainer.save_model(config["output_dir"])



if __name__ == '__main__':
    main()


# nohup python lora-finetune.py >> lora_log.log 2>&1&
