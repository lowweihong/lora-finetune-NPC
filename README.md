# NPC Dialogue Generation with LoRA Fine-tuning

This project fine-tunes Microsoft's Phi-3-mini-4k-instruct model using LoRA (Low-Rank Adaptation) to generate character-consistent NPC (Non-Player Character) dialogue for games.

## Demo

![Demo](demo.gif)

## Overview

The model is trained to generate contextual, in-character responses for NPCs based on their personality, biography, and emotional state. The training uses a supervised fine-tuning (SFT) approach with LoRA adapters to efficiently fine-tune the base model while maintaining low memory requirements.

## Model Details

- **Base Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 16
  - Target Modules: all-linear layers
  - Dropout: 0.05

## Dataset

- **Source**: `amaydle/npc-dialogue` from HuggingFace
- **Format**: Each example contains:
  - Character Name
  - Biography
  - Emotion
  - Query (player input)
  - Response (expected NPC response)
- **Training Examples**: 1,723 examples

## Training Configuration

- **Epochs**: 10
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Optimizer**: paged_adamw_8bit
- **Quantization**: 4-bit (NF4) for memory efficiency
- **Sequence Length**: 4096 tokens
- **Packing**: Enabled for efficient training

## Training Process

The training script (`lora-finetune.py`) performs the following:

1. Loads and formats the NPC dialogue dataset
2. Applies 4-bit quantization to reduce memory usage
3. Prepares the model for LoRA training
4. Trains using SFTTrainer with completion-only loss (focuses on assistant responses)
5. Saves LoRA adapters to `./npc_finetuned_bertscore-eval-noeval/`

## Evaluation

The evaluation notebook (`eval.ipynb`) compares the fine-tuned model against the base model using:

- **BERTScore**: Semantic similarity (F1, Precision, Recall)
- **BLEU**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **Perplexity**: Language modeling quality

Evaluation is performed on the test split of the dataset, with results exported to `final_res_test.xlsx` containing:
- Original responses
- Base model generations
- Fine-tuned model generations
- Character prompts

## Setup

### Environment Variables

Before running the training or evaluation scripts, you need to set the following environment variables:

```bash
# HuggingFace token for accessing models and datasets
export HF_TOKEN="your_huggingface_token_here"

# Optional: Specify which GPU to use (e.g., '0', '1', '4')
# If not set, will use all available GPUs
export CUDA_VISIBLE_DEVICES="1"
```

You can also set these in your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) or create a `.env` file:

```bash
# .env file
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=1
```

**Note**: The training and evaluation scripts now use environment variables for authentication. If `HF_TOKEN` is not set, the scripts will raise a clear error message with instructions.

### Getting a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions (or "write" if you plan to push models)
3. Copy the token and set it as `HF_TOKEN`

## Usage

### Training

```bash
python lora-finetune.py
```

Training configuration can be adjusted in `config.yaml`.

### Evaluation

Open `eval.ipynb` and run the cells to:
1. Load the test dataset
2. Generate responses from both base and fine-tuned models
3. Compute evaluation metrics
4. Export results to Excel

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

# Load base model with quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=quant_config,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./npc_finetuned_bertscore-eval-noeval")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Generate response
messages = [
    {"role": "system", "content": "You are a grumpy blacksmith. Respond in character with emotion: angry."},
    {"role": "user", "content": "What about the dragon?"}
]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator(formatted_prompt, max_new_tokens=50, return_full_text=False)
```

## Output Files

- `npc_finetuned_bertscore-eval-noeval/`: Directory containing the trained LoRA adapters
- `final_res_test.xlsx`: Evaluation results comparing base vs fine-tuned models
- `lora_log_eval_bert-noeval.log`: Training log file

## Dependencies

- transformers
- peft
- trl
- datasets
- torch
- evaluate
- bitsandbytes (for 4-bit quantization)
- pandas, openpyxl (for Excel export)

## Notes

- The model uses 4-bit quantization to fit on GPUs with limited VRAM
- Training was performed without evaluation during training (noeval)
- The model focuses on generating character-consistent responses based on personality and emotion cues

