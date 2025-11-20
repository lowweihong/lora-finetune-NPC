#!/usr/bin/env python3
"""Test script to diagnose model loading issues"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
from pathlib import Path

print("=" * 60)
print("Model Loading Test")
print("=" * 60)

# Check CUDA
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")

# Check paths
print(f"\n2. Current working directory: {os.getcwd()}")

base_model_name = "microsoft/Phi-3-mini-4k-instruct"
possible_paths = [
    "./npc_finetuned_bertscore-eval-noeval",
    "npc_finetuned_bertscore-eval-noeval",
    str(Path.cwd() / "npc_finetuned_bertscore-eval-noeval"),
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        print(f"   Found model at: {path}")
        break

if model_path is None:
    print(f"   ERROR: Model path not found!")
    print(f"   Tried: {possible_paths}")
    exit(1)

# Check HF token
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"\n3. HF_TOKEN: {'*' * (len(hf_token) - 4)}{hf_token[-4:]}")
else:
    print(f"\n3. HF_TOKEN: Not set (may be needed for private models)")

# Try loading tokenizer
print(f"\n4. Loading tokenizer from: {base_model_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("   ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading tokenizer: {e}")
    exit(1)

# Try loading model
print(f"\n5. Loading model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Try PEFT loading first
print("   Attempting PEFT loading...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("   ✓ Base model loaded")
    
    model = PeftModel.from_pretrained(base_model, model_path)
    print("   ✓ PEFT adapters loaded")
    model.eval()
    print("   ✓ Model set to eval mode")
    
except Exception as e:
    print(f"   ✗ PEFT loading failed: {type(e).__name__}: {e}")
    print("   Trying direct loading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("   ✓ Model loaded directly")
        model.eval()
    except Exception as e2:
        print(f"   ✗ Direct loading also failed: {type(e2).__name__}: {e2}")
        import traceback
        traceback.print_exc()
        exit(1)

# Try creating pipeline
print(f"\n6. Creating pipeline...")
try:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        device_map="auto"
    )
    print("   ✓ Pipeline created")
except Exception as e:
    print(f"   ✗ Pipeline creation failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test generation
print(f"\n7. Testing generation...")
try:
    messages = [
        {"role": "system", "content": "You are a grumpy blacksmith. Respond in character with emotion: angry."},
        {"role": "user", "content": "What about the dragon?"}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = generator(formatted, return_full_text=False, max_new_tokens=50)
    response = result[0]["generated_text"].strip()
    print(f"   ✓ Generation successful!")
    print(f"   Response: {response}")
except Exception as e:
    print(f"   ✗ Generation failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)

