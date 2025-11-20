import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from huggingface_hub import login

# Force CPU usage and optimize
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(os.cpu_count() or 4)  # Use all CPU cores

# Page config
st.set_page_config(
    page_title="NPC Dialogue Generator",
    page_icon="🎮",
    layout="wide"
)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer from local path or Hugging Face Hub"""
    
    # Get HF_TOKEN from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print(f"HF login note: {e}")
    
    # Get HF model ID from environment (for deployment)
    hf_model_id = os.environ.get("HF_MODEL_ID")
    
    # Try local paths first (for local development)
    model_path = None
    for path in ["./npc_finetuned_quantized", "./npc_finetuned_bertscore-eval-noeval"]:
        if os.path.exists(path):
            model_path = path
            break
    
    # If local not found and HF_MODEL_ID is set, use Hugging Face Hub
    if model_path is None:
        if hf_model_id:
            print(f"Local model not found. Loading from Hugging Face Hub: {hf_model_id}")
            model_path = hf_model_id
        else:
            raise FileNotFoundError(
                "Model not found locally and HF_MODEL_ID not set.\n"
                "Either:\n"
                "1. Place model in ./npc_finetuned_bertscore-eval-noeval, or\n"
                "2. Set HF_MODEL_ID environment variable (e.g., 'your-username/npc-finetuned-bertscore')"
            )
    else:
        print(f"Loading model from local path: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    
    # Load model on CPU with optimizations
    print("Loading model (this may take a minute)...")
    # Use float16 for faster inference (half the size, faster on CPU)
    try:
        dtype = torch.float16
    except:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,  # Use dtype instead of torch_dtype (deprecated)
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=hf_token
    )
    model.eval()
    
    # Try to compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model for faster inference...")
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Model compiled")
        except Exception as e:
            print(f"Compilation skipped: {e}")
    
    # Create pipeline with optimizations
    print("Creating pipeline...")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    print("✓ Model ready!")
    
    return generator, tokenizer, model

def generate_response(generator, tokenizer, personality, emotion, query, max_new_tokens=150):
    """Generate response using the fine-tuned model"""
    system_content = f"You are {personality}. Respond in character with emotion: {emotion}."
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    result = generator(formatted_prompt, return_full_text=False, max_new_tokens=max_new_tokens)
    response = result[0]["generated_text"].strip()
    
    return response

# Main UI
st.title("🎮 NPC Dialogue Generator")
st.markdown("Generate character responses using your fine-tuned LoRA model")

# Load model with progress indicator
if 'model_loaded' not in st.session_state:
    with st.spinner("Loading model and tokenizer (first time may take 1-2 minutes)..."):
        try:
            generator, tokenizer, model = load_model_and_tokenizer()
            st.session_state.model_loaded = True
            st.session_state.generator = generator
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.exception(e)
            st.stop()
else:
    generator = st.session_state.generator
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

# Input form
with st.form("dialogue_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        personality = st.text_area(
            "Personality/Biography",
            placeholder="e.g., a grumpy blacksmith who has been working for 30 years",
            height=100,
            help="Describe the character's personality and background"
        )
        
        emotion = st.text_input(
            "Emotion",
            placeholder="e.g., angry, happy, sad, excited",
            help="The emotional state of the character"
        )
    
    with col2:
        query = st.text_area(
            "Query/Question",
            placeholder="e.g., What about the dragon?",
            height=100,
            help="The question or statement to respond to"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=300,
            value=150,
            help="Maximum number of tokens to generate"
        )
    
    submitted = st.form_submit_button("Generate Response", type="primary", use_container_width=True)

# Generate and display response
if submitted:
    if not personality or not emotion or not query:
        st.warning("⚠️ Please fill in all fields (Personality, Emotion, and Query)")
    else:
        with st.spinner(f"Generating response (max {max_tokens} tokens)..."):
            try:
                import time
                start_time = time.time()
                response = generate_response(generator, tokenizer, personality, emotion, query, max_tokens)
                elapsed = time.time() - start_time
                st.caption(f"Generated in {elapsed:.1f}s")
                
                st.success("Response generated!")
                st.markdown("### Generated Response:")
                st.info(response)
                
                # Show the formatted prompt for debugging
                with st.expander("View formatted prompt"):
                    system_content = f"You are {personality}. Respond in character with emotion: {emotion}."
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": query}
                    ]
                    formatted = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    st.code(formatted, language="text")
                    
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.exception(e)

# Sidebar with examples
with st.sidebar:
    st.header("📝 Examples")
    st.markdown("""
    **Example 1:**
    - **Personality:** a grumpy blacksmith
    - **Emotion:** angry
    - **Query:** What about the dragon?
    
    **Example 2:**
    - **Personality:** Bikram is a rough and tough smuggler from the streets of Calcutta, India
    - **Emotion:** thoughtful
    - **Query:** What is your opinion on friendship?
    """)
    
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a fine-tuned Phi-3-mini model with LoRA adapters
    trained on NPC dialogue data. The model generates character responses
    based on personality, emotion, and user queries.
    """)
