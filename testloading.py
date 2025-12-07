from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
pruned_model_path = "C:/Users/namnd/Documents/QwenCoder-50"

print(" Loading PRUNED model...")
pruned_tokenizer = AutoTokenizer.from_pretrained(
    pruned_model_path,
    trust_remote_code=True,
    local_files_only=True
)

pruned_model = AutoModelForCausalLM.from_pretrained(
    pruned_model_path,
    torch_dtype=torch.float16,  # Use float16 for CPU
    low_cpu_mem_usage=True,      # Optimize CPU memory usage
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto"
)
print(" Model loaded successfully.")

# Generate a Java method
prompt = """Write a Java method that calculates the factorial of a number:
"""

print("\n Generating Java method...")
inputs = pruned_tokenizer(prompt, return_tensors="pt").to(pruned_model.device)

with torch.no_grad():
    outputs = pruned_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False,
        pad_token_id=pruned_tokenizer.eos_token_id
    )

generated_text = pruned_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n Generated code:")
print(generated_text)

