from transformers import pipeline
import torch

model_path = r"C:\Users\namnd\Documents\Qwen-Coder-Prunned-NEW"

print("Loading model...")
generator = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("\nTesting...")
prompt = "Hello how are your"
response = generator(prompt, max_new_tokens=100)
print(response[0]['generated_text'])
