from transformers import pipeline
import torch

model_path = r"C:\Users\namnd\Documents\QwenCoder-50"

print("Loading model...")
generator = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("\nTesting...")
prompt = "can you generate a  function to calculate factorial of a number?"
response = generator(prompt, max_new_tokens=100)
print(response[0]['generated_text'])


