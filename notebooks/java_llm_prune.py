import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SPARSITY = 0.05  # 30% pruning
CALIBRATION_SIZE = 80
CALIBRATION_DS = "code_search_net"

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# Load dataset 
print("Loading Java calibration samples...")
dataset = load_dataset("json", data_files={
    "train": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "validation": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "test": "/content/drive/MyDrive/Colab Notebooks/data/test/test_small.jsonl"
})

num_samples = min(CALIBRATION_SIZE, len(dataset["train"]))
texts = [ex["code"] for ex in dataset["train"].select(range(num_samples))]
print(f"Using {num_samples} calibration samples")

# Implement pruning technique
def collect_layer_activations(model, tokenizer, texts):
    """
    Collects activation norms for each Linear layer in MLP and Attention blocks.
    Returns: dict[param_name] = activation scale tensor [in_features]
    """
    torch.manual_seed(42)
    
    #Find all the target layers in the LLM l
    target_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
    target_modules, param_to_module = {}, {}

    # Map all Linear layers of interest
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in target_keywords):
            target_modules[name] = module
            for pname, param in module.named_parameters():
                if pname == "weight":
                    full_pname = f"{name}.weight"
                    param_to_module[full_pname] = name
    
    print(f"Targeting {len(target_modules)} linear layers for pruning")
    activations = {name: [] for name in target_modules.keys()}
    # Install hooks 
    def make_hook(name):
        def hook_fn(module, input, output):
            if not input:
                return
            x = input[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach()
            x = x.reshape(-1, x.size(-1))
            mean_act = x.abs().mean(dim=0).cpu()
            activations[name].append(mean_act)
        return hook_fn

    hooks = []
    for name, module in target_modules.items():
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration data through the model
    print("Collecting activations from calibration data...")
    for text in tqdm(texts, desc="Processing samples"):
        # Tokenization
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad(): # observe only
            model(**inputs)

    for h in hooks:
        h.remove()

    print("Computing activation scales per layer...")
    act_scales = {}
    for name, acts in activations.items():
        if len(acts) > 0:
            act_scales[name] = torch.stack(acts,dim=0).mean(dim=0)  # [in_features]

    param_act_scales = {}
    for pname, mname in param_to_module.items():
        if mname in act_scales:
            param_act_scales[pname] = act_scales[mname]

    return param_act_scales

print("Collecting layer-specific activations...")
param_act_scales = collect_layer_activations(model, tokenizer, texts)


def wanda_prune(model, sparsity, param_act_scales):
    pruned_count = 0
    total_params = 0

    for name, param in model.named_parameters():
        if name in param_act_scales and param.dim() == 2:
            act_scale = param_act_scales[name].to(param.device)

            # Wanda: importance = |W_ij| * a_j (activation per input dim)
            if act_scale.size(0) != param.size(1):
                print(f"Warning: skipping {name} due to activation mismatch")
                continue

            importance = param.abs() * act_scale.unsqueeze(0)  # [out, in]

            # Wanda threshold: global per layer
            # Move to CPU to handle potentially large tensors without GPU memory issues
            flat = importance.flatten()
            k = int((1 - sparsity) * flat.numel())
            if k > 0:
                threshold = torch.topk(flat, k, largest=True).values.min()
            else:
                threshold = flat.max()
            mask = importance >= threshold

            param.data = param.data * mask
            pruned_count += (~mask).sum().item()
            total_params += mask.numel()

    actual_sparsity = pruned_count / total_params if total_params > 0 else 0
    print(f"Pruned {pruned_count:,} / {total_params:,} params ({actual_sparsity:.2%})")
    return model

print("Pruning model with Wanda...")
model = wanda_prune(model, SPARSITY, param_act_scales)

# === SAVE PRUNED MODEL ===
SAVE_DIR = "./QwenCoder3B_JavaPruned-10"
model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Pruned model saved to {SAVE_DIR}")

# === SIMPLE VALIDATION ===
print("\nTesting a short generation to validate pruning...")
prompt = "Write a Java function that returns true if a number is even."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
