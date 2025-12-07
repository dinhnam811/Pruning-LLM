import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SPARSITY = 0.2  # Percent of pruning
CALIBRATION_SIZE = 126
SAVE_DIR = "./QwenCoder3B_JavaPruned-20"

# Load Model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
print("Load done")
model.eval() # switch to evaluation mode


# Load Calibration Data
print("Loading Java calibration samples...")
dataset = load_dataset("json", data_files={
    "train": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "validation": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "test": "/content/drive/MyDrive/Colab Notebooks/data/test/test_small.jsonl"
})
texts = [ex["code"] for ex in dataset["train"].select(range(min(CALIBRATION_SIZE, len(dataset["train"]))))]
print(f"Using {len(texts)} calibration samples")

# Collect Layer Activations (Wanda Technique)
def collect_activations(model, tokenizer, texts):
    """Collects activation scales for target Linear layers using Wanda technique."""
    torch.manual_seed(42)
    target_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
    target_modules = {}
    param_to_module = {}

    # Find target layers and store to a dict
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in target_keywords):
            target_modules[name] = module # map layer object to install hook
            param_to_module[f"{name}.weight"] = name # map param name to module name

    print(f"Targeting {len(target_modules)} layers for pruning")
    activations = {name: [] for name in target_modules}

    # Register hooks to capture activations
    def make_hook(layer_name):
        def hook(module, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                act = inp[0].detach().reshape(-1, inp[0].size(-1)).abs().mean(dim=0).cpu()
                activations[layer_name].append(act)
        return hook

    hooks = [module.register_forward_hook(make_hook(name)) for name, module in target_modules.items()]

    # Run calibration data
    print("Collecting activations...")
    for text in tqdm(texts, desc="Processing"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average activations across all samples
    print("Computing activation scalevs...")
    param_act_scales = {}
    for pname, mname in param_to_module.items():
        if activations[mname]:
            param_act_scales[pname] = torch.stack(activations[mname]).mean(dim=0)

    return param_act_scales

# Wanda Pruning
def wanda_prune(model, sparsity, param_act_scales):
    """Prunes model using Wanda (Weights AND Activations) technique."""
    pruned_count = total_params = 0

    for name, param in model.named_parameters():
        if name in param_act_scales and param.dim() == 2: # Only prune Linear weights
            act_scale = param_act_scales[name].to(param.device)

            # Verify dimensions match (act scale might be on CPU)
            if act_scale.size(0) != param.size(1):
                continue

            # Calculate importance: |Weight| × Activation
            importance = param.abs() * act_scale.unsqueeze(0)

            # Find threshold (keep top (1-sparsity)% most important)
            k = int((1 - sparsity) * importance.numel())
            threshold = torch.topk(importance.flatten(), k, largest=True).values.min() if k > 0 else importance.max()

            # Apply mask (keep important, zero out unimportant)
            mask = importance >= threshold
            param.data *= mask

            # Track statistics
            pruned_count += (~mask).sum().item()
            total_params += mask.numel()

    print(f"Pruned {pruned_count:,} / {total_params:,} params ({pruned_count/total_params:.2%})")
    return model

# Execute Pruning
print("\n" + "="*50)
param_act_scales = collect_activations(model, tokenizer, texts)
print("="*50 + "\n")
model = wanda_prune(model, SPARSITY, param_act_scales)

# Save Pruned Model
print(f"\nSaving to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model saved successfully!")
