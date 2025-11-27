import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "distilgpt2"   # LLM nhỏ, giống foundation model thu nhỏ
SPARSITY = 0.3             # 30% sparsity (prune 30% weight)
DEVICE = "cpu"  

# ================================
# STEP 2: LOAD MODEL + TOKENIZER
# (Paper: "Model selection")
# ================================
print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()  # chế độ inference, không train

print("Model loaded.")
print("Number of transformer blocks:", len(model.transformer.h))
# ================================
# STEP 3: CHỌN LAYER ĐỂ PRUNE
# ================================
target_block = model.transformer.h[1]       # BLOCK số 1
target_layer = target_block.mlp.c_fc        # Linear layer đầu của MLP
weight = target_layer.weight                # Ma trận weight cần prune

print("Target BLOCK:", target_block)
print("Pruning Layer:", target_layer)
print("Weight shape:", weight.shape)
# ================================
# STEP 4: THU THẬP ACTIVATIONS
# ================================
activations = []  # để lưu tất cả activation

def hook_fn(module, input, output):
    """
    Hook được gọi mỗi lần layer target_layer chạy forward.
    Wanda cần activations ở ĐẦU VÀO layer.
    - input[0]: tensor [batch, seq_len, hidden]
    """
    x = input[0].detach()               # tách khỏi graph, không gradient
    x = x.reshape(-1, x.size(-1))       # gom batch và seq_len → [N, hidden]
    activations.append(x)

# gắn hook vào layer c_fc
hook = target_layer.register_forward_hook(hook_fn)

# calibration dataset nhỏ:
calib_texts = [
    "This is a simple example for pruning a language model.",
    "We are collecting activations from the first MLP layer.",
    "Large language models can be pruned to be smaller and faster.",
    "Domain-specific pruning can create coding-only sub-models.",
]

print("Collecting activations...")
with torch.no_grad():
    for text in calib_texts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        _ = model(**inputs)

# gỡ hook
hook.remove()

# nối tất cả activation lại thành một tensor lớn
all_acts = torch.cat(activations, dim=0)  # [N, hidden]
print("Collected activations shape:", all_acts.shape)
# ================================
# STEP 5: TÍNH act_scale (NORM CỦA ACTIVATION)
# ================================
# all_acts: [N_tokens, hidden_dim] = [51, 768]
# Ta muốn 1 con số cho mỗi hidden feature -> vector [768]
act_scale = all_acts.abs().mean(dim=0)  # [hidden_dim]
print("Activation scale shape:", act_scale.shape)
# ================================
# STEP 6: TÍNH ĐỘ QUAN TRỌNG CHO TỪNG WEIGHT (WANDA)
# ================================
W = weight.data  # [in_features, out_features] = [768, 3072]

# Wanda: importance_ij = |W_ij| * act_scale_i
# -> act_scale: [in_features] -> reshape thành [in_features, 1] để nhân theo hàng
importance = W.abs() * act_scale.unsqueeze(1)  # [768, 3072]

print("Weight shape:", W.shape)
print("Importance shape:", importance.shape)
# ================================
# STEP 7: ĐO SPARSITY TRƯỚC KHI PRUNE
# ================================
def count_zeros(t: torch.Tensor) -> int:
    return (t == 0).sum().item()

total_params = W.numel()              # tổng số phần tử trong ma trận W
zero_before = count_zeros(W)          # số phần tử đang = 0

print(f"Before pruning: zeros = {zero_before} / {total_params} "
      f"({zero_before / total_params:.2%})")
# ================================
# STEP 8: CHỌN NGƯỠNG ĐỂ PRUNE (DỰA TRÊN IMPORTANCE)
# ================================
num_prune = int(SPARSITY * total_params)
print(f"Target sparsity: {SPARSITY:.0%} -> prune {num_prune} weights")

# Flatten importance để chọn trực tiếp trên vector 1D
importance_flat = importance.view(-1)  # [768 * 3072]

# Lấy num_prune phần tử NHỎ NHẤT (ít quan trọng nhất)
# largest=False => lấy từ nhỏ đến lớn
threshold_value = torch.topk(
    importance_flat,
    k=num_prune,
    largest=False
).values.max()

# Tạo mask: True = giữ lại, False = prune
mask = importance > threshold_value     # [768, 3072]
# ================================
# STEP 9: ÁP DỤNG MASK ĐỂ PRUNE
# ================================
W_pruned = W * mask           # chỗ nào mask=False -> W_pruned = 0
weight.data = W_pruned        # gán lại vào layer

zero_after = count_zeros(weight.data)
print(f"After pruning: zeros = {zero_after} / {total_params} "
      f"({zero_after / total_params:.2%})")
# ================================
# STEP 10: THỬ GENERATE TEXT SAU PRUNE
# ================================
def generate_text(prompt: str, max_new_tokens: int = 40) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = "The goal of model pruning is"
print("\n=== Generated text after pruning ===")
print(generate_text(test_prompt))

