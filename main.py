import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "distilgpt2"   # LLM nhỏ, giống foundation model thu nhỏ
SPARSITY = 0.5              # 50% sparsity (prune 50% weight)
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
target_block = model.transformer.h[0]       # BLOCK số 0
target_layer = target_block.mlp.c_fc        # Linear layer đầu của MLP
weight = target_layer.weight                # Ma trận weight cần prune

print("Target BLOCK:", target_block)
print("Pruning Layer:", target_layer)
print("Weight shape:", weight.shape)
