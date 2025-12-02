### Step 1: Clone llama.cpp
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```
### Step 2: Convert to gguf
```
python convert_hf_to_gguf.py \
    --model "LLM-folder-path" \
    --outfile "{destination_of_gguf}/qwen-pruned.gguf" \
    --outtype f16
```