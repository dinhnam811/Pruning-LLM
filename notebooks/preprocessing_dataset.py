import json
from pathlib import Path

def reduce_jsonl(input_path, output_path, max_lines=100):
    """Keep only first max_lines from JSONL file"""
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i >= max_lines:
                break
            f_out.write(line)
    
    print(f"✓ Reduced {input_path} to {max_lines} lines → {output_path}")

# Reduce all splits
data_dir = Path("data")
splits = ["train", "val", "test"]

for split in splits:
    input_file = data_dir / split / f"{split}.jsonl"
    output_file = data_dir / split / f"{split}_small.jsonl"
    
    if input_file.exists():
        reduce_jsonl(input_file, output_file, max_lines=256)
    else:
        print(f"⚠ File not found: {input_file}")

print("\n✓ Done! Use these files:")
print("  data/train/train_small.jsonl")
print("  data/val/val_small.jsonl") 
print("  data/test/test_small.jsonl")