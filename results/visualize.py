import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract evaluation summary data
def parse_eval_summary(text):
    total = re.search(r"Total Samples:\s*(\d+)", text)
    perfect = re.search(r"Perfect.*?:\s*(\d+)", text)
    partial = re.search(r"Partial.*?:\s*(\d+)", text)
    failed = re.search(r"Failed.*?:\s*(\d+)", text)
    return {
        "total": int(total.group(1)) if total else 0,
        "perfect": int(perfect.group(1)) if perfect else 0,
        "partial": int(partial.group(1)) if partial else 0,
        "failed": int(failed.group(1)) if failed else 0,
    }

# Load your .txt files (update paths as needed)
files = {
    "instruction-eval-llm-20.txt": "C:/Users/namnd/Documents/Pruning-LLM-1/results/instruction-eval-llm-20.txt",
    "instruction-eval-llm-40.txt": "C:/Users/namnd/Documents/Pruning-LLM-1/results/instruction-eval-llm-40.txt",
    "instruction-eval-llm-original.txt": "C:/Users/namnd/Documents/Pruning-LLM-1/results/instruction-eval-llm-original.txt"
}

data = {}
for name, path in files.items():
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    data[name] = parse_eval_summary(text)

# Convert to DataFrame
df = pd.DataFrame(data).T
df["pass_rate"] = (df["perfect"] + df["partial"]) / df["total"] * 100

# Normalize to percentages for stacked visualization
df_stacked = df[["perfect", "partial", "failed"]].div(df["total"], axis=0) * 100

# Plot 1️⃣ — Pass Rate as stacked bar (percentages)
plt.figure(figsize=(8, 5))
bottom = None
colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
for i, col in enumerate(["perfect", "partial", "failed"]):
    plt.bar(df_stacked.index, df_stacked[col], bottom=bottom, label=col.capitalize(), color=colors[i])
    bottom = df_stacked[col] if bottom is None else bottom + df_stacked[col]
plt.ylabel("Percentage (%)")
plt.title("Stacked Pass Rate by Evaluation File")
plt.legend(title="Result Type")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2️⃣ — Raw counts stacked (absolute numbers)
plt.figure(figsize=(8, 5))
bottom = None
for i, col in enumerate(["perfect", "partial", "failed"]):
    plt.bar(df.index, df[col], bottom=bottom, label=col.capitalize(), color=colors[i])
    bottom = df[col] if bottom is None else bottom + df[col]
plt.ylabel("Count")
plt.title("Stacked Evaluation Results (Perfect / Partial / Failed)")
plt.legend(title="Result Type")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
