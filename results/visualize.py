import re
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Function to extract evaluation summary data
def parse_eval_summary(text):
    total = re.search(r"Total Samples:\s*(\d+)", text)
    perfect = re.search(r"Perfect.*?:\s*(\d+)", text)
    partial = re.search(r"Partial.*?:\s*(\d+)", text)
    failed = re.search(r"Failed.*?:\s*(\d+)", text)
    avg_pass = re.search(r"Average Pass Rate:\s*([\d.]+)%", text)
    return {
        "total": int(total.group(1)) if total else 0,
        "perfect": int(perfect.group(1)) if perfect else 0,
        "partial": int(partial.group(1)) if partial else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "avg_pass_rate": float(avg_pass.group(1)) if avg_pass else 0,
    }

# Get the directory where the script is located
script_dir = Path(__file__).parent
results_dir = script_dir

# Organize files by version
versions = {
    "Qwen-5": [],
    "Qwen-20": [],
    "Qwen-40": [],
    "Original": []
}

# Scan for all evaluation files
for file in os.listdir(results_dir):
    if file.endswith(".txt") and "Qwen" in file:
        file_path = results_dir / file
        if file.startswith("5-"):
            versions["Qwen-5"].append(str(file_path))
        elif file.startswith("20-"):
            versions["Qwen-20"].append(str(file_path))
        elif file.startswith("40-"):
            versions["Qwen-40"].append(str(file_path))
        elif file.startswith("Origin"):
            versions["Original"].append(str(file_path))

# Parse all files for each version
version_data = {}
for version_name, file_paths in versions.items():
    if not file_paths:
        continue

    runs_data = []
    for path in sorted(file_paths):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        runs_data.append(parse_eval_summary(text))

    version_data[version_name] = runs_data

# Calculate max pass rate for each version
max_pass_rates = {}
for version_name, runs in version_data.items():
    max_pass_rates[version_name] = max([run["avg_pass_rate"] for run in runs])

# Plot 1: Max Pass Rate Comparison
plt.figure(figsize=(10, 6))
versions_list = list(max_pass_rates.keys())
max_rates = list(max_pass_rates.values())
colors_bar = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]
bars = plt.bar(versions_list, max_rates, color=colors_bar, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel("Pass Rate (%)", fontsize=12)
plt.xlabel("Model Version", fontsize=12)
plt.title("Maximum Pass Rate Comparison Across Qwen Versions", fontsize=14, fontweight='bold')

# Dynamic y-axis: set range based on data with some padding
min_rate = min(max_rates)
max_rate = max(max_rates)
range_padding = (max_rate - min_rate) * 0.2  # 20% padding
plt.ylim(0,100)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(results_dir / "max_pass_rate_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 2-5: Individual bar charts for each version showing all runs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors_stacked = ["#2ecc71", "#f1c40f", "#e74c3c"]
version_names = list(version_data.keys())

for idx, (version_name, runs) in enumerate(version_data.items()):
    ax = axes[idx]

    # Prepare data for this version
    run_labels = [f"Run {i+1}" for i in range(len(runs))]
    pass_rates = [run["avg_pass_rate"] for run in runs]

    # Create simple bar chart (same style as main comparison)
    x_pos = range(len(run_labels))
    bars = ax.bar(x_pos, pass_rates, color=colors_bar[idx], edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')


    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title(f"{version_name} - Evaluation Results", fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(results_dir / "individual_version_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
for version_name, runs in version_data.items():
    print(f"\n{version_name}:")
    for i, run in enumerate(runs, 1):
        print(f"  Run {i}: Pass Rate = {run['avg_pass_rate']:.1f}% "
              f"(Perfect: {run['perfect']}, Partial: {run['partial']}, Failed: {run['failed']})")
    max_rate = max([run["avg_pass_rate"] for run in runs])
    avg_rate = sum([run["avg_pass_rate"] for run in runs]) / len(runs)
    print(f"  Max: {max_rate:.1f}% | Average: {avg_rate:.1f}%")
print("="*80)
