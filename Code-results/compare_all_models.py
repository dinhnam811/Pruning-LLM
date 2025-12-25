"""
Compare all 4 models side by side
Each chart type shows all models together for easy comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
sns.set_style("whitegrid")
COLORS = {
    'Accepted': '#2ecc71',
    'Wrong Answer': '#e74c3c',
    'Time Limit Exceeded': '#f39c12',
    'Runtime Error': '#e67e22',
    'Compilation Error': '#c0392b'
}

# Model configurations
MODELS = [
    {'file': 'OriginalStatic.xlsx', 'name': 'Origin'},
    {'file': '5QwenStatic.xlsx', 'name': '5% Prune'},
    {'file': '20QwenStatic.xlsx', 'name': '20% Prune'},
    {'file': '40QwenStatic.xlsx', 'name': '40% Prune'}
]

# Load all model data
print("Loading data for all models...")
model_data = {}

for model in MODELS:
    filepath = f"Code-results/{model['file']}"
    try:
        df = pd.read_excel(filepath)
        df["Status"] = df["Status"].str.strip()
        df["Difficulty"] = df["Difficulty"].str.strip()
        df["Runtime(ms)"] = pd.to_numeric(df["Runtime(ms)"], errors='coerce')
        df["Memory(MB)"] = pd.to_numeric(df["Memory(MB)"], errors='coerce')
        model_data[model['name']] = df
        print(f"✓ Loaded {model['name']}: {len(df)} problems")
    except Exception as e:
        print(f"✗ Error loading {model['name']}: {e}")

if len(model_data) == 0:
    print("No data loaded!")
    exit(1)

# =========================
# CHART 1: Success Rate Pie Charts (2x2 grid)
# =========================
print("\nGenerating success rate comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Overall Success Rate - Model Comparison',
             fontsize=16, fontweight='bold', x=0.5, y=0.98)

axes = axes.flatten()
model_names = ['Origin', '5% Prune', '20% Prune', '40% Prune']

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        accepted = df[df["Status"] == "Accepted"]
        success_counts = [len(accepted), len(df) - len(accepted)]
        colors_pie = ['#2ecc71', '#e74c3c']

        wedges, texts, autotexts = axes[idx].pie(
            success_counts,
            labels=['Accepted', 'Failed'],
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0)
        )

        axes[idx].set_title(model_name, fontsize=13, fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = "Code-results/comparison_success_rate.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# CHART 2: Status Distribution Bar Charts (2x2 grid)
# =========================
print("Generating status distribution comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Status Distribution - Model Comparison',
             fontsize=16, fontweight='bold', x=0.5, y=0.98)

axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        status_counts = df["Status"].value_counts()
        colors = [COLORS.get(status, '#95a5a6') for status in status_counts.index]

        bars = axes[idx].bar(range(len(status_counts)), status_counts.values,
                            color=colors, alpha=0.8, edgecolor='black')

        axes[idx].set_xticks(range(len(status_counts)))
        axes[idx].set_xticklabels(status_counts.index, rotation=45, ha='right')
        axes[idx].set_title(model_name, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel("Number of Problems")
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim(0, max(status_counts.values) * 1.15)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}\n({height/len(df)*100:.1f}%)',
                          ha='center', va='bottom', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = "Code-results/comparison_status_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# CHART 3: Status by Difficulty Stacked Bar (2x2 grid)
# =========================
print("Generating status by difficulty comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Status by Difficulty - Model Comparison',
             fontsize=16, fontweight='bold', x=0.5, y=0.98)

axes = axes.flatten()
difficulty_order = ['Easy', 'Medium', 'Hard']

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        pivot = pd.pivot_table(df, index="Difficulty", columns="Status",
                              values="Problem", aggfunc="count", fill_value=0)
        pivot = pivot.reindex([d for d in difficulty_order if d in pivot.index])

        pivot.plot(kind="bar", stacked=True, ax=axes[idx],
                  color=[COLORS.get(col, '#95a5a6') for col in pivot.columns],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

        axes[idx].set_title(model_name, fontsize=13, fontweight='bold')
        axes[idx].set_xlabel("Difficulty")
        axes[idx].set_ylabel("Number of Problems")
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
        axes[idx].legend(loc='upper right', fontsize=7)
        axes[idx].grid(axis='y', alpha=0.3)

        # Add total count labels
        for i, difficulty in enumerate(pivot.index):
            total = pivot.loc[difficulty].sum()
            axes[idx].text(i, total + 0.3, f'n={int(total)}',
                          ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = "Code-results/comparison_difficulty_status.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# CHART 4: Success Rate by Difficulty (2x2 grid)
# =========================
print("Generating success rate by difficulty comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Success Rate by Difficulty - Model Comparison',
             fontsize=16, fontweight='bold', x=0.5, y=0.98)

axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        difficulty_stats = df.groupby('Difficulty').apply(
            lambda x: (x['Status'] == 'Accepted').sum() / len(x) * 100
        ).reindex([d for d in difficulty_order if d in df['Difficulty'].unique()])

        bars = axes[idx].bar(range(len(difficulty_stats)), difficulty_stats.values,
                            color=['#3498db', '#9b59b6', '#e74c3c'],
                            alpha=0.8, edgecolor='black')

        axes[idx].set_xticks(range(len(difficulty_stats)))
        axes[idx].set_xticklabels(difficulty_stats.index, rotation=0)
        axes[idx].set_title(model_name, fontsize=13, fontweight='bold')
        axes[idx].set_ylabel("Success Rate (%)")
        axes[idx].set_ylim(0, 100)
        axes[idx].axhline(y=50, color='red', linestyle='--',
                         alpha=0.5, label='50% threshold')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].legend(fontsize=7)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, difficulty_stats.values)):
            count = len(df[df['Difficulty'] == difficulty_stats.index[i]])
            axes[idx].text(bar.get_x() + bar.get_width()/2., val + 2,
                          f'{val:.1f}%\n(n={count})',
                          ha='center', va='bottom', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_file = "Code-results/comparison_success_by_difficulty.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# SUMMARY TABLE
# =========================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"{'Model':<15} {'Total':<8} {'Accepted':<10} {'Success Rate':<15} {'Avg Runtime':<15} {'Avg Memory'}")
print("-"*80)

for model_name in model_names:
    if model_name in model_data:
        df = model_data[model_name]
        accepted = df[df["Status"] == "Accepted"]
        total = len(df)
        accepted_count = len(accepted)
        success_rate = (accepted_count / total) * 100 if total > 0 else 0

        if len(accepted) > 0:
            avg_runtime = accepted['Runtime(ms)'].mean()
            avg_memory = accepted['Memory(MB)'].mean()
            print(f"{model_name:<15} {total:<8} {accepted_count:<10} {success_rate:<15.1f} "
                  f"{avg_runtime:<15.2f} {avg_memory:.2f}")
        else:
            print(f"{model_name:<15} {total:<8} {accepted_count:<10} {success_rate:<15.1f} "
                  f"{'N/A':<15} {'N/A'}")

print("="*80)
print("\n✓ All comparison charts generated successfully!")
print("\nGenerated files:")
print("  - comparison_success_rate.png")
print("  - comparison_status_distribution.png")
print("  - comparison_difficulty_status.png")
print("  - comparison_success_by_difficulty.png")
