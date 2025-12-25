import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# =========================
# ENHANCED CONFIGURATION
# =========================
sns.set_style("whitegrid")

# Global font size increases
plt.rcParams['font.size'] = 18             # General text
plt.rcParams['axes.titlesize'] = 22        # Subplot titles
plt.rcParams['axes.labelsize'] = 18        # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 16       # X tick labels
plt.rcParams['ytick.labelsize'] = 16       # Y tick labels
plt.rcParams['legend.fontsize'] = 16       # Legend text
plt.rcParams['figure.titlesize'] = 26      # Main super title

COLORS = {
    'Accepted': '#2ecc71',
    'Wrong Answer': '#e74c3c',
    'Time Limit Exceeded': '#f39c12',
    'Runtime Error': '#e67e22',
    'Compilation Error': '#c0392b'
}

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
# CHART 1: Success Rate Pie Charts
# =========================
print("\nGenerating success rate comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Overall Success Rate: Model Comparison', fontweight='bold', y=0.98)

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
            explode=(0.05, 0),
            textprops={'fontsize': 16}
        )

        axes[idx].set_title(model_name, fontweight='bold', pad=20)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(18) # Increased percentage font
            autotext.set_fontweight('bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Code-results/comparison_success_rate.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# CHART 2: Status Distribution Bar Charts
# =========================
print("Generating status distribution comparison...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Status Distribution: Model Comparison', fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        status_counts = df["Status"].value_counts()
        colors = [COLORS.get(status, '#95a5a6') for status in status_counts.index]

        bars = axes[idx].bar(range(len(status_counts)), status_counts.values,
                            color=colors, alpha=0.8, edgecolor='black')

        axes[idx].set_xticks(range(len(status_counts)))
        axes[idx].set_xticklabels(status_counts.index, rotation=35, ha='right', fontsize=18)
        axes[idx].set_title(model_name, fontweight='bold', pad=15)
        axes[idx].set_ylabel("Number of Problems")
        axes[idx].set_ylim(0, max(status_counts.values) * 1.25)

        # Add larger value labels
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{int(height)}\n({height/len(df)*100:.1f}%)',
                          ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Code-results/comparison_status_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# CHART 3: Status by Difficulty Stacked Bar
# =========================
print("Generating status by difficulty comparison...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Status by Difficulty: Model Comparison', fontweight='bold', y=0.98)

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

        axes[idx].set_title(model_name, fontweight='bold', pad=15)
        axes[idx].set_xlabel("Difficulty")
        axes[idx].set_ylabel("Number of Problems")
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
        axes[idx].legend(loc='upper right', fontsize=12, frameon=True)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Code-results/comparison_difficulty_status.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# CHART 4: Success Rate by Difficulty
# =========================
print("Generating success rate by difficulty comparison...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Success Rate by Difficulty: Model Comparison', fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    if model_name in model_data:
        df = model_data[model_name]
        diff_order_local = [d for d in difficulty_order if d in df['Difficulty'].unique()]
        difficulty_stats = df.groupby('Difficulty').apply(
            lambda x: (x['Status'] == 'Accepted').sum() / len(x) * 100
        ).reindex(diff_order_local)

        bars = axes[idx].bar(range(len(difficulty_stats)), difficulty_stats.values,
                            color=['#3498db', '#9b59b6', '#e74c3c'],
                            alpha=0.8, edgecolor='black')

        axes[idx].set_xticks(range(len(difficulty_stats)))
        axes[idx].set_xticklabels(difficulty_stats.index, rotation=0)
        axes[idx].set_title(model_name, fontweight='bold', pad=15)
        axes[idx].set_ylabel("Success Rate (%)")
        axes[idx].set_ylim(0, 115) 
        axes[idx].axhline(y=50, color='black', linestyle='--', alpha=0.3)
        axes[idx].grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, difficulty_stats.values)):
            count = len(df[df['Difficulty'] == difficulty_stats.index[i]])
            axes[idx].text(bar.get_x() + bar.get_width()/2., val + 2,
                          f'{val:.1f}%\n(n={count})',
                          ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Code-results/comparison_success_by_difficulty.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Comparison charts with enlarged text generated successfully!")