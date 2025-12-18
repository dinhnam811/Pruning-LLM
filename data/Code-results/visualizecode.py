import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# =========================
# Configuration
# =========================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 10
COLORS = {'Accepted': '#2ecc71', 'Wrong Answer': '#e74c3c', 
          'Time Limit Exceeded': '#f39c12', 'Runtime Error': '#e67e22',
          'Compilation Error': '#c0392b'}

# =========================
# Load Excel file
# =========================
print("Loading data...")
df = pd.read_excel("C:/Users/namnd/Documents/Pruning-LLM-1/data/40QwenStatic.xlsx")

# Clean up
df["Status"] = df["Status"].str.strip()
df["Difficulty"] = df["Difficulty"].str.strip()

# Convert numeric columns
df["Runtime(ms)"] = pd.to_numeric(df["Runtime(ms)"], errors='coerce')
df["Memory(MB)"] = pd.to_numeric(df["Memory(MB)"], errors='coerce')

# Print summary statistics
print("\n" + "="*80)
print("QWEN MODEL - LEETCODE PERFORMANCE SUMMARY (30 Problems)")
print("="*80)
print(f"Total Problems: {len(df)}")
print(f"\nStatus Breakdown:")
for status, count in df["Status"].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {status:25s}: {count:2d} ({percentage:5.1f}%)")

print(f"\nDifficulty Breakdown:")
for difficulty, count in df["Difficulty"].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {difficulty:10s}: {count:2d} ({percentage:5.1f}%)")

accepted = df[df["Status"] == "Accepted"]
if len(accepted) > 0:
    print(f"\nAccepted Problems Performance:")
    print(f"  Avg Runtime: {accepted['Runtime(ms)'].mean():.2f} ms")
    print(f"  Avg Memory:  {accepted['Memory(MB)'].mean():.2f} MB")
print("="*80 + "\n")

# =========================
# Create comprehensive visualization
# =========================
fig = plt.figure(figsize=(14, 10))

# 1. Status distribution with percentages
ax1 = plt.subplot(3, 3, 1)
status_counts = df["Status"].value_counts()
colors = [COLORS.get(status, '#95a5a6') for status in status_counts.index]
bars = ax1.bar(range(len(status_counts)), status_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(status_counts)))
ax1.set_xticklabels(status_counts.index, rotation=45, ha='right')
ax1.set_title("Status Distribution", fontsize=12, fontweight='bold')
ax1.set_ylabel("Number of Problems")
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)

# 2. Success rate pie chart
ax2 = plt.subplot(3, 3, 2)
success_counts = [len(accepted), len(df) - len(accepted)]
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax2.pie(success_counts, 
                                     labels=['Accepted', 'Failed'],
                                     colors=colors_pie,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     explode=(0.05, 0))
ax2.set_title("Overall Success Rate", fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# 3. Difficulty vs Status stacked bar
ax3 = plt.subplot(3, 3, 4)
pivot = pd.pivot_table(df, index="Difficulty", columns="Status", 
                       values="Problem", aggfunc="count", fill_value=0)
# Reorder difficulty
difficulty_order = ['Easy', 'Medium', 'Hard']
pivot = pivot.reindex([d for d in difficulty_order if d in pivot.index])

pivot.plot(kind="bar", stacked=True, ax=ax3, 
          color=[COLORS.get(col, '#95a5a6') for col in pivot.columns],
          alpha=0.8, edgecolor='black', linewidth=0.5)
ax3.set_title("Status by Difficulty", fontsize=12, fontweight='bold')
ax3.set_xlabel("Difficulty")
ax3.set_ylabel("Number of Problems")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# 4. Success rate by difficulty
ax4 = plt.subplot(3, 3, 5)
difficulty_stats = df.groupby('Difficulty').apply(
    lambda x: (x['Status'] == 'Accepted').sum() / len(x) * 100
).reindex([d for d in difficulty_order if d in df['Difficulty'].unique()])

bars = ax4.bar(range(len(difficulty_stats)), difficulty_stats.values, 
               color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(difficulty_stats)))
ax4.set_xticklabels(difficulty_stats.index, rotation=0)
ax4.set_title("Success Rate by Difficulty", fontsize=12, fontweight='bold')
ax4.set_ylabel("Success Rate (%)")
ax4.set_ylim(0, 100)
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax4.grid(axis='y', alpha=0.3)
ax4.legend()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, difficulty_stats.values)):
    count = len(df[df['Difficulty'] == difficulty_stats.index[i]])
    ax4.text(bar.get_x() + bar.get_width()/2., val + 2,
            f'{val:.1f}%\n(n={count})',
            ha='center', va='bottom', fontsize=9)

# Adjust spacing between subplots
plt.tight_layout(h_pad=3.5, w_pad=3.0)

# Save figure with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"qwen_leetcode_analysis_{timestamp}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_file}")

plt.show()

# =========================
# Additional detailed tables
# =========================
print("\nDetailed Statistics by Difficulty:")
print("-" * 80)
for difficulty in difficulty_order:
    if difficulty in df['Difficulty'].unique():
        diff_df = df[df['Difficulty'] == difficulty]
        accepted_count = len(diff_df[diff_df['Status'] == 'Accepted'])
        success_rate = (accepted_count / len(diff_df)) * 100
        print(f"\n{difficulty}:")
        print(f"  Total: {len(diff_df)}")
        print(f"  Accepted: {accepted_count} ({success_rate:.1f}%)")
        
        if accepted_count > 0:
            accepted_diff = diff_df[diff_df['Status'] == 'Accepted']
            print(f"  Avg Runtime: {accepted_diff['Runtime(ms)'].mean():.2f} ms")
            print(f"  Avg Memory: {accepted_diff['Memory(MB)'].mean():.2f} MB")
