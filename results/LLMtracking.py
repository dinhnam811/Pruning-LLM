"""
Per-Question Analysis Script
Tracks average pass rate for each question across all runs per model
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
from collections import defaultdict


def parse_result_file(filepath):
    """Parse a single result text file and extract per-sample pass rates."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract per-sample pass rates
    sample_pass_rates = {}
    sample_pattern = r'SAMPLE (\d+) \(Key: (\d+)\).*?Pass Rate: ([\d.]+)%'
    for match in re.finditer(sample_pattern, content, re.DOTALL):
        sample_idx = int(match.group(1))
        pass_rate = float(match.group(3))
        sample_pass_rates[sample_idx] = pass_rate

    return sample_pass_rates


def collect_all_results(results_dir='.'):
    """Collect all result files organized by model type."""
    # Manually specify the files for each model type
    file_mapping = {
        'Origin': [
            'Origin-Qwen-eval-instruction-follow-1.txt',
            'Origin-Qwen-eval-instruction-follow-2.txt',
            'Original-Qwen-eval-instruction-follow-3.txt',
            'Original-Qwen-eval-instruction-follow-4.txt',
            'Original-Qwen-eval-instruction-follow-5.txt',
            'Original-Qwen-eval-instruction-follow-6.txt',
            'Original-Qwen-eval-instruction-follow-7.txt',
            'Original-Qwen-eval-instruction-follow-8.txt',
            'Original-Qwen-eval-instruction-follow-9.txt',
            'Original-Qwen-eval-instruction-follow-10.txt'
        ],
        '40': [
            '40-Qwen-eval-instruction-follow-1st.txt',
            '40-Qwen-eval-instruction-follow-2rd.txt',
            '40-Qwen-eval-instruction-follow-3rd.txt'
        ],
        '20': [
            '20-Qwen-eval-instruction-follow-1st.txt',
            '20-Qwen-eval-instruction-follow-2nd.txt'
        ],
        '5': [
            '5-Qwen-eval-instruction-follow-1st.txt',
            '5-Qwen-eval-instruction-follow-2nd.txt',
            '5-Qwen-eval-instruction-follow-3rd.txt'
        ]
    }

    results = defaultdict(list)

    for model_type, filenames in file_mapping.items():
        for filename in filenames:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                parsed = parse_result_file(filepath)
                results[model_type].append(parsed)
            else:
                print(f"⚠ Warning: {filename} not found")

    return results


def calculate_per_question_stats(all_results):
    """Calculate average pass rate per question for each model."""
    question_stats = defaultdict(lambda: defaultdict(list))

    for model_type, runs in all_results.items():
        for run in runs:
            for sample_idx, pass_rate in run.items():
                question_stats[sample_idx][model_type].append(pass_rate)

    # Calculate averages
    question_avg_stats = {}
    for sample_idx, model_data in question_stats.items():
        question_avg_stats[sample_idx] = {
            model_type: np.mean(pass_rates)
            for model_type, pass_rates in model_data.items()
        }

    return question_avg_stats


def plot_question_heatmap(question_stats, output_file='question_heatmap.png'):
    """Create heatmap showing pass rates per question per model."""
    question_df = pd.DataFrame(question_stats).T
    question_df = question_df.round(1)

    # Sort columns: Origin first, then others
    cols = sorted(question_df.columns, key=lambda x: (x != 'Origin', x))
    question_df = question_df[cols]

    plt.figure(figsize=(8, 12))
    sns.heatmap(question_df, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Pass Rate (%)'})
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Question ID', fontsize=12)
    plt.title('Per-Question Average Pass Rate Across Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_question_performance_lines(question_stats, output_file='question_performance_lines.png'):
    """Line plot showing each model's performance across questions."""
    question_df = pd.DataFrame(question_stats).T
    question_df = question_df.sort_index()

    # Sort columns: Origin first, then others
    cols = sorted(question_df.columns, key=lambda x: (x != 'Origin', x))

    plt.figure(figsize=(14, 6))
    for model in cols:
        if model in question_df.columns:
            plt.plot(question_df.index, question_df[model], marker='o', label=model, alpha=0.7, linewidth=2)

    plt.xlabel('Question ID', fontsize=12)
    plt.ylabel('Average Pass Rate (%)', fontsize=12)
    plt.title('Model Performance Across Questions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def find_hardest_questions(question_stats, top_n=5):
    """Find the hardest questions for each model."""
    question_df = pd.DataFrame(question_stats).T

    print(f"\n{'='*80}")
    print(f"Top {top_n} Hardest Questions per Model")
    print('='*80)

    for model in sorted(question_df.columns, key=lambda x: (x != 'Origin', x)):
        hardest = question_df[model].nsmallest(top_n)
        print(f"\n{model}:")
        for q_id, score in hardest.items():
            print(f"  Question {q_id}: {score:.1f}%")


def print_question_summary(question_stats):
    """Print per-question statistics summary."""
    question_df = pd.DataFrame(question_stats).T

    # Overall average per question across all models
    question_df['Overall_Avg'] = question_df.mean(axis=1)

    print("\n" + "="*80)
    print("PER-QUESTION STATISTICS SUMMARY")
    print("="*80)
    print("\nFirst 30 Questions:")
    print(question_df.head(30).round(1).to_string())

    print("\n\nQuestion Difficulty Analysis:")
    print("-"*80)

    easiest = question_df['Overall_Avg'].nlargest(5)
    hardest = question_df['Overall_Avg'].nsmallest(5)

    print("\nEasiest Questions (across all models):")
    for q_id, score in easiest.items():
        print(f"  Question {q_id}: {score:.1f}%")

    print("\nHardest Questions (across all models):")
    for q_id, score in hardest.items():
        print(f"  Question {q_id}: {score:.1f}%")

    print("="*80)


if __name__ == "__main__":
    print("Starting Per-Question Analysis...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Collect results
    all_results = collect_all_results(script_dir)

    print("\nCollected results:")
    for model_type, runs in sorted(all_results.items(), key=lambda x: (x[0] != 'Origin', x[0])):
        print(f"  {model_type}: {len(runs)} runs")

    if not all_results:
        print("\n❌ No result files found!")
        exit(1)

    # Calculate per-question statistics
    print("\nCalculating per-question statistics...")
    question_stats = calculate_per_question_stats(all_results)

    # Print summary
    print_question_summary(question_stats)

    print("\n✓ Per-Question Analysis complete!")
