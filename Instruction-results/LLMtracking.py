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


def parse_result_file_with_constraints(filepath):
    """Parse a single result text file and extract individual constraint results."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract constraint-level results
    constraint_results = {}

    # Pattern to find each sample block
    sample_blocks = re.split(r'SAMPLE (\d+) \(Key: \d+\)', content)[1:]  # Skip first empty split

    for i in range(0, len(sample_blocks), 2):
        if i + 1 >= len(sample_blocks):
            break

        sample_idx = int(sample_blocks[i])
        sample_content = sample_blocks[i + 1]

        # Extract constraint results (✓ PASS or ✗ FAIL)
        constraint_pattern = r'([✓✗]) (PASS|FAIL) - ([\w:_]+)'
        for match in re.finditer(constraint_pattern, sample_content):
            status = match.group(2)  # PASS or FAIL
            constraint_name = match.group(3)  # e.g., "punctuation:no_comma"

            key = f"Q{sample_idx}-{constraint_name}"
            # Convert PASS to 100, FAIL to 0
            constraint_results[key] = 100.0 if status == "PASS" else 0.0

    return constraint_results


def collect_all_results(results_dir='.'):
    """Collect all result files organized by model type."""
    # Manually specify the files for each model type
    file_mapping = {
        'Origin': [
            'Original-Qwen-eval-instruction-follow-1.txt',
            'Original-Qwen-eval-instruction-follow-2.txt',
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
            '40-Qwen-eval-instruction-follow-1.txt',
            '40-Qwen-eval-instruction-follow-2.txt',
            '40-Qwen-eval-instruction-follow-3.txt',
            '40-Qwen-eval-instruction-follow-4.txt',
            '40-Qwen-eval-instruction-follow-5.txt',
            '40-Qwen-eval-instruction-follow-6.txt',
            '40-Qwen-eval-instruction-follow-7.txt',
            '40-Qwen-eval-instruction-follow-8.txt',
            '40-Qwen-eval-instruction-follow-9.txt',
            '40-Qwen-eval-instruction-follow-10.txt'
        ],
        '20': [
            '20-Qwen-eval-instruction-follow-1.txt',
            '20-Qwen-eval-instruction-follow-2.txt',
            '20-Qwen-eval-instruction-follow-3.txt',
            '20-Qwen-eval-instruction-follow-4.txt',
            '20-Qwen-eval-instruction-follow-5.txt',
            '20-Qwen-eval-instruction-follow-6.txt',
            '20-Qwen-eval-instruction-follow-7.txt',
            '20-Qwen-eval-instruction-follow-8.txt',
            '20-Qwen-eval-instruction-follow-9.txt',
            '20-Qwen-eval-instruction-follow-10.txt'
        ],
        '5': [
            '5-Qwen-eval-instruction-follow-1.txt',
            '5-Qwen-eval-instruction-follow-2.txt',
            '5-Qwen-eval-instruction-follow-3.txt',
            '5-Qwen-eval-instruction-follow-4.txt',
            '5-Qwen-eval-instruction-follow-5.txt',
            '5-Qwen-eval-instruction-follow-6.txt',
            '5-Qwen-eval-instruction-follow-7.txt',
            '5-Qwen-eval-instruction-follow-8.txt',
            '5-Qwen-eval-instruction-follow-9.txt',
            '5-Qwen-eval-instruction-follow-10.txt'
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


def collect_constraint_results(results_dir='.'):
    """Collect constraint-level results from all files."""
    file_mapping = {
        'Origin': [
            'Original-Qwen-eval-instruction-follow-1.txt',
            'Original-Qwen-eval-instruction-follow-2.txt',
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
            '40-Qwen-eval-instruction-follow-1.txt',
            '40-Qwen-eval-instruction-follow-2.txt',
            '40-Qwen-eval-instruction-follow-3.txt',
            '40-Qwen-eval-instruction-follow-4.txt',
            '40-Qwen-eval-instruction-follow-5.txt',
            '40-Qwen-eval-instruction-follow-6.txt',
            '40-Qwen-eval-instruction-follow-7.txt',
            '40-Qwen-eval-instruction-follow-8.txt',
            '40-Qwen-eval-instruction-follow-9.txt',
            '40-Qwen-eval-instruction-follow-10.txt'
        ],
        '20': [
            '20-Qwen-eval-instruction-follow-1.txt',
            '20-Qwen-eval-instruction-follow-2.txt',
            '20-Qwen-eval-instruction-follow-3.txt',
            '20-Qwen-eval-instruction-follow-4.txt',
            '20-Qwen-eval-instruction-follow-5.txt',
            '20-Qwen-eval-instruction-follow-6.txt',
            '20-Qwen-eval-instruction-follow-7.txt',
            '20-Qwen-eval-instruction-follow-8.txt',
            '20-Qwen-eval-instruction-follow-9.txt',
            '20-Qwen-eval-instruction-follow-10.txt'
        ],
        '5': [
            '5-Qwen-eval-instruction-follow-1.txt',
            '5-Qwen-eval-instruction-follow-2.txt',
            '5-Qwen-eval-instruction-follow-3.txt',
            '5-Qwen-eval-instruction-follow-4.txt',
            '5-Qwen-eval-instruction-follow-5.txt',
            '5-Qwen-eval-instruction-follow-6.txt',
            '5-Qwen-eval-instruction-follow-7.txt',
            '5-Qwen-eval-instruction-follow-8.txt',
            '5-Qwen-eval-instruction-follow-9.txt',
            '5-Qwen-eval-instruction-follow-10.txt'
        ]
    }

    results = defaultdict(list)

    for model_type, filenames in file_mapping.items():
        for filename in filenames:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                parsed = parse_result_file_with_constraints(filepath)
                results[model_type].append(parsed)

    return results


def calculate_constraint_stats(all_constraint_results):
    """Calculate average pass rate per constraint for each model."""
    constraint_stats = defaultdict(lambda: defaultdict(list))

    for model_type, runs in all_constraint_results.items():
        for run in runs:
            for constraint_key, pass_fail in run.items():
                constraint_stats[constraint_key][model_type].append(pass_fail)

    # Calculate averages (average of 100s and 0s = pass percentage)
    constraint_avg_stats = {}
    for constraint_key, model_data in constraint_stats.items():
        constraint_avg_stats[constraint_key] = {
            model_type: np.mean(pass_rates)
            for model_type, pass_rates in model_data.items()
        }

    return constraint_avg_stats


def plot_constraint_heatmap(constraint_stats, output_file='question_constraint_heatmap.png'):
    """Create detailed heatmap showing pass rates per constraint per model."""
    df = pd.DataFrame(constraint_stats).T

    # Sort columns: Origin, 5, 20, 40
    model_order = ['Origin', '5', '20', '40']
    cols = [col for col in model_order if col in df.columns]
    df = df[cols]

    # Extract question number and constraint name for sorting
    df['question_num'] = df.index.str.extract(r'^Q(\d+)')[0].astype(int)
    df['constraint_name'] = df.index.str.split('-', n=1).str[1]

    # Sort by constraint name first, then by question number
    # This groups same constraints together
    df = df.sort_values(['constraint_name', 'question_num'])
    df = df.drop(['question_num', 'constraint_name'], axis=1)

    # Create larger figure for more rows
    fig_height = max(12, len(df) * 0.3)
    plt.figure(figsize=(10, fig_height))

    sns.heatmap(df, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Pass Rate (%)'})

    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Question - Constraint', fontsize=12)
    plt.title('Per-Constraint Pass Rate Across Models (Grouped by Constraint Type)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_question_heatmap(question_stats, output_file='question_heatmap.png'):
    """Create heatmap showing pass rates per question per model."""
    question_df = pd.DataFrame(question_stats).T
    question_df = question_df.round(1)

    # Sort columns: Origin, 5, 20, 40
    model_order = ['Origin', '5', '20', '40']
    cols = [col for col in model_order if col in question_df.columns]
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
        print("\nNo result files found!")
        exit(1)

    # Calculate per-question statistics
    print("\nCalculating per-question statistics...")
    question_stats = calculate_per_question_stats(all_results)
    # Plot heatmap
    plot_question_heatmap(question_stats, script_dir / 'question_heatmap.png')
    # Print summary
    print_question_summary(question_stats)

    # NEW: Constraint-level analysis
    print("\n" + "="*80)
    print("CONSTRAINT-LEVEL ANALYSIS")
    print("="*80)

    # Collect constraint-level results
    print("\nCollecting constraint-level results...")
    constraint_results = collect_constraint_results(script_dir)

    print("Constraint results collected:")
    for model_type, runs in sorted(constraint_results.items(), key=lambda x: (x[0] != 'Origin', x[0])):
        print(f"  {model_type}: {len(runs)} runs")

    # Calculate constraint statistics
    print("\nCalculating constraint statistics...")
    constraint_stats = calculate_constraint_stats(constraint_results)

    # Plot constraint heatmap
    plot_constraint_heatmap(constraint_stats, script_dir / 'question_constraint_heatmap.png')

    print("\n✓ Per-Question Analysis complete!")
    print("✓ Constraint-Level Analysis complete!")
