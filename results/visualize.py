"""
Simple visualization script for LLM evaluation results
Focuses on per-model statistics: Average, Min, Max pass rates
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from collections import defaultdict


def parse_eval_summary(text):
    """Extract average pass rate from evaluation text file."""
    avg_pass = re.search(r"Average Pass Rate:\s*([\d.]+)%", text)
    return float(avg_pass.group(1)) if avg_pass else 0.0


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
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                pass_rate = parse_eval_summary(text)
                results[model_type].append(pass_rate)
            else:
                print(f"⚠ Warning: {filename} not found")

    return results

def plot_models_all_run(all_results, output_file='model_pass_rates_all_runs.png'):
    """Plot pass rates for all runs of each model as bar charts."""
    # Sort models: Origin first, then others
    models = sorted(all_results.keys(), key=lambda x: (x != 'Origin', x))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {'Origin': '#1f77b4', '5': '#ff7f0e', '20': '#2ca02c', '40': '#d62728'}

    for idx, model_type in enumerate(models):
        ax = axes[idx]
        pass_rates = all_results[model_type]
        runs = list(range(1, len(pass_rates) + 1))

        bars = ax.bar(runs, pass_rates, color=colors.get(model_type, '#8c564b'), alpha=0.8)

        ax.set_xlabel('Run Number', fontsize=11)
        ax.set_ylabel('Pass Rate (%)', fontsize=11)
        ax.set_title(f'{model_type} Model', fontsize=13, fontweight='bold')
        ax.set_xticks(runs)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Pass Rates Across All Runs', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('model_pass_rates_all_runs.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_pass_rates_all_runs.png")
    plt.show()
def calculate_model_stats(all_results):
    """Calculate average, min, max scores for each model."""
    model_stats = {}

    for model_type, pass_rates in all_results.items():
        if not pass_rates:
            continue

        model_stats[model_type] = {
            'average': np.mean(pass_rates),
            'min': np.min(pass_rates),
            'max': np.max(pass_rates),
            'std': np.std(pass_rates),
            'runs': len(pass_rates)
        }

    return model_stats


def plot_model_stats(model_stats, output_file='model_stats_comparison.png'):
    """Visualize average, min, max for each model."""
    models = sorted(model_stats.keys(), key=lambda x: (x != 'Origin', x))
    averages = [model_stats[m]['average'] for m in models]
    mins = [model_stats[m]['min'] for m in models]
    maxs = [model_stats[m]['max'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, mins, width, label='Min', alpha=0.8, color='#ff7f0e')
    bars2 = ax.bar(x, averages, width, label='Average', alpha=0.8, color='#2ca02c')
    bars3 = ax.bar(x + width, maxs, width, label='Max', alpha=0.8, color='#1f77b4')

    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Model Performance: Average, Min, Max Pass Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.show()


def print_summary(model_stats):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("MODEL STATISTICS SUMMARY")
    print("="*80)
    print(f"{'Model':<10} {'Runs':<8} {'Average':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-"*80)

    for model in sorted(model_stats.keys(), key=lambda x: (x != 'Origin', x)):
        stats = model_stats[model]
        print(f"{model:<10} {stats['runs']:<8} {stats['average']:<10.2f} "
              f"{stats['min']:<10.2f} {stats['max']:<10.2f} {stats['std']:<10.2f}")

    print("="*80)


if __name__ == "__main__":
    print("Starting visualization...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Collect results
    all_results = collect_all_results(script_dir)

    print("\nCollected results:")
    for model_type, runs in sorted(all_results.items(), key=lambda x: (x[0] != 'Origin', x[0])):
        print(f"  {model_type}: {len(runs)} runs")

    if not all_results:
        print("\n No result files found!")
        exit(1)

    # Calculate statistics
    model_stats = calculate_model_stats(all_results)

    # Print summary
    print_summary(model_stats)

    # Generate visualization
    plot_model_stats(model_stats, script_dir / 'model_stats_comparison.png')
    plot_models_all_run(all_results, script_dir / 'model_pass_rates_all_runs.png')
    print("\n✓ Visualization complete!")
