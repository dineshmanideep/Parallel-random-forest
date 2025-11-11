import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLOR_SERIAL = '#2E86AB'
COLOR_TREE = '#A23B72'
COLOR_FOREST = '#F18F01'

# ============================================================================
# DATA EXTRACTION FROM BENCHMARKS
# ============================================================================

# 1. Varying number of trees (2, 10, 100, 200) - all with progress bar
trees_data = {
    2: {
        'serial': 2455,
        'tree_parallel': 1890,
        'forest_parallel': 2662,
        'speedup_tree': 1.30,
        'speedup_forest': 0.92,
        'accuracy': 0.8956
    },
    10: {
        'serial': 13237,
        'tree_parallel': 10537,
        'forest_parallel': 3320,
        'speedup_tree': 1.26,
        'speedup_forest': 3.99,
        'accuracy': 0.9235
    },
    100: {
        'serial': 130327,
        'tree_parallel': 100561,
        'forest_parallel': 28140,
        'speedup_tree': 1.30,
        'speedup_forest': 4.63,
        'accuracy': 0.9191
    },
    200: {
        'serial': 262589,
        'tree_parallel': 200831,
        'forest_parallel': 58047,
        'speedup_tree': 1.31,
        'speedup_forest': 4.52,
        'accuracy': 0.9191
    }
}

# 2. Progress bar comparison (100 trees)
progress_bar_data = {
    'with_progress': {
        'serial': 130327,
        'tree_parallel': 100561,
        'forest_parallel': 28140,
        'speedup_tree': 1.30,
        'speedup_forest': 4.63
    },
    'without_progress': {
        'serial': 132386,
        'tree_parallel': 102267,
        'forest_parallel': 29406,
        'speedup_tree': 1.29,
        'speedup_forest': 4.50
    }
}

# 3. Accuracy: Decision Tree vs Random Forest (200 trees)
accuracy_data = {
    'Decision Tree': {
        'accuracy': 0.8985,
        'precision': 0.9204,
        'recall': 0.9168,
        'f1': 0.9186
    },
    'Random Forest (200 trees)': {
        'accuracy': 0.9191,
        'precision': 0.9367,
        'recall': 0.9302,
        'f1': 0.9334
    }
}

# ============================================================================
# FIGURE 1: SPEEDUP vs NUMBER OF TREES
# ============================================================================

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

tree_counts = list(trees_data.keys())
speedup_tree = [trees_data[n]['speedup_tree'] for n in tree_counts]
speedup_forest = [trees_data[n]['speedup_forest'] for n in tree_counts]

# Plot 1a: Speedup comparison
x = np.arange(len(tree_counts))
width = 0.35

bars1 = ax1.bar(x - width/2, speedup_tree, width, label='Tree-level Parallelism',
                color=COLOR_TREE, alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, speedup_forest, width, label='Forest-level Parallelism',
                color=COLOR_FOREST, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add ideal speedup line (theoretical maximum)
ax1.axhline(y=4.63, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Max Achieved Speedup')

ax1.set_xlabel('Number of Trees', fontweight='bold')
ax1.set_ylabel('Speedup Factor', fontweight='bold')
ax1.set_title('(a) Parallelization Speedup vs Number of Trees', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tree_counts)
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}×',
                ha='center', va='bottom', fontsize=8)

# Plot 1b: Training time comparison
serial_times = [trees_data[n]['serial']/1000 for n in tree_counts]  # Convert to seconds
tree_times = [trees_data[n]['tree_parallel']/1000 for n in tree_counts]
forest_times = [trees_data[n]['forest_parallel']/1000 for n in tree_counts]

ax2.plot(tree_counts, serial_times, 'o-', label='Serial', 
         color=COLOR_SERIAL, linewidth=2, markersize=8)
ax2.plot(tree_counts, tree_times, 's-', label='Tree-level Parallel',
         color=COLOR_TREE, linewidth=2, markersize=8)
ax2.plot(tree_counts, forest_times, '^-', label='Forest-level Parallel',
         color=COLOR_FOREST, linewidth=2, markersize=8)

ax2.set_xlabel('Number of Trees', fontweight='bold')
ax2.set_ylabel('Training Time (seconds)', fontweight='bold')
ax2.set_title('(b) Training Time vs Number of Trees', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks(tree_counts)
ax2.set_xticklabels(tree_counts)

plt.tight_layout()
plt.savefig('figure1_speedup_vs_trees.png', bbox_inches='tight')
print("✓ Saved: figure1_speedup_vs_trees.png")

# ============================================================================
# FIGURE 2: PROGRESS BAR OVERHEAD ANALYSIS
# ============================================================================

fig2, ax = plt.subplots(figsize=(10, 5))

configurations = ['Serial', 'Tree-level\nParallel', 'Forest-level\nParallel']
with_progress = [
    progress_bar_data['with_progress']['serial']/1000,
    progress_bar_data['with_progress']['tree_parallel']/1000,
    progress_bar_data['with_progress']['forest_parallel']/1000
]
without_progress = [
    progress_bar_data['without_progress']['serial']/1000,
    progress_bar_data['without_progress']['tree_parallel']/1000,
    progress_bar_data['without_progress']['forest_parallel']/1000
]

x = np.arange(len(configurations))
width = 0.35

bars1 = ax.bar(x - width/2, without_progress, width, label='Without Progress Bar',
               color='#06A77D', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, with_progress, width, label='With Progress Bar',
               color='#D62246', alpha=0.8, edgecolor='black', linewidth=0.5)

# Calculate overhead percentage
overhead_pct = [(with_progress[i] - without_progress[i]) / without_progress[i] * 100 
                for i in range(len(configurations))]

ax.set_xlabel('Parallelism Configuration', fontweight='bold')
ax.set_ylabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Progress Bar Overhead Impact (100 Trees, 3,403 Samples)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configurations)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels and overhead percentages
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{height1:.1f}s',
            ha='center', va='bottom', fontsize=8)
    ax.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{height2:.1f}s',
            ha='center', va='bottom', fontsize=8)
    # Add overhead percentage above
    ax.text(x[i], max(height1, height2) * 1.05,
            f'Δ {overhead_pct[i]:+.1f}%',
            ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_progress_bar_overhead.png', bbox_inches='tight')
print("✓ Saved: figure2_progress_bar_overhead.png")

# ============================================================================
# FIGURE 3: ACCURACY COMPARISON - DECISION TREE VS RANDOM FOREST
# ============================================================================

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 3a: All metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
dt_values = [
    accuracy_data['Decision Tree']['accuracy'],
    accuracy_data['Decision Tree']['precision'],
    accuracy_data['Decision Tree']['recall'],
    accuracy_data['Decision Tree']['f1']
]
rf_values = [
    accuracy_data['Random Forest (200 trees)']['accuracy'],
    accuracy_data['Random Forest (200 trees)']['precision'],
    accuracy_data['Random Forest (200 trees)']['recall'],
    accuracy_data['Random Forest (200 trees)']['f1']
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, dt_values, width, label='Single Decision Tree',
                color='#5C7CFA', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, rf_values, width, label='Random Forest (200 trees)',
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('(a) Performance Metrics Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0.85, 0.95)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=7, rotation=0)

# Plot 3b: Accuracy improvement with number of trees
tree_accuracy = [trees_data[n]['accuracy'] for n in tree_counts]
ax2.plot(tree_counts, tree_accuracy, 'o-', color='#FF6B6B', 
         linewidth=2.5, markersize=10, label='Random Forest')
ax2.axhline(y=accuracy_data['Decision Tree']['accuracy'], 
           color='#5C7CFA', linestyle='--', linewidth=2, 
           label='Single Decision Tree')

ax2.set_xlabel('Number of Trees in Random Forest', fontweight='bold')
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('(b) Accuracy vs Ensemble Size', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xscale('log')
ax2.set_xticks(tree_counts)
ax2.set_xticklabels(tree_counts)
ax2.set_ylim(0.89, 0.93)

# Add value labels
for i, (trees, acc) in enumerate(zip(tree_counts, tree_accuracy)):
    ax2.text(trees, acc + 0.001, f'{acc:.4f}',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure3_accuracy_comparison.png', bbox_inches='tight')
print("✓ Saved: figure3_accuracy_comparison.png")

# ============================================================================
# FIGURE 4: COMPREHENSIVE SPEEDUP ANALYSIS
# ============================================================================

fig4, ax = plt.subplots(figsize=(10, 6))

# Create a heatmap-style visualization
tree_counts_list = list(trees_data.keys())
speedups_tree = [trees_data[n]['speedup_tree'] for n in tree_counts_list]
speedups_forest = [trees_data[n]['speedup_forest'] for n in tree_counts_list]

x_pos = np.arange(len(tree_counts_list))
width = 0.35

bars1 = ax.bar(x_pos - width/2, speedups_tree, width, 
               label='Tree-level Parallelism',
               color=COLOR_TREE, alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x_pos + width/2, speedups_forest, width,
               label='Forest-level Parallelism',
               color=COLOR_FOREST, alpha=0.8, edgecolor='black', linewidth=1)

# Add theoretical speedup reference lines
ax.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax.axhline(y=14, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, 
           label='Theoretical Max (14 cores)')

ax.set_xlabel('Number of Trees', fontweight='bold', fontsize=12)
ax.set_ylabel('Speedup Factor (× faster than serial)', fontweight='bold', fontsize=12)
ax.set_title('Parallel Random Forest: Speedup Analysis\n(Dataset: Dry Beans, 3,403 samples, 16 features | Hardware: Intel i9-12900H, 14 cores)', 
             fontweight='bold', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(tree_counts_list)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 15)

# Add efficiency percentage
for i, (bars, speedups) in enumerate([(bars1, speedups_tree), (bars2, speedups_forest)]):
    for j, bar in enumerate(bars):
        height = bar.get_height()
        efficiency = (height / 14) * 100  # 14 cores
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}×\n({efficiency:.0f}% eff.)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('figure4_comprehensive_speedup.png', bbox_inches='tight')
print("✓ Saved: figure4_comprehensive_speedup.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)

print("\n1. SPEEDUP ANALYSIS (varying tree count):")
print(f"   - Best speedup achieved: {max(speedups_forest):.2f}× (forest-level, 100 trees)")
print(f"   - Speedup range (forest-level): {min(speedups_forest):.2f}× to {max(speedups_forest):.2f}×")
print(f"   - Speedup range (tree-level): {min(speedups_tree):.2f}× to {max(speedups_tree):.2f}×")

print("\n2. PROGRESS BAR OVERHEAD:")
overhead_serial = ((with_progress[0] - without_progress[0]) / without_progress[0]) * 100
overhead_forest = ((with_progress[2] - without_progress[2]) / without_progress[2]) * 100
print(f"   - Serial overhead: {overhead_serial:+.2f}%")
print(f"   - Forest-level parallel overhead: {overhead_forest:+.2f}%")
print(f"   - Average overhead: {np.mean(overhead_pct):+.2f}%")

print("\n3. ACCURACY IMPROVEMENT:")
acc_improvement = (accuracy_data['Random Forest (200 trees)']['accuracy'] - 
                   accuracy_data['Decision Tree']['accuracy']) / \
                   accuracy_data['Decision Tree']['accuracy'] * 100
print(f"   - Single Decision Tree: {accuracy_data['Decision Tree']['accuracy']:.4f}")
print(f"   - Random Forest (200 trees): {accuracy_data['Random Forest (200 trees)']['accuracy']:.4f}")
print(f"   - Relative improvement: {acc_improvement:+.2f}%")

print("\n4. EFFICIENCY ANALYSIS:")
max_speedup = max(speedups_forest)
theoretical_max = 14  # number of cores
efficiency = (max_speedup / theoretical_max) * 100
print(f"   - Peak efficiency: {efficiency:.1f}% of theoretical maximum")
print(f"   - Hardware: Intel i9-12900H (14 cores, 20 logical processors)")

print("\n" + "="*70)
print("All visualizations saved successfully!")
print("="*70)

plt.show()

