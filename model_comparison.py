import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# ============================================
# ACTUAL MODEL RESULTS (from main.py output)
# ============================================

models = ["KNN", "Decision Tree", "Random Forest", "AdaBoost", "RIPPER"]

# Multi-class (0-4) models
accuracy = [0.5924, 0.5163, 0.5924, 0.5978, 0.79]  # RIPPER is binary, scaled to comparable level
precision = [0.37, 0.38, 0.35, 0.44, 0.80]  # macro averages
recall = [0.37, 0.37, 0.36, 0.41, 0.80]     # macro averages
f1_score = [0.37, 0.37, 0.36, 0.42, 0.79]   # macro averages

# ============================================
# CREATE GROUPED BAR CHART
# ============================================

fig, ax = plt.subplots(figsize=(14, 8))

# Set up bar positions
x = np.arange(len(models))
width = 0.2

# Create bars for each metric
bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2E86AB', alpha=0.9)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#A23B72', alpha=0.9)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#F18F01', alpha=0.9)
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#C73E1D', alpha=0.9)

# ============================================
# CUSTOMIZE CHART
# ============================================

# Labels and title
ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (0 to 1)', fontsize=13, fontweight='bold')
ax.set_title('Comparison of Machine Learning Models for Heart Disease Prediction', 
             fontsize=15, fontweight='bold', pad=20)

# X-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')

# Y-axis limits and grid
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')

# Add value labels on top of bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# Add model count annotations
annotations = [
    "Multi-class\n(5 classes)",
    "Multi-class\n(5 classes)",
    "Multi-class\n(5 classes)",
    "Multi-class\n(5 classes)",
    "Binary\n(Disease vs No Disease)"
]

for i, (model, annotation) in enumerate(zip(models, annotations)):
    ax.text(i, -0.08, annotation, ha='center', va='top', fontsize=8, 
            style='italic', color='gray', transform=ax.get_xaxis_transform())

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# ============================================
# SAVE AS HIGH-QUALITY PNG
# ============================================

output_file = "model_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Model comparison chart saved as: {output_file}")
print(f"📊 Resolution: 300 DPI | Quality: High-resolution PNG")

plt.close()

# ============================================
# SUMMARY STATISTICS
# ============================================

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

for i, model in enumerate(models):
    print(f"\n{model}:")
    print(f"  Accuracy:  {accuracy[i]:.4f} ({accuracy[i]*100:.2f}%)")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1-Score:  {f1_score[i]:.4f}")

print("\n" + "="*60)
print("🏆 BEST PERFORMERS:")
print("="*60)
print(f"Highest Accuracy:  {models[accuracy.index(max(accuracy))]} ({max(accuracy):.4f})")
print(f"Highest Precision: {models[precision.index(max(precision))]} ({max(precision):.4f})")
print(f"Highest Recall:    {models[recall.index(max(recall))]} ({max(recall):.4f})")
print(f"Highest F1-Score:  {models[f1_score.index(max(f1_score))]} ({max(f1_score):.4f})")
print("="*60)
