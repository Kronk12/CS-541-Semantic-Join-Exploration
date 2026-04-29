import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# 1. Load Data
# ==========================================
# A. Block Data (IMDB, Block_10_Full)
block_df = pd.read_csv('src/results/baseline_block.csv')
imdb_block = block_df[(block_df['Dataset'] == 'IMDB') & (block_df['Baseline_Type'] == 'Block_10_Full')]
block_f1 = imdb_block['F1'].mean()
block_tokens = imdb_block['Tokens'].mean()

# B. Cluster Data (IMDB, Ratio 0.025, Thresh 0.05 without projection)
cluster_df = pd.read_csv('src/results/imdb_aggregated_results.csv')
imdb_cluster = cluster_df[(cluster_df['Ratio'] == 0.025) & ((cluster_df['Threshold'] - 0.05).abs() < 0.001)].iloc[0]
cluster_f1 = imdb_cluster['F1 (%)']
cluster_tokens = imdb_cluster['Total Tokens']

# C. Classifier Data (IMDB)
class_df = pd.read_csv('src/results/cluster_join_imdb_classification.csv')
imdb_class = class_df[class_df['Baseline_Type'] == 'Classifier_Join'].iloc[0]
class_f1 = imdb_class['F1']
class_tokens = imdb_class['Tokens']

# Combine
plot_data = [
    {'Method': 'Block Join\n(Size=10)', 'F1 Score (%)': block_f1, 'Total Tokens': block_tokens},
    {'Method': 'Optimal Semantic\nJoin', 'F1 Score (%)': cluster_f1, 'Total Tokens': cluster_tokens},
    {'Method': 'Classifier Join', 'F1 Score (%)': class_f1, 'Total Tokens': class_tokens}
]
plot_df = pd.DataFrame(plot_data)

# ==========================================
# 2. Setup Plot (1x2 Subplots)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

palette = sns.color_palette("Set2", 3)

# ---------------------------
# Panel A: F1 Score
# ---------------------------
sns.barplot(
    data=plot_df, 
    x='Method', 
    y='F1 Score (%)', 
    ax=axes[0], 
    palette=palette, 
    edgecolor='black',
    hue='Method',
    dodge=False
)
axes[0].set_ylim(0, 100)
axes[0].set_title('Accuracy (F1 Score)', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')

# Annotate bars
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f"{height:.1f}%", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontweight='bold', fontsize=11)

if axes[0].legend_ is not None:
    axes[0].legend_.remove()

# ---------------------------
# Panel B: Token Expenditure
# ---------------------------
sns.barplot(
    data=plot_df, 
    x='Method', 
    y='Total Tokens', 
    ax=axes[1], 
    palette=palette, 
    edgecolor='black',
    hue='Method',
    dodge=False
)
axes[1].set_title('Token Expenditure', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Total Tokens', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Add headroom for labels
axes[1].set_ylim(0, plot_df['Total Tokens'].max() * 1.15)

# Annotate bars
for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f"{int(height):,}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontweight='bold', fontsize=11)

if axes[1].legend_ is not None:
    axes[1].legend_.remove()

# ==========================================
# 3. Formatting and Export
# ==========================================
plt.suptitle('IMDB Dataset: Join Strategy Comparison', fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()

filepath = 'src/figures/imdb_classifier_comparison_subplots.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {filepath}")