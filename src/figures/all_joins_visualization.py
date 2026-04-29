import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# 1. Load Data
# ==========================================
naive_df = pd.read_csv('src/results/baseline_naive.csv')
naive_f1 = naive_df.groupby('Dataset')['F1'].mean().to_dict()

block_df = pd.read_csv('src/results/baseline_block.csv')
block_10_df = block_df[block_df['Baseline_Type'] == 'Block_10_Full']
block_f1 = block_10_df.groupby('Dataset')['F1'].mean().to_dict()

cluster_configs = [
    {'Dataset': 'Emails', 'File': 'src/results/emails_aggregated_results.csv', 'Ratio': 0.025, 'Thresh': 0.05},
    {'Dataset': 'StackOverflow', 'File': 'src/results/stackoverflow_no_desc_aggregated_results.csv', 'Ratio': 0.075, 'Thresh': 0.01},
    {'Dataset': 'IMDB', 'File': 'src/results/imdb_aggregated_results.csv', 'Ratio': 0.025, 'Thresh': 0.05},
]

cluster_f1 = {}
for c in cluster_configs:
    if os.path.exists(c['File']):
        df = pd.read_csv(c['File'])
        match = df[(df['Ratio'] == c['Ratio']) & ((df['Threshold'] - c['Thresh']).abs() < 0.001)]
        if not match.empty:
            cluster_f1[c['Dataset']] = match.iloc[0]['F1 (%)']

# ==========================================
# 2. Format Plot Data
# ==========================================
plot_data = []
datasets = ['Emails', 'StackOverflow', 'IMDB']
for ds in datasets:
    plot_data.append({'Dataset': ds, 'Method': 'Naive LLM Join', 'F1 Score (%)': naive_f1.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Block Join (Size=10)', 'F1 Score (%)': block_f1.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Optimal Cluster Join', 'F1 Score (%)': cluster_f1.get(ds, 0)})

plot_df = pd.DataFrame(plot_data)
methods_order = ['Naive LLM Join', 'Block Join (Size=10)', 'Optimal Cluster Join']

# ==========================================
# 3. Render Visualization
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=plot_df, 
    x='Dataset',           # <-- Swapped to Dataset
    y='F1 Score (%)', 
    hue='Method',          # <-- Swapped to Method
    hue_order=methods_order,
    ax=ax, 
    edgecolor='black',
    palette='Set2'
)

ax.set_ylim(0, 100)
ax.set_title('Join Strategy Comparison: F1 Score by Dataset', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', labelpad=10)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.1f}%", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontweight='bold', fontsize=10)

plt.legend(title='Join Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()

filepath = 'src/figures/method_comparison_f1_grouped.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')