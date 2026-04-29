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
naive_df = pd.read_csv('src/results/baseline_naive.csv')
naive_tokens = naive_df.groupby('Dataset')['Projected_Full_Tokens'].mean().to_dict()

block_df = pd.read_csv('src/results/baseline_block.csv')
block_10_df = block_df[block_df['Baseline_Type'] == 'Block_10_Full']
block_tokens = block_10_df.groupby('Dataset')['Tokens'].mean().to_dict()

cluster_configs = [
    {'Dataset': 'Emails', 'File': 'src/results/emails_aggregated_results.csv', 'Ratio': 0.025, 'Thresh': 0.05},
    {'Dataset': 'StackOverflow', 'File': 'src/results/stackoverflow_no_desc_aggregated_results.csv', 'Ratio': 0.075, 'Thresh': 0.01},
    {'Dataset': 'IMDB', 'File': 'src/results/imdb_aggregated_results.csv', 'Ratio': 0.025, 'Thresh': 0.05},
]
cluster_tokens = {}
for c in cluster_configs:
    if os.path.exists(c['File']):
        df = pd.read_csv(c['File'])
        match = df[(df['Ratio'] == c['Ratio']) & ((df['Threshold'] - c['Thresh']).abs() < 0.001)]
        if not match.empty:
            cluster_tokens[c['Dataset']] = match.iloc[0]['Total Tokens']

# ==========================================
# 2. Format Plot Data
# ==========================================
plot_data = []
datasets = ['Emails', 'StackOverflow', 'IMDB']
for ds in datasets:
    plot_data.append({'Dataset': ds, 'Method': 'Naive LLM Join', 'Tokens': naive_tokens.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Block Join (Size=10)', 'Tokens': block_tokens.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Optimal Cluster Join', 'Tokens': cluster_tokens.get(ds, 0)})

plot_df = pd.DataFrame(plot_data)
methods_order = ['Naive LLM Join', 'Block Join (Size=10)', 'Optimal Cluster Join']

# ==========================================
# 3. Render Visualization
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=plot_df, 
    x='Dataset',           # <-- Swapped to Dataset
    y='Tokens', 
    hue='Method',          # <-- Swapped to Method
    hue_order=methods_order,
    ax=ax, 
    edgecolor='black',
    palette='Set2'
)

ax.set_yscale('log')
ax.set_title('Join Strategy Comparison: Token Expenditure by Dataset (Log Scale)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Total Tokens (Log Scale)', fontsize=12, fontweight='bold', labelpad=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height):,}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontweight='bold', fontsize=10)

plt.legend(title='Join Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()

filepath = 'src/figures/method_comparison_tokens_log_grouped.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')