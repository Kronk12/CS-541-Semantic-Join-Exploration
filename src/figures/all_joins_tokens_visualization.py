import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# 1. Load Naive Data
# ==========================================
naive_df = pd.read_csv('src/results/baseline_naive.csv')
# Use the Projected_Full_Tokens to account for the extrapolation
naive_tokens = naive_df.groupby('Dataset')['Projected_Full_Tokens'].mean().to_dict()

# ==========================================
# 2. Load Block Data (Strictly Block_10_Full)
# ==========================================
block_df = pd.read_csv('src/results/baseline_block.csv')
block_10_df = block_df[block_df['Baseline_Type'] == 'Block_10_Full']
block_tokens = block_10_df.groupby('Dataset')['Tokens'].mean().to_dict()

# ==========================================
# 3. Load Cluster Data (Optimal without projection)
# ==========================================
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
# 4. Format Plot Data
# ==========================================
plot_data = []
datasets = ['Emails', 'StackOverflow', 'IMDB']
for ds in datasets:
    plot_data.append({'Dataset': ds, 'Method': 'Naive LLM Join', 'Tokens': naive_tokens.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Block Join (Size=10)', 'Tokens': block_tokens.get(ds, 0)})
    plot_data.append({'Dataset': ds, 'Method': 'Optimal Cluster Join', 'Tokens': cluster_tokens.get(ds, 0)})

plot_df = pd.DataFrame(plot_data)

# ==========================================
# 5. Render Visualization
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=plot_df, 
    x='Method', 
    y='Tokens', 
    hue='Dataset', 
    hue_order=datasets,
    ax=ax, 
    edgecolor='black',
    palette='Set2'
)

# Convert the Y-axis to a logarithmic scale
ax.set_yscale('log')

# Formatting
ax.set_title('Join Strategy Comparison: Token Expenditure (Log Scale)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Join Strategy', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Total Tokens (Log Scale)', fontsize=12, fontweight='bold', labelpad=10)

# Format Y axis with commas for readability (e.g. 100,000)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Add the exact token counts above each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{int(height):,}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontweight='bold', fontsize=10)

plt.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()

filepath = 'src/figures/method_comparison_tokens_log.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {filepath}")