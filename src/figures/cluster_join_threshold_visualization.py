import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Load Data
# ==========================================
datasets = {
    'Emails': 'src/results/emails_aggregated_results.csv',
    'StackOverflow': 'src/results/stackoverflow_no_desc_aggregated_results.csv',
    'IMDB': 'src/results/imdb_aggregated_results.csv'
}

dfs = []
for name, path in datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Dataset'] = name
        dfs.append(df)
    else:
        print(f"Warning: Could not find {path}")

combined_df = pd.concat(dfs, ignore_index=True)

# ==========================================
# 2. Filter and Process Data
# ==========================================
# Extend the x-axis to 0.40 to capture the late drop-off for IMDB
combined_df = combined_df[combined_df['Threshold'] <= 0.40]

# Convert Ratio to string for discrete categorical hue
combined_df['Ratio'] = combined_df['Ratio'].astype(str)

# ==========================================
# 3. Setup 1x3 Subplot Grid
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
dataset_names = ['Emails', 'StackOverflow', 'IMDB']

ratios = sorted(combined_df['Ratio'].unique())
palette = sns.color_palette("viridis", len(ratios))

for i, (ax, ds_name) in enumerate(zip(axes, dataset_names)):
    ds_data = combined_df[combined_df['Dataset'] == ds_name]
    
    # Plot F1 Score on the primary Y-axis (Solid Lines)
    sns.lineplot(
        data=ds_data, 
        x='Threshold', 
        y='F1 (%)', 
        hue='Ratio', 
        hue_order=ratios,
        palette=palette,
        linewidth=2.5,
        ax=ax,
        legend=False
    )
    
    # Create a secondary Y-axis for Tokens
    ax2 = ax.twinx()
    
    # Plot Tokens on the secondary Y-axis (Dashed Lines)
    sns.lineplot(
        data=ds_data, 
        x='Threshold', 
        y='Total Tokens', 
        hue='Ratio', 
        hue_order=ratios,
        palette=palette,
        linewidth=2.5,
        linestyle='--',
        ax=ax2,
        legend=False
    )
    
    # Formatting X-axis
    ax.set_title(f'{ds_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Filter Threshold', fontsize=12)
    ax.set_xlim(0, 0.40)
    
    # Formatting Primary Y-axis (Left)
    ax.set_ylim(0, 100)
    if i == 0:
        ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

    # Formatting Secondary Y-axis (Right)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.grid(False) # Turn off right axis grid to avoid clutter
    if i == 2:
        ax2.set_ylabel('Total Token Expenditure', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    else:
        ax2.set_ylabel('')
        ax2.tick_params(labelright=False)

# ==========================================
# 4. Custom Unified Legend
# ==========================================
# Build custom legend elements to clearly explain colors and line styles
legend_elements = []

# Add color keys for the ratios
for ratio, color in zip(ratios, palette):
    legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Ratio: {ratio}'))

# Add style keys for the metrics
legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='Metric: F1 Score'))
legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='Metric: Tokens'))

fig.legend(
    handles=legend_elements, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.05), 
    ncol=len(ratios) + 2, 
    fontsize=11, 
    frameon=True
)

plt.suptitle(
    'Hyperparameter Optimization: Ratio vs. Threshold (Tokens & F1 Score)', 
    fontsize=16, 
    fontweight='bold', 
    y=1.12
)

plt.tight_layout()

filepath = 'src/figures/ratio_threshold_optimization_with_tokens.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {filepath}")