import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# 1. Load Data
# ==========================================
datasets = {
    'Emails': 'src/results/emails_aggregated_results.csv',
    'StackOverflow': 'src/results/stackoverflow_no_desc_aggregated_results.csv',
    'IMDB': 'src/results/imdb_aggregated_results.csv'
}

dfs = []
for name, filename in datasets.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df['Threshold'] == 0.0].copy()
        df['Dataset'] = name
        dfs.append(df)
    else:
        print(f"Warning: Could not find {filename}")

combined_df = pd.concat(dfs, ignore_index=True)

# Sort and format Ratio for X-axis
combined_df = combined_df.sort_values(['Ratio', 'Dataset'])
combined_df['Ratio_Str'] = combined_df['Ratio'].astype(str)

# ==========================================
# 2. Setup Plot
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 7))

datasets_list = sorted(combined_df['Dataset'].unique())
bar_palette = sns.color_palette("pastel", len(datasets_list))
line_palette = sns.color_palette("dark", len(datasets_list))

# Plot Bars (F1 Score) on primary y-axis
sns.barplot(
    data=combined_df,
    x='Ratio_Str',
    y='F1 (%)',
    hue='Dataset',
    ax=ax1,
    palette=bar_palette,
    edgecolor='black'
)

# Configure primary y-axis (F1 Score)
ax1.set_xlabel('Clustering Ratio', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create twin axis for Tokens
ax2 = ax1.twinx()

# Plot Lines (Tokens) on secondary y-axis
sns.lineplot(
    data=combined_df,
    x='Ratio_Str',
    y='Total Tokens',
    hue='Dataset',
    ax=ax2,
    palette=line_palette,
    marker='o',
    linewidth=3,
    markersize=9,
    legend=False
)

# Configure secondary y-axis (Tokens)
ax2.set_ylabel('Total Token Expenditure', fontsize=12, fontweight='bold', labelpad=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax2.grid(False)

# ==========================================
# 3. Custom Unified Legend
# ==========================================
if ax1.legend_ is not None:
    ax1.legend_.remove() 

custom_handles = []
for i, dataset in enumerate(datasets_list):
    bar_patch = mpatches.Patch(facecolor=bar_palette[i], edgecolor='black')
    line_marker = mlines.Line2D([], [], color=line_palette[i], marker='o', markersize=8, linewidth=3)
    custom_handles.append((bar_patch, line_marker))

fig.legend(
    handles=custom_handles, 
    labels=datasets_list, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.85), # Lowered from 0.95
    ncol=3, 
    title='Dataset (Bars = F1 Score, Lines = Token Cost)', 
    title_fontsize='11', 
    fontsize='10',
    frameon=True
)

# Title and Layout Adjustments
# Increased pad from 40 to 70 to push the title above the legend
plt.title('Optimal Clustering Ratio: Accuracy vs. Token Expenditure (Threshold = 0.0)', fontsize=16, fontweight='bold', pad=70)

plt.tight_layout()

# Squished the top of the plot down to 0.75 (from 0.85) to make room
plt.subplots_adjust(top=0.75) 

filepath = 'src/figures/ratio_optimization_combo.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {filepath}")