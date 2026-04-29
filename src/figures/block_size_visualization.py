import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# 1. Load and process data
df = pd.read_csv('src/results/baseline_block.csv')

# Extract integer block size from Baseline_Type (e.g., "Block_5_Full" -> 5)
df['Block_Size'] = df['Baseline_Type'].str.extract(r'Block_(\d+)_Full').astype(int)

# Aggregate over trials to get average F1 and Tokens
agg_df = df.groupby(['Block_Size', 'Dataset'])[['F1', 'Tokens']].mean().reset_index()

# Sort for consistent plotting on the X-axis
agg_df = agg_df.sort_values(['Block_Size', 'Dataset'])
agg_df['Block_Size_Str'] = agg_df['Block_Size'].astype(str)

# 2. Setup Plot
fig, ax1 = plt.subplots(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Define distinct palettes so the lines stand out against the bars
datasets = sorted(agg_df['Dataset'].unique())
bar_palette = sns.color_palette("pastel", len(datasets))
line_palette = sns.color_palette("dark", len(datasets))

# Plot Bars (F1 Score) on primary y-axis
sns.barplot(
    data=agg_df, 
    x='Block_Size_Str', 
    y='F1', 
    hue='Dataset', 
    ax=ax1, 
    palette=bar_palette,
    edgecolor='black'
)

# Configure primary y-axis (F1 Score)
ax1.set_xlabel('Block Size (Grid Partitioning)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Average F1 Score (%)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create twin axis for Tokens
ax2 = ax1.twinx()

# Plot Lines (Tokens) on secondary y-axis
sns.lineplot(
    data=agg_df,
    x='Block_Size_Str',
    y='Tokens',
    hue='Dataset',
    ax=ax2,
    palette=line_palette,
    marker='o',
    linewidth=3,
    markersize=9,
    legend=False # Prevent duplicate legend from Seaborn
)

# Configure secondary y-axis (Tokens)
ax2.set_ylabel('Average Token Expenditure', fontsize=12, fontweight='bold', labelpad=10)
# Format the right y-axis with commas for readability (e.g., 100,000)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax2.grid(False) # Turn off grid for secondary axis to avoid visual clutter

# Fix Legends (Merge them cleanly)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend_.remove() # Remove default bar legend

# Create a unified legend at the top
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
           ncol=3, title='Dataset (Bars = F1 Score, Lines = Token Cost)', 
           title_fontsize='11', fontsize='10')

# Title and Layout Adjustments
plt.title('Determining Optimal Block Size: Accuracy vs. Token Expenditure', fontsize=16, fontweight='bold', pad=40)
plt.tight_layout()
plt.subplots_adjust(top=0.85) # Push plot down slightly to make room for legend
plt.tight_layout()

filepath = 'src/figures/block_size_optimization_combo.png'
plt.savefig(filepath, dpi=300)
print(f"Successfully generated: {filepath}")