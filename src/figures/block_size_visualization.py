import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Load and process data
# ==========================================
df = pd.read_csv('src/results/baseline_block.csv')

# Extract integer block size from Baseline_Type (e.g., "Block_5_Full" -> 5)
df['Block_Size'] = df['Baseline_Type'].str.extract(r'Block_(\d+)_Full').astype(int)

# Aggregate over trials to get average F1 and Tokens
agg_df = df.groupby(['Block_Size', 'Dataset'])[['F1', 'Tokens']].mean().reset_index()

# Sort for consistent ordering
agg_df = agg_df.sort_values(['Block_Size', 'Dataset'])
agg_df['Block_Size_Str'] = agg_df['Block_Size'].astype(str)

# ==========================================
# 2. Setup Plot
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 7))

# Determine unique datasets and block sizes dynamically
datasets_order = sorted(agg_df['Dataset'].unique())
block_sizes = sorted(agg_df['Block_Size'].unique())
block_sizes_str = [str(x) for x in block_sizes]

# Define distinct palettes based on the number of block sizes
bar_palette = sns.color_palette("pastel", len(block_sizes_str))
line_palette = sns.color_palette("dark", len(block_sizes_str))

# Plot Bars (F1 Score) on primary y-axis
sns.barplot(
    data=agg_df, 
    x='Dataset',               # <-- Swapped to Dataset
    y='F1', 
    hue='Block_Size_Str',      # <-- Swapped to Block Size
    hue_order=block_sizes_str,
    order=datasets_order,
    ax=ax1, 
    palette=bar_palette,
    edgecolor='black'
)

# Configure primary y-axis (F1 Score)
ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Average F1 Score (%)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create twin axis for Tokens
ax2 = ax1.twinx()

# Plot Lines (Tokens) on secondary y-axis using pointplot for dodge alignment
sns.pointplot(
    data=agg_df,
    x='Dataset',
    y='Tokens',
    hue='Block_Size_Str',
    hue_order=block_sizes_str,
    order=datasets_order,
    ax=ax2,
    palette=line_palette,
    dodge=0.8, # Match the default barplot dodge width
    markers='o',
    linestyles='-',
    scale=1.2
)

# Configure secondary y-axis (Tokens)
ax2.set_ylabel('Average Token Expenditure', fontsize=12, fontweight='bold', labelpad=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax2.grid(False) # Turn off grid for secondary axis to avoid visual clutter

# ==========================================
# 3. Unified Legend & Formatting
# ==========================================
# Clean up default legends
if ax1.legend_ is not None:
    ax1.legend_.remove()
if ax2.legend_ is not None:
    ax2.legend_.remove()

# Create a unified custom legend at the top
custom_handles = []
for i, bs in enumerate(block_sizes_str):
    bar_patch = mpatches.Patch(facecolor=bar_palette[i], edgecolor='black')
    line_marker = mlines.Line2D([], [], color=line_palette[i], marker='o', markersize=8, linewidth=2)
    custom_handles.append((bar_patch, line_marker))

fig.legend(
    handles=custom_handles, 
    labels=block_sizes_str, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.95), 
    ncol=len(block_sizes_str), 
    title='Block Size (Bars = F1 Score, Lines = Token Cost)', 
    title_fontsize='11', 
    fontsize='10',
    frameon=True
)

# Title and Layout Adjustments
plt.title('Determining Optimal Block Size: Accuracy vs. Token Expenditure', fontsize=16, fontweight='bold', pad=40)
plt.tight_layout()
plt.subplots_adjust(top=0.85) # Push plot down slightly to make room for legend

filepath = 'src/figures/block_size_optimization_grouped_by_dataset.png'
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {filepath}")