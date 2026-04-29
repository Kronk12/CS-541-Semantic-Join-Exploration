import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D

# Ensure output directory exists
os.makedirs('src/figures', exist_ok=True)
sns.set_theme(style="whitegrid")

# Target combinations
ratios = [0.025, 0.05, 0.075, 0.1]
projection_states = [False, True]
xticks = np.arange(0, 0.251, 0.025)

for is_projected in projection_states:
    for ratio in ratios:
        
        # 1. Select the correct datasets based on projection state
        if is_projected:
            datasets = {
                'Emails': 'src/results/emails_projection_aggregated_results.csv',
                'StackOverflow': 'src/results/stackoverflow_no_desc_projection_aggregated_results.csv',
                'IMDB': 'src/results/imdb_projection_aggregated_results.csv'
            }
            proj_label = "With Domain Projection"
            file_suffix = "projected"
        else:
            datasets = {
                'Emails': 'src/results/emails_aggregated_results.csv',
                'StackOverflow': 'src/results/stackoverflow_no_desc_aggregated_results.csv',
                'IMDB': 'src/results/imdb_aggregated_results.csv'
            }
            proj_label = "Standard Pairwise"
            file_suffix = "standard"

        # 2. Load and filter data
        dfs = []
        for name, filename in datasets.items():
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Filter for specific ratio and threshold limit
                df = df[(df['Ratio'] == ratio) & (df['Threshold'] <= 0.25)].copy()
                if not df.empty:
                    df['Dataset'] = name
                    dfs.append(df)

        if not dfs:
            print(f"Skipping {file_suffix} Ratio {ratio} due to missing data.")
            continue
            
        combined_df = pd.concat(dfs, ignore_index=True)

        # 3. Setup Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        dataset_names = ['Emails', 'StackOverflow', 'IMDB']

        color_f1 = 'tab:blue'
        color_cost = 'tab:red'

        for ax, ds_name in zip(axes, dataset_names):
            ds_data = combined_df[combined_df['Dataset'] == ds_name]
            
            if ds_data.empty:
                continue

            # Plot F1
            sns.lineplot(
                data=ds_data, 
                x='Threshold', 
                y='F1 (%)', 
                ax=ax, 
                color=color_f1, 
                linewidth=3, 
                marker='o',
                markersize=5,
                legend=False
            )
            
            ax.set_title(f'{ds_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Filter Threshold', fontsize=12)
            ax.set_ylabel('F1 Score (%)', color=color_f1, fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=color_f1)
            
            f1_min = max(0, ds_data['F1 (%)'].min() - 5)
            f1_max = min(100, ds_data['F1 (%)'].max() + 5)
            ax.set_ylim(f1_min, f1_max)
            
            ax.set_xticks(xticks)
            ax.set_xlim(-0.01, 0.26)
            ax.set_xticklabels([f"{x:.3f}" for x in xticks], rotation=45)
            
            # Plot Cost
            ax2 = ax.twinx()
            sns.lineplot(
                data=ds_data, 
                x='Threshold', 
                y='Cost ($)', 
                ax=ax2, 
                color=color_cost, 
                linewidth=3, 
                linestyle='--', 
                marker='s',
                markersize=5,
                legend=False
            )
            
            ax2.set_ylabel('Cost ($)', color=color_cost, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color_cost)
            
            cost_min = ds_data['Cost ($)'].min() * 0.95
            cost_max = ds_data['Cost ($)'].max() * 1.05
            ax2.set_ylim(cost_min, cost_max)
            ax2.grid(False) 

        # 4. Custom Legend and Formatting
        custom_lines = [
            Line2D([0], [0], color=color_f1, lw=3, marker='o', markersize=8, label='F1 Score (%)'),
            Line2D([0], [0], color=color_cost, lw=3, linestyle='--', marker='s', markersize=8, label='Cost ($)')
        ]

        fig.legend(
            handles=custom_lines, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.10), 
            ncol=2, 
            fontsize=12, 
            frameon=True
        )

        plt.suptitle(
            f'F1 Score vs Cost Trade-off | {proj_label} (Ratio = {ratio})', 
            fontsize=16, 
            fontweight='bold', 
            y=1.20
        )

        plt.tight_layout()
        
        # Save dynamically named file, stringify ratio to handle decimals safely
        safe_ratio_str = str(ratio).replace(".", "")
        filepath = f'src/figures/f1_vs_cost_ratio_{safe_ratio_str}_{file_suffix}.png'
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig) # Prevent overlap and memory issues during loops
        print(f"Generated: {filepath}")