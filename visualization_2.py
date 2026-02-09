import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set plotting style and font
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# Read data
df_raw = pd.read_csv('datasets/2026_MCM_Problem_C_Data.csv')
df_train = pd.read_csv('datasets/DWTS_XGBoost_Training_Data.csv')

# Data processing (left plot)
raw_industry_counts = df_raw['celebrity_industry'].value_counts()
top_n = 7

if len(raw_industry_counts) > top_n:
    main_industries = raw_industry_counts[:top_n]
    others_count = raw_industry_counts[top_n:].sum()
    raw_plot_data = pd.concat([main_industries, pd.Series({'Others': others_count})])

else:
    raw_plot_data = raw_industry_counts

print(f"Raw data for plotting:\n{raw_plot_data}")

# Data processing (right plot)
df_train_unique = df_train.drop_duplicates(subset=['celebrity_name'])

# Define one-hot encoding columns
cluster_cols = ['Ind_Athletes', 'Ind_Media_Fashion', 'Ind_Performing_Arts', 'Ind_Professional', 'Ind_Vocal_Artists']

# Rename column labels
cluster_labels_map = {
    'Ind_Athletes': 'Athletes',
    'Ind_Media_Fashion': 'Media & Fashion',
    'Ind_Performing_Arts': 'Performing Arts',
    'Ind_Professional': 'Professional',
    'Ind_Vocal_Artists': 'Vocal Artists'
}

cluster_counts = df_train_unique[cluster_cols].sum().rename(index=cluster_labels_map)
cluster_counts = cluster_counts.sort_values(ascending=False)

print(f"\nTrain data for plotting (descending order):\n{cluster_counts}")


# Plot comparison pie charts
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Define color schemes
colors_left = sns.color_palette("Set3", len(raw_plot_data))
colors_right = sns.color_palette("Set2", len(cluster_counts))

# --- Original industries (left plot) ---
wedges1, texts1, autotexts1 = axes[0].pie(
    raw_plot_data,
    labels=raw_plot_data.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_left,
    pctdistance=0.85,
    labeldistance=1.1,
    textprops={'fontsize': 10, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    rotatelabels=False
)

axes[0].set_title(f"Original Industry Distribution\n(Top {top_n} + Others)", fontsize=16, fontweight='bold')
axes[0].legend(wedges1, raw_plot_data.index, loc="lower right", bbox_to_anchor=(1.05, 0.05), fontsize=9, framealpha=0.9)

# --- Clustered industries (right plot) ---
wedges2, texts2, autotexts2 = axes[1].pie(
    cluster_counts,
    labels=cluster_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_right,
    pctdistance=0.85,
    labeldistance=1.1,
    explode=[0.05] * len(cluster_counts),
    textprops={'fontsize': 12, 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    rotatelabels=False
)

axes[1].set_title("Clustered Industry Distribution\n(5 Major Categories)", fontsize=16, fontweight='bold')
axes[1].legend(wedges2, cluster_counts.index, loc="lower right", bbox_to_anchor=(0.95, 0.05), fontsize=9, framealpha=0.9)

# Optional choice for visualization
centre_circle1 = plt.Circle((0, 0), 0.70, fc='white')
centre_circle2 = plt.Circle((0, 0), 0.70, fc='white')
axes[0].add_artist(centre_circle1)
axes[1].add_artist(centre_circle2)

plt.tight_layout()


# Save image
output_filename = 'visualization_outputs/Industry_Comparison_Visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nImage saved as: {output_filename}")
print("Processing complete!")
