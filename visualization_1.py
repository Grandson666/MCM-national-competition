import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors


# Step 1: Set plotting style and font
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# Step 2: Read data
file_path = 'datasets/DWTS_Viewers.xlsx'
print(f"Reading file: {file_path} ...")
df = pd.read_excel(file_path)


# Step 3: Data preprocessing
# --- 3D Chart Data Preparation ---
x_data = []  # X-axis: Season
y_data = []  # Y-axis: Week
z_data = []  # Z-axis: Viewers

week_cols = [col for col in df.columns if str(col).lower().startswith('week')]

for index, row in df.iterrows():
    season = row['season']

    for col_name in week_cols:
        viewers = row[col_name]

        if pd.notna(viewers) and viewers > 0:
            try:
                week_num = int(col_name.lower().replace('week', ''))
                x_data.append(season)
                y_data.append(week_num)
                z_data.append(viewers)
            except ValueError:
                continue

print(f"\n3D Chart: Extracted {len(z_data)} valid data points.")

# Prepare 3D plotting data
x = np.array(x_data)
y = np.array(y_data)
z = np.zeros(len(z_data))
dx = np.ones(len(x_data)) * 0.8
dy = np.ones(len(y_data)) * 0.8
dz = np.array(z_data)

# Create color map
cmap = plt.colormaps.get_cmap('Spectral_r')
norm = mcolors.Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))

# --- 2D Chart Data Preparation ---
# Premiere week
premiere_data = df['week1']

# Finale week
finale_data = df[week_cols].apply(lambda row: row.dropna().iloc[-1], axis=1)
season_labels = df['season'].astype(str)

print(f"2D Chart: Total {len(premiere_data)} seasons.")


# Step 4: Create combined chart
fig = plt.figure(figsize=(18, 8))

# 3D Bar Chart (left position)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
bars = ax1.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

# Set 3D chart axis labels
ax1.set_xlabel('Season', fontsize=11, labelpad=10)
ax1.set_ylabel('Week', fontsize=11, labelpad=10)
ax1.set_zlabel('Viewers (Millions)', fontsize=11, labelpad=10)
ax1.set_title('DWTS Viewership Distribution by Season and Week', fontsize=13, fontweight='bold', pad=10)

# Adjust axis ticks
ax1.set_xticks(np.arange(min(x), max(x)+1, 2))
ax1.set_yticks(np.arange(1, int(max(y))+1, 1))

# Add color bar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(dz)
cbar = plt.colorbar(mappable, ax=ax1, shrink=0.5, aspect=15, pad=0.08)
cbar.set_label('Viewers (Millions)', fontsize=10)

# Adjust view angle
ax1.view_init(elev=30, azim=-60)

# 2D Bar Chart (right position)
ax2 = fig.add_subplot(1, 2, 2)

# Set bar width, position and color
bar_width = 0.4
index = np.arange(len(df['season']))
color_premiere = '#5b74e6'
color_finale = '#e65540'

# Plot bars
ax2.bar(index - bar_width/2, premiere_data, bar_width, label='Premiere', color=color_premiere, alpha=0.9)
ax2.bar(index + bar_width/2, finale_data, bar_width, label='Finale', color=color_finale, alpha=0.9)

# Set 2D chart axis labels
ax2.set_xlabel('Seasons', fontsize=11, labelpad=10)
ax2.set_ylabel('Viewers (Millions)', fontsize=11)
ax2.set_title('DWTS Viewership: Premiere vs Finale by Season', fontsize=13, fontweight='bold', pad=10)

# Set X-axis ticks and Y-axis grid lines
ax2.set_xticks(index)
ax2.set_xticklabels(season_labels, fontsize=9)
ax2.grid(axis='y', linestyle='-', alpha=0.2, color='gray', zorder=0)

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add legend
ax2.legend(frameon=False, loc='upper right', fontsize=11)

# Adjust layout
plt.tight_layout()


# Step 5: Save image
output_filename = 'visualization_outputs/DWTS_Viewership_Combined_Visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nImage saved as: {output_filename}")
print("Processing complete!")
