import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')

np.random.seed(42)
mean_height = 170
std_dev = 8
num_samples = 200
heights = np.random.normal(mean_height, std_dev, num_samples)
heights = np.round(heights, 1)

bins = np.arange(145, 196, 5)

plt.figure(figsize=(16, 9), dpi=240)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(16, 9), dpi=240)
fig.patch.set_facecolor('#121212')

ax = plt.gca()
histogram_color = '#FF9912'
grid_color = '#555555'

counts, edges, patches = plt.hist(heights, bins=bins, density=False, color=histogram_color,
                                  alpha=0.8, edgecolor='white', linewidth=0.5)

normalized_counts = counts / len(heights)
plt.cla()
bars = plt.bar(edges[:-1], normalized_counts, width=np.diff(edges),
               color=histogram_color, alpha=0.8, edgecolor='white', linewidth=0.5,
               align='edge')

plt.title('Height Distribution: Normalized Histogram', fontsize=24, color='white', pad=20)
plt.xlabel('Height (cm)', fontsize=18, color='white')
plt.ylabel('Probability', fontsize=18, color='white')

plt.grid(True, alpha=0.3, color=grid_color)
plt.tick_params(colors='white', labelsize=14)

plt.axvline(heights.mean(), color='#FF6347', linestyle='--', linewidth=2,
            label=f'Mean: {heights.mean():.2f} cm')
plt.axvline(np.median(heights), color='#98FB98', linestyle='-.', linewidth=2,
            label=f'Median: {np.median(heights):.2f} cm')

# Add text with statistics
stats_text = (f'Sample size: {len(heights)}\n'
              f'Mean: {heights.mean():.2f} cm\n'
              f'Median: {np.median(heights):.2f} cm\n'
              f'Std Dev: {heights.std():.2f} cm\n'
              f'Min: {heights.min():.2f} cm\n'
              f'Max: {heights.max():.2f} cm')

props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
         verticalalignment='top', horizontalalignment='right', bbox=props, color='white')

plt.legend(fontsize=14)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
for prob, x in zip(normalized_counts, bin_centers):
    if prob > 0.001:
        plt.text(x, prob + 0.005, f'{prob:.3f}', ha='center', va='bottom',
                 color='white', fontsize=12)

plt.tight_layout()
normalized_filename = "height_normalized_histogram_no_outliers.png"
plt.savefig(normalized_filename, dpi=300, bbox_inches='tight', facecolor='#121212')
plt.close()
print(f"Normalized histogram saved as {normalized_filename}")
