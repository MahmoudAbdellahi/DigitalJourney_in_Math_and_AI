import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# general option
SHOW_KDE_FIT = True
SHOW_NORMAL_FIT = True


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
kde_color = '#1E90FF'
normal_color = '#FF6347'

counts, edges, patches = plt.hist(heights, bins=bins, density=True, color=histogram_color,
                                  alpha=0.8, edgecolor='white', linewidth=0.5)
x = np.linspace(140, 200, 1000)

if SHOW_KDE_FIT:
    kde = stats.gaussian_kde(heights)
    kde_y = kde(x)
    plt.plot(x, kde_y, color=kde_color, linewidth=3, label='KDE Fit')

if SHOW_NORMAL_FIT:
    normal_y = stats.norm.pdf(x, heights.mean(), heights.std())
    plt.plot(x, normal_y, color=normal_color, linewidth=3, linestyle='--',
             label=f'Normal Fit (μ={heights.mean():.2f}, σ={heights.std():.2f})')

plt.title('Height Distribution: Density Histogram', fontsize=24, color='white', pad=20)
plt.xlabel('Height (cm)', fontsize=18, color='white')
plt.ylabel('Density', fontsize=18, color='white')

plt.xlim(140, 200)

plt.grid(True, alpha=0.3, color=grid_color)
plt.tick_params(colors='white', labelsize=14)

plt.axvline(heights.mean(), color='#98FB98', linestyle='--', linewidth=2,
            label=f'Mean: {heights.mean():.2f} cm')
plt.axvline(np.median(heights), color='#FFFFFF', linestyle='-.', linewidth=2,
            label=f'Median: {np.median(heights):.2f} cm')

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
for height, x in zip(counts, bin_centers):
    if height > 0.001:
        plt.text(x, height + 0.002, f'{height:.4f}', ha='center', va='bottom',
                 color='white', fontsize=12)

plt.tight_layout()
output_filename = "height_density_histogram_with_fits.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='#121212')
plt.close()

print(f"Density histogram saved as {output_filename}")
