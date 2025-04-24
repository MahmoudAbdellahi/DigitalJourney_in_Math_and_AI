import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os

plt.style.use('dark_background')

mean_height = 170
std_dev = 8
num_samples = 195
heights = np.random.normal(mean_height, std_dev, num_samples)
heights = np.round(heights, 1)

high_outliers = np.array([205.5, 208.7, 212.3])
low_outliers = np.array([145.2, 143.8])
heights = np.append(heights, high_outliers)
heights = np.append(heights, low_outliers)

bins = np.arange(145, 196, 5)
bin_ranges = [f"{bins[i]}-{bins[i+1]-0.1}" for i in range(len(bins)-1)]
hist, bin_edges = np.histogram(heights, bins=bins)
bin_numbers = range(1, len(bin_ranges) + 1)

percentages = (hist / len(heights) * 100).round(1)

freq_table = pd.DataFrame({
    'Bin Number': bin_numbers,
    'Bin Range (cm)': bin_ranges,
    'Count': hist,
    'Percentage': [f"{p}%" for p in percentages]
})


def create_histogram_and_boxplot(heights, bins, filename_prefix="height_histogram"):
    plt.style.use('dark_background')

    histogram_color = '#4F94CD'
    boxplot_color = '#FF9912'
    outlier_color = '#FF4500'
    grid_color = '#555555'

    fig1, ax1 = plt.subplots(figsize=(16, 9), dpi=240)
    fig1.patch.set_facecolor('#121212')

    extended_bins = np.append(bins, np.arange(bins[-1], 215, 5))

    counts, edges, bars = ax1.hist(heights, bins=extended_bins, color=histogram_color, alpha=0.8,
                                   edgecolor='white', linewidth=0.5)

    ax1.set_title('Height Distribution: Histogram with Outliers', fontsize=24, color='white', pad=20)
    ax1.set_xlabel('Height (cm)', fontsize=18, color='white')
    ax1.set_ylabel('Frequency (Count)', fontsize=18, color='white')
    ax1.tick_params(colors='white', labelsize=14)
    ax1.grid(True, alpha=0.3, color=grid_color)

    ax1.axvline(heights.mean(), color='#FF6347', linestyle='--', linewidth=2,
                label=f'Mean: {heights.mean():.2f} cm')
    ax1.axvline(np.median(heights), color='#98FB98', linestyle='-.', linewidth=2,
                label=f'Median: {np.median(heights):.2f} cm')

    stats_text = (f'Sample size: {len(heights)}\n'
                  f'Mean: {heights.mean():.2f} cm\n'
                  f'Median: {np.median(heights):.2f} cm\n'
                  f'Std Dev: {heights.std():.2f} cm\n'
                  f'Min: {heights.min():.2f} cm\n'
                  f'Max: {heights.max():.2f} cm')

    props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', horizontalalignment='right', bbox=props, color='white')

    ax1.legend(fontsize=14)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    for count, x in zip(counts, bin_centers):
        if count > 0:
            ax1.text(x, count + 0.5, str(int(count)), ha='center', va='bottom',
                     color='white', fontsize=12)

    plt.tight_layout()
    counts_filename = f"{filename_prefix}_counts.png"
    plt.savefig(counts_filename, dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close(fig1)
    print(f"Histogram with outliers saved as {counts_filename}")

    fig2, ax2 = plt.subplots(figsize=(16, 9), dpi=240)
    fig2.patch.set_facecolor('#121212')

    boxplot = ax2.boxplot(heights,
                          vert=True,
                          patch_artist=True,
                          widths=0.2,
                          showmeans=False,
                          flierprops={'marker': 'o', 'markerfacecolor': outlier_color,
                                      'markeredgecolor': 'white', 'markersize': 10,
                                      'markeredgewidth': 1.5})

    for box in boxplot['boxes']:
        box.set(facecolor=boxplot_color, alpha=0.7, edgecolor='white', linewidth=2)

    for whisker in boxplot['whiskers']:
        whisker.set(color='white', linewidth=2, linestyle='--')

    for cap in boxplot['caps']:
        cap.set(color='white', linewidth=2)

    for median in boxplot['medians']:
        median.set(color='#98FB98', linewidth=3)

    ax2.set_title('Height Distribution: Box Plot with Outliers', fontsize=24, color='white', pad=20)
    ax2.set_ylabel('Height (cm)', fontsize=20, color='white', labelpad=15)
    ax2.set_xlabel('', fontsize=18)
    ax2.tick_params(axis='y', colors='white', labelsize=16)
    ax2.tick_params(axis='x', colors='white', labelsize=0)
    ax2.grid(True, axis='y', alpha=0.3, color=grid_color)

    q1 = np.percentile(heights, 25)
    q2 = np.median(heights)
    q3 = np.percentile(heights, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    mean = np.mean(heights)

    detailed_stats = (
        f"Detailed Statistics:\n"
        f"IQR (Q3-Q1): {iqr:.2f}\n"
        f"Lower Fence (Q1-1.5×IQR): {lower_fence:.2f}\n"
        f"Upper Fence (Q3+1.5×IQR): {upper_fence:.2f}"
    )

    ax2.text(0.02, 0.95, detailed_stats, transform=ax2.transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8, edgecolor='white', linewidth=1),
             color='white')

    max_y = np.max(heights) + 5
    min_y = np.min(heights) - 5
    ax2.set_ylim(min_y, max_y)

    ax2.set_xlim(0.5, 1.5)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_color('white')
    ax2.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    boxplot_filename = f"{filename_prefix}_boxplot.png"
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight', facecolor='#121212')
    plt.close(fig2)
    print(f"Box plot saved as {boxplot_filename}")

    return counts_filename, boxplot_filename


hist_image, boxplot_image = create_histogram_and_boxplot(heights, bins, "height_histogram_dark_mode")

print("\nVisualization files have been created:")
print(f"1. Histogram with outliers: {hist_image}")
print(f"2. Box plot (whisker plot): {boxplot_image}")
