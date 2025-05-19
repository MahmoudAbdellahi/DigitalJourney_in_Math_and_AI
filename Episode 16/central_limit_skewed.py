import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import stats

plt.style.use('dark_background')

np.random.seed(42)

base_mean = 165
base_std = 8
outlier_mean = 190
outlier_std = 7
outlier_percentage = 0.1


def generate_skewed_normal(size):
    base_sample_size = int(size * (1 - outlier_percentage))
    outlier_sample_size = size - base_sample_size

    base_sample = np.random.normal(base_mean, base_std, base_sample_size)
    outlier_sample = np.random.normal(outlier_mean, outlier_std, outlier_sample_size)

    return np.concatenate([base_sample, outlier_sample])


population = generate_skewed_normal(100000)
population_mean = np.mean(population)
population_std = np.std(population)

true_mean = population_mean
true_std = population_std

fixed_sample_size = 100
max_frames = 500

clt_sample_means = []

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), dpi=120,
                                    gridspec_kw={'height_ratios': [3, 1, 2]})
fig.patch.set_facecolor('#121212')

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#121212')
    ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
    ax.tick_params(colors='white', labelsize=14)

hist_color = '#4682B4'
sample_color = '#FF9912'
pop_color = '#98FB98'
theory_color = '#FF6347'
point_color = '#DDDDDD'
mean_color = '#00FF00'

x_range = (130, 210)
x = np.linspace(x_range[0], x_range[1], 1000)


def theoretical_pdf(x):
    base_dist = (1 - outlier_percentage) * stats.norm.pdf(x, base_mean, base_std)
    outlier_dist = outlier_percentage * stats.norm.pdf(x, outlier_mean, outlier_std)
    return base_dist + outlier_dist


theoretical_values = [theoretical_pdf(xi) for xi in x]

hist_bins = np.linspace(x_range[0], x_range[1], 40)
bin_width = hist_bins[1] - hist_bins[0]

n_bins = len(hist_bins) - 1
hist_heights = np.zeros(n_bins)
histogram = ax1.bar(hist_bins[:-1], hist_heights, width=bin_width, alpha=0.6,
                    color=hist_color, align='edge')

sample_kde_line, = ax1.plot([], [], color=sample_color, lw=2.5)
true_pdf_line, = ax1.plot(x, theoretical_values, color=theory_color, linestyle='--',
                          lw=2.5)

sample_mean_line = ax1.axvline(x=0, color=sample_color, linestyle='-', lw=1.8)
population_mean_line = ax1.axvline(x=population_mean, color=pop_color, linestyle='-',
                                   lw=1.8)
theoretical_mean_line = ax1.axvline(x=true_mean, color=theory_color, linestyle='--',
                                    lw=1.8)

props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
stats_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                      bbox=props, color='white')

ax1.set_xlim(x_range)
ax1.set_ylim(0, 0.05)
ax1.set_ylabel('Density', fontsize=18, color='white')
ax1.set_title('Height Sample Distribution vs. Theoretical Distribution',
              fontsize=24, color='white', pad=20)
ax1.legend(['Sample KDE', 'Theoretical PDF', 'Sample Mean',
           'Population Mean', 'Theoretical Mean', 'Sample Histogram'],
           loc='upper right', fontsize=14)

ax2.set_xlim(x_range)
ax2.set_ylabel('Iteration', fontsize=18, color='white')
ax2.set_title('Individual Height Observations with Sample Means',
              fontsize=18, color='white')

sem = true_std / np.sqrt(fixed_sample_size)

ax3.set_xlim(x_range)
ax3.set_ylim(0, 0.5)
ax3.set_xlabel('Sample Mean Height (cm)', fontsize=18, color='white')
ax3.set_ylabel('Density', fontsize=18, color='white')
ax3.set_title(f'Distribution of Sample Mean Heights (n={fixed_sample_size})',
              fontsize=18, color='white')

x_clt = np.linspace(x_range[0], x_range[1], 1000)
clt_theoretical_pdf = stats.norm.pdf(x_clt, true_mean, sem)
clt_theory_line, = ax3.plot(x_clt, clt_theoretical_pdf, color=theory_color, linestyle='--',
                            lw=2.5)

clt_hist_bins = np.linspace(x_range[0], x_range[1], 40)
clt_bin_width = clt_hist_bins[1] - clt_hist_bins[0]
clt_hist_heights = np.zeros(len(clt_hist_bins)-1)
clt_histogram = ax3.bar(clt_hist_bins[:-1], clt_hist_heights, width=clt_bin_width,
                        alpha=0.6, color=hist_color, align='edge')

clt_kde_line, = ax3.plot([], [], color=sample_color, lw=2.5)

ax3.axvline(x=true_mean, color=theory_color, linestyle='--', lw=1.5)
ax3.legend(['Theoretical (σ/√n, n=100)', 'Means KDE', 'Population Mean', 'Mean Histogram'],
           loc='upper right', fontsize=12)

point_collections = {}
mean_markers = {}

random_seeds = np.random.randint(0, 10000, size=max_frames)


def init():
    sample_kde_line.set_data([], [])
    sample_mean_line.set_xdata([0])
    stats_text.set_text('')
    clt_theory_line.set_data(x_clt, clt_theoretical_pdf)
    clt_kde_line.set_data([], [])

    for bar in histogram:
        bar.set_height(0)

    for bar in clt_histogram:
        bar.set_height(0)

    for collection in point_collections.values():
        if collection in ax2.collections:
            collection.remove()
    point_collections.clear()

    for marker in mean_markers.values():
        if marker in ax2.collections:
            marker.remove()
    mean_markers.clear()

    return (sample_kde_line, sample_mean_line, stats_text, histogram,
            clt_theory_line, clt_kde_line, clt_histogram)


def update(frame):
    np.random.seed(random_seeds[frame])

    current_sample = generate_skewed_normal(fixed_sample_size)

    sample_mean = np.mean(current_sample)
    sample_std = np.std(current_sample)

    clt_sample_means.append(sample_mean)

    counts, _ = np.histogram(current_sample, bins=hist_bins, density=True)
    for i, (count, bar) in enumerate(zip(counts, histogram)):
        bar.set_height(count)

    try:
        kde = stats.gaussian_kde(current_sample)
        sample_kde = kde(x)
        sample_kde_line.set_data(x, sample_kde)
    except np.linalg.LinAlgError:
        sample_kde_line.set_data([], [])

    sample_mean_line.set_xdata([sample_mean])

    stats_text.set_text(f'Sample Size: {fixed_sample_size}\n'
                        f'Sample Mean: {sample_mean:.2f} cm (σ: {sample_std:.2f} cm)\n'
                        f'Population Mean: {population_mean:.2f} cm (σ: {population_std:.2f} cm)')

    point_collections[frame] = ax2.scatter(
        current_sample, np.ones(len(current_sample)) * frame,
        color=point_color, alpha=0.7, s=15
    )

    glow_size = 200
    mean_markers[frame] = ax2.scatter(
        [sample_mean], [frame],
        color=mean_color, alpha=1.0, s=glow_size, marker='D',
        edgecolor='white', linewidth=2.0, zorder=10
    )

    mean_markers[f"{frame}_glow"] = ax2.scatter(
        [sample_mean], [frame],
        color=mean_color, alpha=0.3, s=glow_size*1.5,
        marker='o', zorder=9
    )

    if len(clt_sample_means) > 10:
        clt_counts, _ = np.histogram(clt_sample_means, bins=clt_hist_bins, density=True)
        for i, (count, bar) in enumerate(zip(clt_counts, clt_histogram)):
            bar.set_height(count)

        try:
            clt_kde = stats.gaussian_kde(clt_sample_means)
            clt_kde_values = clt_kde(x_clt)
            clt_kde_line.set_data(x_clt, clt_kde_values)
        except np.linalg.LinAlgError:
            clt_kde_line.set_data([], [])

        ax3.set_title(f'Distribution of Sample Mean Heights (n={fixed_sample_size}, samples={len(clt_sample_means)})',
                      fontsize=18, color='white')

    ax2.set_ylim(-10, frame + 10)

    if frame % 50 == 0:
        current_ticks = list(ax2.get_yticks())
        if frame not in current_ticks:
            current_ticks.append(frame)
            ax2.set_yticks(current_ticks)

    return (sample_kde_line, sample_mean_line, stats_text, histogram,
            clt_theory_line, clt_kde_line, clt_histogram)


ani = FuncAnimation(fig, update, frames=max_frames, init_func=init,
                    blit=False, interval=50, repeat=False)

plt.tight_layout()
animation_filename = "human_height_clt_animation_skewed.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=15, dpi=120, bitrate=-1,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
plt.show()
