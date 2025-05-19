import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import stats

plt.style.use('dark_background')

np.random.seed(42)

true_mean = 170
true_std = 10

population = np.random.normal(true_mean, true_std, 100000)
population_mean = np.mean(population)
population_std = np.std(population)

initial_sample_size = 5
max_frames = 600

sample_sizes = [initial_sample_size]

for i in range(1, max_frames):
    current_size = sample_sizes[-1]

    if current_size < 20:
        increment = 1 if i % 7 == 0 else 0
    elif current_size < 50:
        increment = 1 if i % 3 == 0 else 0
    elif current_size < 100:
        increment = 1
    elif current_size < 500:
        increment = 5
    elif current_size < 1000:
        increment = 10
    elif current_size < 5000:
        increment = 25
    elif current_size < 10000:
        increment = 50
    elif current_size < 20000:
        increment = 100
    elif current_size < 50000:
        increment = 200
    else:
        increment = 300

    next_size = current_size + increment

    if next_size >= len(population):
        next_size = len(population)

    sample_sizes.append(next_size)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=120,
                               gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#121212')

for ax in [ax1, ax2]:
    ax.set_facecolor('#121212')
    ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
    ax.tick_params(colors='white', labelsize=14)

hist_color = '#4682B4'
sample_color = '#FF9912'
pop_color = '#98FB98'
theory_color = '#FF6347'
point_color = '#DDDDDD'

x = np.linspace(true_mean - 4*true_std, true_mean + 4*true_std, 1000)
theoretical_pdf = stats.norm.pdf(x, true_mean, true_std)

hist_bins = np.linspace(true_mean - 4*true_std, true_mean + 4*true_std, 40)
bin_width = hist_bins[1] - hist_bins[0]

n_bins = len(hist_bins) - 1
hist_heights = np.zeros(n_bins)
histogram = ax1.bar(hist_bins[:-1], hist_heights, width=bin_width, alpha=0.6,
                    color=hist_color, label='Sample Histogram', align='edge')

sample_kde_line, = ax1.plot([], [], color=sample_color, lw=2.5, label='Sample KDE')
true_pdf_line, = ax1.plot(x, theoretical_pdf, color=theory_color, linestyle='--',
                          lw=2.5, label='Theoretical PDF')

sample_mean_line = ax1.axvline(x=0, color=sample_color, linestyle='-', lw=1.8,
                               label='Sample Mean')
population_mean_line = ax1.axvline(x=population_mean, color=pop_color, linestyle='-',
                                   lw=1.8, label='Population Mean')
theoretical_mean_line = ax1.axvline(x=true_mean, color=theory_color, linestyle='--',
                                    lw=1.8, label='Theoretical Mean')

props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
stats_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                      bbox=props, color='white')

ax1.set_xlim(true_mean - 4*true_std, true_mean + 4*true_std)
ax1.set_ylim(0, 0.05)
ax1.set_ylabel('Density', fontsize=18, color='white')
ax1.set_title('Convergence of Sample Distribution to Theoretical Distribution',
              fontsize=24, color='white', pad=20)
ax1.legend(loc='upper right', fontsize=14)

ax2.set_xlim(true_mean - 4*true_std, true_mean + 4*true_std)
ax2.set_xlabel('Value', fontsize=18, color='white')
ax2.set_ylabel('Iteration', fontsize=18, color='white')

point_collections = {}

random_seeds = np.random.randint(0, 10000, size=max_frames)


def init():
    sample_kde_line.set_data([], [])
    sample_mean_line.set_xdata([0])
    stats_text.set_text('')

    for bar in histogram:
        bar.set_height(0)

    for collection in point_collections.values():
        if collection in ax2.collections:
            collection.remove()
    point_collections.clear()

    return sample_kde_line, sample_mean_line, stats_text, histogram


def update(frame):
    np.random.seed(random_seeds[frame])

    current_sample_size = sample_sizes[frame]

    current_sample = np.random.normal(true_mean, true_std, current_sample_size)

    sample_mean = np.mean(current_sample)
    sample_std = np.std(current_sample)

    counts, _ = np.histogram(current_sample, bins=hist_bins, density=True)
    for i, (count, bar) in enumerate(zip(counts, histogram)):
        bar.set_height(count)

    if current_sample_size > 10:
        try:
            if current_sample_size > 1000:
                subset_size = 1000
                subset_indices = np.random.choice(current_sample_size, subset_size, replace=False)
                kde_sample = current_sample[subset_indices]
            else:
                kde_sample = current_sample

            kde = stats.gaussian_kde(kde_sample)
            sample_kde = kde(x)
            sample_kde_line.set_data(x, sample_kde)
        except np.linalg.LinAlgError:
            sample_kde_line.set_data([], [])
    else:
        sample_kde_line.set_data([], [])

    sample_mean_line.set_xdata([sample_mean])

    stats_text.set_text(f'Sample Size: {current_sample_size:,}\n'
                        f'Sample Mean: {sample_mean:.2f} (σ: {sample_std:.2f})\n'
                        f'Population Mean: {population_mean:.2f} (σ: {population_std:.2f})\n'
                        f'Theoretical Mean: {true_mean} (σ: {true_std})')

    show_this_frame = (
        current_sample_size < 100 or
        frame % 10 == 0 or
        frame % (max_frames // 20) == 0
    )

    if show_this_frame:
        display_limit = min(100, current_sample_size)
        if current_sample_size > display_limit:
            display_indices = np.random.choice(current_sample_size, display_limit, replace=False)
            display_points = current_sample[display_indices]
        else:
            display_points = current_sample

        y_values = np.ones(len(display_points)) * frame

        point_collections[frame] = ax2.scatter(
            display_points, y_values,
            color=point_color, alpha=0.7, s=15
        )

    ax2.set_ylim(-10, frame + 10)

    if frame % 50 == 0:
        current_ticks = list(ax2.get_yticks())
        if frame not in current_ticks:
            current_ticks.append(frame)
            ax2.set_yticks(current_ticks)

    return sample_kde_line, sample_mean_line, stats_text, histogram


ani = FuncAnimation(fig, update, frames=max_frames, init_func=init,
                    blit=False, interval=50, repeat=False)

plt.tight_layout()
animation_filename = "sample_convergence_animation.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=15, dpi=120, bitrate=-1,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
plt.show()
