import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
import scipy.stats as stats

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12), dpi=100)
fig.patch.set_facecolor('#121212')

gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
ax_main = plt.subplot(gs[0])
ax_z = plt.subplot(gs[1])

ax_main.set_facecolor('#121212')
ax_main.grid(True, linestyle='--', alpha=0.3, color='#555555')
ax_main.tick_params(colors='white', labelsize=16)
for spine in ax_main.spines.values():
    spine.set_color('white')

ax_z.set_facecolor('#121212')
ax_z.grid(True, linestyle='--', alpha=0.3, color='#555555')
ax_z.tick_params(colors='white', labelsize=16)
for spine in ax_z.spines.values():
    spine.set_color('white')

colors = {
    'original': '#FF6347',
    'centered': '#32CD32',
    'standardized': '#00BFFF',
    'highlight': '#FFFF00',
    'text': '#FFFFFF',
    'point_trail': '#FFFF00',
    'area': '#32CD32',
    'arrow': '#FF69B4',
}

x_full = np.linspace(-8, 8, 1000)

lines = {
    'original': ax_main.plot([], [], color=colors['original'], lw=3,
                             label='Original Distribution X ~ N(μ, σ²)')[0],
    'centered': ax_main.plot([], [], color=colors['centered'], lw=3,
                             label='Centered: X - μ ~ N(0, σ²)')[0],
    'standardized': ax_main.plot([], [], color=colors['standardized'], lw=3,
                                 label='Z-Score: Z = (X-μ)/σ ~ N(0, 1)')[0],
    'moving_point_orig': ax_main.plot([], [], 'o', color=colors['original'], ms=12)[0],
    'moving_point_centered': ax_main.plot([], [], 'o', color=colors['centered'], ms=12)[0],
    'moving_point_std': ax_main.plot([], [], 'o', color=colors['standardized'], ms=12)[0],
    'point_trail_orig': ax_main.plot([], [], 'o', color=colors['original'], ms=6, alpha=0.4)[0],
    'point_trail_centered': ax_main.plot([], [], 'o', color=colors['centered'], ms=6, alpha=0.4)[0],
    'point_trail_std': ax_main.plot([], [], 'o', color=colors['standardized'], ms=6, alpha=0.4)[0],
    'z_axis': ax_z.plot([], [], '-', color='white', lw=2)[0],
    'z_tick_marks': ax_z.plot([], [], '|', color='white', ms=20, mew=2)[0],
    'orig_axis': ax_z.plot([], [], '-', color='white', lw=2)[0],
    'orig_tick_marks': ax_z.plot([], [], '|', color='white', ms=20, mew=2)[0],
    'centered_axis': ax_z.plot([], [], '-', color='white', lw=2)[0],
    'centered_tick_marks': ax_z.plot([], [], '|', color='white', ms=20, mew=2)[0],
}

arrows = []
fills = []
annotations = []

text_objects = {
    'title': ax_main.text(0.5, 0.95, '', transform=ax_main.transAxes,
                          fontsize=24, color=colors['text'], ha='center',
                          bbox=dict(facecolor='#121212', alpha=0.8,
                                    edgecolor='white', boxstyle='round,pad=0.5')),
    'parameters': ax_main.text(0.05, 0.95, '', transform=ax_main.transAxes,
                               fontsize=18, color=colors['text'], ha='left',
                               bbox=dict(facecolor='#121212', alpha=0.7,
                                         edgecolor='white', boxstyle='round,pad=0.5')),
    'step_description': ax_main.text(0.5, 0.87, '', transform=ax_main.transAxes,
                                     fontsize=20, color=colors['text'], ha='center',
                                     bbox=dict(facecolor='#121212', alpha=0.8,
                                               edgecolor='white', boxstyle='round,pad=0.5')),
}

point_history = {
    'x_orig': [],
    'y_orig': [],
    'x_centered': [],
    'y_centered': [],
    'x_std': [],
    'y_std': []
}

max_trail_points = 30

ax_main.set_xlabel('Value', fontsize=18, color='white')
ax_main.set_ylabel('Probability Density', fontsize=18, color='white')
ax_z.set_xlabel('Original vs Centered vs Z-Scale', fontsize=18, color='white')
ax_z.set_yticks([])

ax_main.legend(loc='upper right', fontsize=16, framealpha=0.7)

num_frames = 900
animation_mode = 'centering'
mode_duration = num_frames // 2


def normal_pdf(x, mu=0, sigma=1):
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    exp_term = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coef * exp_term


def init():
    global arrows, fills, annotations

    for key, line in lines.items():
        line.set_data([], [])

    for key, text_obj in text_objects.items():
        text_obj.set_text('')

    for arrow in arrows:
        arrow.remove()
    arrows = []

    for fill in fills:
        fill.remove()
    fills = []

    for annotation in annotations:
        annotation.remove()
    annotations = []

    for key in point_history:
        point_history[key] = []

    ax_main.set_xlim(-8, 8)
    ax_main.set_ylim(0, 0.5)
    ax_z.set_xlim(-8, 8)
    ax_z.set_ylim(-1.5, 1.5)

    return tuple(lines.values()) + tuple(text_objects.values())


def update_point_history(x_orig=None, y_orig=None, x_centered=None, y_centered=None, x_std=None, y_std=None):
    if x_orig is not None and y_orig is not None:
        point_history['x_orig'].append(x_orig)
        point_history['y_orig'].append(y_orig)
        if len(point_history['x_orig']) > max_trail_points:
            point_history['x_orig'].pop(0)
            point_history['y_orig'].pop(0)
        lines['point_trail_orig'].set_data(point_history['x_orig'], point_history['y_orig'])

    if x_centered is not None and y_centered is not None:
        point_history['x_centered'].append(x_centered)
        point_history['y_centered'].append(y_centered)
        if len(point_history['x_centered']) > max_trail_points:
            point_history['x_centered'].pop(0)
            point_history['y_centered'].pop(0)
        lines['point_trail_centered'].set_data(point_history['x_centered'], point_history['y_centered'])

    if x_std is not None and y_std is not None:
        point_history['x_std'].append(x_std)
        point_history['y_std'].append(y_std)
        if len(point_history['x_std']) > max_trail_points:
            point_history['x_std'].pop(0)
            point_history['y_std'].pop(0)
        lines['point_trail_std'].set_data(point_history['x_std'], point_history['y_std'])


def clear_annotations():
    global arrows, fills, annotations

    for arrow in arrows:
        if arrow in ax_main.patches:
            arrow.remove()
    arrows = []

    for fill in fills:
        if fill in ax_main.collections:
            fill.remove()
    fills = []

    for annotation in annotations:
        if annotation in ax_main.texts or annotation in ax_z.texts:
            annotation.remove()
    annotations = []


def add_annotation(ax, x, y, text, color, fontsize=16, ha='center', va='center'):
    txt = ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
                  path_effects=[path_effects.withStroke(linewidth=4, foreground='black')])
    annotations.append(txt)
    return txt


def add_arrow(ax, x_start, y_start, x_end, y_end, color, width=0.02, head_width=0.1, head_length=0.2):
    arrow = ax.arrow(x_start, y_start, x_end-x_start, y_end-y_start,
                     fc=color, ec=color, width=width, head_width=head_width,
                     head_length=head_length, length_includes_head=True,
                     alpha=0.8, zorder=5)
    arrows.append(arrow)
    return arrow


def add_fill(ax, x, y1, y2, color, alpha=0.3):
    fill = ax.fill_between(x, y1, y2, color=color, alpha=alpha)
    fills.append(fill)
    return fill


def animate(i):
    global animation_mode, arrows, fills, annotations

    clear_annotations()

    animation_mode = 'centering' if i < mode_duration else 'standardization'
    progress = (i % mode_duration) / mode_duration

    if animation_mode == 'centering':
        sigma = 1.5

        if progress < 0.5:
            mu = 4 * progress
        else:
            mu = 2 - 6 * (progress - 0.5)

        text_objects['title'].set_text('Z-Scoring Step 1: CENTERING')
        text_objects['parameters'].set_text(f'μ = {mu:.2f}, σ = {sigma:.2f}')
        text_objects['step_description'].set_text('X - μ')

        y_original = normal_pdf(x_full, mu, sigma)
        y_centered = normal_pdf(x_full, 0, sigma)

        lines['original'].set_data(x_full, y_original)
        lines['centered'].set_data(x_full, y_centered)
        lines['standardized'].set_data([], [])

        x_point_original = mu + 2 * sigma * np.sin(progress * 6 * np.pi)
        x_point_centered = x_point_original - mu

        if -8 <= x_point_original <= 8 and -8 <= x_point_centered <= 8:
            y_point_original = normal_pdf(x_point_original, mu, sigma)
            y_point_centered = normal_pdf(x_point_centered, 0, sigma)

            lines['moving_point_orig'].set_data([x_point_original], [y_point_original])
            lines['moving_point_centered'].set_data([x_point_centered], [y_point_centered])
            lines['moving_point_std'].set_data([], [])

            update_point_history(
                x_orig=x_point_original,
                y_orig=y_point_original,
                x_centered=x_point_centered,
                y_centered=y_point_centered
            )

            add_arrow(
                ax_main,
                x_point_original,
                y_point_original * 0.9,
                x_point_centered,
                y_point_centered * 0.9,
                colors['arrow']
            )

            add_annotation(
                ax_main,
                (x_point_original + x_point_centered) / 2,
                (y_point_original + y_point_centered) / 2 * 0.7,
                f"-{mu:.1f}",
                colors['highlight'],
                fontsize=16
            )

        z_axis_x = np.linspace(-4, 4, 9)
        z_axis_y = np.zeros_like(z_axis_x)

        orig_axis_x = mu + sigma * z_axis_x
        orig_axis_y = np.zeros_like(orig_axis_x) + 1

        centered_axis_x = 0 + sigma * z_axis_x
        centered_axis_y = np.zeros_like(centered_axis_x)

        lines['orig_axis'].set_data(orig_axis_x, orig_axis_y)
        lines['orig_tick_marks'].set_data(orig_axis_x, orig_axis_y)

        lines['centered_axis'].set_data(centered_axis_x, centered_axis_y)
        lines['centered_tick_marks'].set_data(centered_axis_x, centered_axis_y)

        lines['z_axis'].set_data([], [])
        lines['z_tick_marks'].set_data([], [])

        for i, x in enumerate(orig_axis_x):
            if i % 2 == 0:
                add_annotation(ax_z, x, 1.1, f'{x:.1f}', colors['original'], fontsize=12)
                add_annotation(ax_z, centered_axis_x[i], 0.1, f'{centered_axis_x[i]:.1f}', colors['centered'], fontsize=12)

        add_annotation(ax_z, mu, 0.7, 'Original', colors['original'], fontsize=16)
        add_annotation(ax_z, 0, -0.3, 'Centered', colors['centered'], fontsize=16)

        add_annotation(ax_main, mu, normal_pdf(mu, mu, sigma), "μ", colors['original'], fontsize=18)
        add_annotation(ax_main, 0, normal_pdf(0, 0, sigma), "0", colors['centered'], fontsize=18)

    else:
        mu = 0

        lines['original'].set_data([], [])
        lines['point_trail_orig'].set_data([], [])
        lines['moving_point_orig'].set_data([], [])

        y_standardized = normal_pdf(x_full, 0, 1)
        lines['standardized'].set_data(x_full, y_standardized)

        if progress < 0.5:
            sigma = max(2.5 - 3.0 * progress, 1.0001)
            sigma_progress = (2.5 - sigma) / 1.5

            y_centered = normal_pdf(x_full, 0, sigma)
            lines['centered'].set_data(x_full, y_centered)

            z_value = 1.5
            x_point_standardized = z_value
            x_point_centered = sigma * z_value

            if -8 <= x_point_centered <= 8:
                y_point_centered = normal_pdf(x_point_centered, 0, sigma)
                y_point_standardized = normal_pdf(x_point_standardized, 0, 1)

                lines['moving_point_centered'].set_data([x_point_centered], [y_point_centered])
                lines['moving_point_std'].set_data([x_point_standardized], [y_point_standardized])

                update_point_history(
                    x_centered=x_point_centered,
                    y_centered=y_point_centered,
                    x_std=x_point_standardized,
                    y_std=y_point_standardized
                )

                add_arrow(
                    ax_main,
                    x_point_centered,
                    y_point_centered * 0.9,
                    x_point_standardized,
                    y_point_standardized * 0.9,
                    colors['arrow']
                )

                add_annotation(
                    ax_main,
                    (x_point_centered + x_point_standardized) / 2,
                    (y_point_centered + y_point_standardized) / 2 * 0.7,
                    f"÷{sigma:.1f}",
                    colors['highlight'],
                    fontsize=16
                )

            text_objects['title'].set_text('Z-Scoring Step 2: STANDARDIZATION')
            text_objects['parameters'].set_text(f'σ = {sigma:.2f} (Wide Distribution)')
            text_objects['step_description'].set_text('(X - μ) / σ')

            if sigma < 1.1:
                add_annotation(
                    ax_main,
                    0,
                    0.45,
                    f"CONVERGING TO STANDARD! ({(sigma_progress * 100):.0f}%)",
                    colors['highlight'],
                    fontsize=22
                )
        else:
            sigma = min(0.5 + (progress - 0.5) * 3.0, 0.9999)
            sigma_progress = (sigma - 0.5) / 0.5

            y_centered = normal_pdf(x_full, 0, sigma)
            lines['centered'].set_data(x_full, y_centered)

            z_value = 1.5
            x_point_standardized = z_value
            x_point_centered = sigma * z_value

            if -8 <= x_point_centered <= 8:
                y_point_centered = normal_pdf(x_point_centered, 0, sigma)
                y_point_standardized = normal_pdf(x_point_standardized, 0, 1)

                lines['moving_point_centered'].set_data([x_point_centered], [y_point_centered])
                lines['moving_point_std'].set_data([x_point_standardized], [y_point_standardized])

                update_point_history(
                    x_centered=x_point_centered,
                    y_centered=y_point_centered,
                    x_std=x_point_standardized,
                    y_std=y_point_standardized
                )

                add_arrow(
                    ax_main,
                    x_point_centered,
                    y_point_centered * 0.9,
                    x_point_standardized,
                    y_point_standardized * 0.9,
                    colors['arrow']
                )

                add_annotation(
                    ax_main,
                    (x_point_centered + x_point_standardized) / 2,
                    (y_point_centered + y_point_standardized) / 2 * 0.7,
                    f"÷{sigma:.1f}",
                    colors['highlight'],
                    fontsize=16
                )

            text_objects['title'].set_text('Z-Scoring Step 2: STANDARDIZATION')
            text_objects['parameters'].set_text(f'σ = {sigma:.2f} (Narrow Distribution)')
            text_objects['step_description'].set_text('(X - μ) / σ')

            if sigma > 0.9:
                add_annotation(
                    ax_main,
                    0,
                    0.45,
                    f"CONVERGING TO STANDARD! ({(sigma_progress * 100):.0f}%)",
                    colors['highlight'],
                    fontsize=22
                )

        lines['orig_axis'].set_data([], [])
        lines['orig_tick_marks'].set_data([], [])

        z_axis_x = np.linspace(-4, 4, 9)
        z_axis_y = np.zeros_like(z_axis_x) - 1

        centered_axis_x = sigma * z_axis_x
        centered_axis_y = np.zeros_like(centered_axis_x)

        lines['z_axis'].set_data(z_axis_x, z_axis_y)
        lines['z_tick_marks'].set_data(z_axis_x, z_axis_y)

        lines['centered_axis'].set_data(centered_axis_x, centered_axis_y)
        lines['centered_tick_marks'].set_data(centered_axis_x, centered_axis_y)

        for i, z in enumerate(z_axis_x):
            if i % 2 == 0:
                add_annotation(ax_z, centered_axis_x[i], 0.1, f'{centered_axis_x[i]:.1f}', colors['centered'], fontsize=12)
                add_annotation(ax_z, z, -1.1, f'{z:.1f}', colors['standardized'], fontsize=12)

        add_annotation(ax_z, 0, 0.7, f'σ = {sigma:.1f}', colors['centered'], fontsize=16)
        add_annotation(ax_z, 0, -0.7, 'σ = 1', colors['standardized'], fontsize=16)

    return tuple(lines.values()) + tuple(text_objects.values()) + tuple(arrows) + tuple(fills) + tuple(annotations)


fps = 30
animation_duration = 30
ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                    interval=1000/fps, blit=True)

animation_filename = "z_score_animation_steps.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=fps, dpi=100, bitrate=12000,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
