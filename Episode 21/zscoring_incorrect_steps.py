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
    'arrow': '#FF69B4',
    'incorrect': '#FF00FF',
}

x_full = np.linspace(-8, 8, 1000)

lines = {
    'original': ax_main.plot([], [], color=colors['original'], lw=3,
                             label='Original Distribution X ~ N(μ, σ²)')[0],
    'correct': ax_main.plot([], [], color=colors['centered'], lw=3,
                            label='Correct Operation')[0],
    'incorrect': ax_main.plot([], [], color=colors['incorrect'], lw=3,
                              label='Incorrect Operation', linestyle='--')[0],
    'moving_point_orig': ax_main.plot([], [], 'o', color=colors['original'], ms=12)[0],
    'moving_point_correct': ax_main.plot([], [], 'o', color=colors['centered'], ms=12)[0],
    'moving_point_incorrect': ax_main.plot([], [], 'o', color=colors['incorrect'], ms=12)[0],
    'point_trail_orig': ax_main.plot([], [], 'o', color=colors['original'], ms=6, alpha=0.4)[0],
    'point_trail_correct': ax_main.plot([], [], 'o', color=colors['centered'], ms=6, alpha=0.4)[0],
    'point_trail_incorrect': ax_main.plot([], [], 'o', color=colors['incorrect'], ms=6, alpha=0.4)[0],
    'orig_axis': ax_z.plot([], [], '-', color='white', lw=2)[0],
    'orig_tick_marks': ax_z.plot([], [], '|', color='white', ms=20, mew=2)[0],
    'correct_axis': ax_z.plot([], [], '-', color=colors['centered'], lw=2)[0],
    'correct_tick_marks': ax_z.plot([], [], '|', color=colors['centered'], ms=20, mew=2)[0],
    'incorrect_axis': ax_z.plot([], [], '-', color=colors['incorrect'], lw=2, linestyle='--')[0],
    'incorrect_tick_marks': ax_z.plot([], [], '|', color=colors['incorrect'], ms=20, mew=2)[0],
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
    'x_correct': [],
    'y_correct': [],
    'x_incorrect': [],
    'y_incorrect': []
}

max_trail_points = 30

ax_main.set_xlabel('Value', fontsize=18, color='white')
ax_main.set_ylabel('Probability Density', fontsize=18, color='white')
ax_z.set_xlabel('Scales Comparison', fontsize=18, color='white')
ax_z.set_yticks([])

ax_main.legend(loc='upper right', fontsize=14, framealpha=0.7)

num_frames = 900
num_phases = 2
phase_duration = num_frames // num_phases


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


def update_point_history(x_orig=None, y_orig=None,
                         x_correct=None, y_correct=None,
                         x_incorrect=None, y_incorrect=None):
    if x_orig is not None and y_orig is not None:
        point_history['x_orig'].append(x_orig)
        point_history['y_orig'].append(y_orig)
        if len(point_history['x_orig']) > max_trail_points:
            point_history['x_orig'].pop(0)
            point_history['y_orig'].pop(0)
        lines['point_trail_orig'].set_data(point_history['x_orig'], point_history['y_orig'])

    if x_correct is not None and y_correct is not None:
        point_history['x_correct'].append(x_correct)
        point_history['y_correct'].append(y_correct)
        if len(point_history['x_correct']) > max_trail_points:
            point_history['x_correct'].pop(0)
            point_history['y_correct'].pop(0)
        lines['point_trail_correct'].set_data(point_history['x_correct'], point_history['y_correct'])

    if x_incorrect is not None and y_incorrect is not None:
        point_history['x_incorrect'].append(x_incorrect)
        point_history['y_incorrect'].append(y_incorrect)
        if len(point_history['x_incorrect']) > max_trail_points:
            point_history['x_incorrect'].pop(0)
            point_history['y_incorrect'].pop(0)
        lines['point_trail_incorrect'].set_data(point_history['x_incorrect'], point_history['y_incorrect'])


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


def animate(i):
    global arrows, fills, annotations

    clear_annotations()

    phase = i // phase_duration
    progress = (i % phase_duration) / phase_duration

    if phase == 0:
        sigma = 1.5

        if progress < 0.5:
            mu = 4 * progress
        else:
            mu = 2 - 6 * (progress - 0.5)

        text_objects['title'].set_text('INCORRECT CENTERING')
        text_objects['parameters'].set_text(f'μ = {mu:.2f}, σ = {sigma:.2f}')
        text_objects['step_description'].set_text('X + μ (WRONG!) vs X - μ (CORRECT)')

        y_original = normal_pdf(x_full, mu, sigma)
        y_correct = normal_pdf(x_full, 0, sigma)
        y_incorrect = normal_pdf(x_full, 2*mu, sigma)

        lines['original'].set_data(x_full, y_original)
        lines['correct'].set_data(x_full, y_correct)
        lines['incorrect'].set_data(x_full, y_incorrect)

        x_point_original = mu + 2 * sigma * np.sin(progress * 6 * np.pi)
        x_point_correct = x_point_original - mu
        x_point_incorrect = x_point_original + mu

        if -8 <= x_point_original <= 8 and -8 <= x_point_incorrect <= 8:
            y_point_original = normal_pdf(x_point_original, mu, sigma)
            y_point_correct = normal_pdf(x_point_correct, 0, sigma)
            y_point_incorrect = normal_pdf(x_point_incorrect, 2*mu, sigma)

            lines['moving_point_orig'].set_data([x_point_original], [y_point_original])
            lines['moving_point_correct'].set_data([x_point_correct], [y_point_correct])
            lines['moving_point_incorrect'].set_data([x_point_incorrect], [y_point_incorrect])

            update_point_history(
                x_orig=x_point_original,
                y_orig=y_point_original,
                x_correct=x_point_correct,
                y_correct=y_point_correct,
                x_incorrect=x_point_incorrect,
                y_incorrect=y_point_incorrect
            )

            add_arrow(
                ax_main,
                x_point_original,
                y_point_original * 0.9,
                x_point_correct,
                y_point_correct * 0.9,
                colors['centered']
            )

            add_arrow(
                ax_main,
                x_point_original,
                y_point_original * 0.9,
                x_point_incorrect,
                y_point_incorrect * 0.9,
                colors['incorrect']
            )

            add_annotation(
                ax_main,
                (x_point_original + x_point_correct) / 2,
                (y_point_original + y_point_correct) / 2 * 0.7,
                f"-{mu:.1f}",
                colors['centered'],
                fontsize=16
            )

            add_annotation(
                ax_main,
                (x_point_original + x_point_incorrect) / 2,
                (y_point_original + y_point_incorrect) / 2 * 0.7,
                f"+{mu:.1f}",
                colors['incorrect'],
                fontsize=16
            )

        z_axis_x = np.linspace(-4, 4, 9)
        z_axis_y = np.zeros_like(z_axis_x)

        orig_axis_x = mu + sigma * z_axis_x
        orig_axis_y = np.zeros_like(orig_axis_x) + 1

        correct_axis_x = 0 + sigma * z_axis_x
        correct_axis_y = np.zeros_like(correct_axis_x)

        incorrect_axis_x = 2*mu + sigma * z_axis_x
        incorrect_axis_y = np.zeros_like(incorrect_axis_x) - 1

        lines['orig_axis'].set_data(orig_axis_x, orig_axis_y)
        lines['orig_tick_marks'].set_data(orig_axis_x, orig_axis_y)

        lines['correct_axis'].set_data(correct_axis_x, correct_axis_y)
        lines['correct_tick_marks'].set_data(correct_axis_x, correct_axis_y)

        lines['incorrect_axis'].set_data(incorrect_axis_x, incorrect_axis_y)
        lines['incorrect_tick_marks'].set_data(incorrect_axis_x, incorrect_axis_y)

        for i, x in enumerate(orig_axis_x):
            if i % 2 == 0:
                add_annotation(ax_z, x, 1.1, f'{x:.1f}', colors['original'], fontsize=12)
                add_annotation(ax_z, correct_axis_x[i], 0.1, f'{correct_axis_x[i]:.1f}', colors['centered'], fontsize=12)
                add_annotation(ax_z, incorrect_axis_x[i], -1.1, f'{incorrect_axis_x[i]:.1f}', colors['incorrect'], fontsize=12)

        add_annotation(ax_z, mu, 0.7, 'Original', colors['original'], fontsize=16)
        add_annotation(ax_z, 0, 0.0, 'Correct: X - μ', colors['centered'], fontsize=16)
        add_annotation(ax_z, 2*mu, -0.7, 'Incorrect: X + μ', colors['incorrect'], fontsize=16)

        add_annotation(
            ax_main,
            0,
            0.45,
            "ADDING μ MOVES AWAY FROM ZERO!",
            colors['incorrect'],
            fontsize=22
        )

    elif phase == 1:
        mu = 0

        if i % 30 == 0:
            for key in point_history:
                point_history[key] = []

            for key, line in lines.items():
                if 'trail' in key:
                    line.set_data([], [])

        if progress < 0.5:
            sigma = 2.0
        else:
            sigma = 0.5

        text_objects['title'].set_text('INCORRECT STANDARDIZATION')
        text_objects['parameters'].set_text(f'σ = {sigma:.2f}')
        text_objects['step_description'].set_text('(X - μ) × σ (WRONG!) vs (X - μ) ÷ σ (CORRECT)')

        y_centered = normal_pdf(x_full, 0, sigma)

        y_correct = normal_pdf(x_full, 0, 1)

        if sigma > 1:
            y_incorrect = normal_pdf(x_full, 0, sigma * sigma)
        else:
            y_incorrect = normal_pdf(x_full, 0, sigma * sigma)

        lines['original'].set_data(x_full, y_centered)
        lines['correct'].set_data(x_full, y_correct)
        lines['incorrect'].set_data(x_full, y_incorrect)

        z_value = 1.5
        x_point_original = sigma * z_value
        x_point_correct = z_value
        x_point_incorrect = sigma * sigma * z_value

        if -8 <= x_point_original <= 8 and -8 <= x_point_incorrect <= 8:
            y_point_original = normal_pdf(x_point_original, 0, sigma)
            y_point_correct = normal_pdf(x_point_correct, 0, 1)
            y_point_incorrect = normal_pdf(x_point_incorrect, 0, sigma * sigma)

            lines['moving_point_orig'].set_data([x_point_original], [y_point_original])
            lines['moving_point_correct'].set_data([x_point_correct], [y_point_correct])
            lines['moving_point_incorrect'].set_data([x_point_incorrect], [y_point_incorrect])

            update_point_history(
                x_orig=x_point_original,
                y_orig=y_point_original,
                x_correct=x_point_correct,
                y_correct=y_point_correct,
                x_incorrect=x_point_incorrect,
                y_incorrect=y_point_incorrect
            )

            add_arrow(
                ax_main,
                x_point_original,
                y_point_original * 0.9,
                x_point_correct,
                y_point_correct * 0.9,
                colors['centered']
            )

            add_arrow(
                ax_main,
                x_point_original,
                y_point_original * 0.9,
                x_point_incorrect,
                y_point_incorrect * 0.9,
                colors['incorrect']
            )

            add_annotation(
                ax_main,
                (x_point_original + x_point_correct) / 2,
                (y_point_original + y_point_correct) / 2 * 0.7,
                f"÷{sigma:.1f}",
                colors['centered'],
                fontsize=16
            )

            add_annotation(
                ax_main,
                (x_point_original + x_point_incorrect) / 2,
                (y_point_original + y_point_incorrect) / 2 * 0.7,
                f"×{sigma:.1f}",
                colors['incorrect'],
                fontsize=16
            )

        z_axis_x = np.linspace(-4, 4, 9)
        z_axis_y = np.zeros_like(z_axis_x) - 1

        centered_axis_x = sigma * z_axis_x
        centered_axis_y = np.zeros_like(centered_axis_x) + 1

        correct_axis_x = z_axis_x
        correct_axis_y = np.zeros_like(correct_axis_x)

        incorrect_axis_x = sigma * sigma * z_axis_x
        incorrect_axis_y = np.zeros_like(incorrect_axis_x) - 1

        lines['orig_axis'].set_data(centered_axis_x, centered_axis_y)
        lines['orig_tick_marks'].set_data(centered_axis_x, centered_axis_y)

        lines['correct_axis'].set_data(correct_axis_x, correct_axis_y)
        lines['correct_tick_marks'].set_data(correct_axis_x, correct_axis_y)

        lines['incorrect_axis'].set_data(incorrect_axis_x, incorrect_axis_y)
        lines['incorrect_tick_marks'].set_data(incorrect_axis_x, incorrect_axis_y)

        for i, z in enumerate(z_axis_x):
            if i % 2 == 0:
                add_annotation(ax_z, centered_axis_x[i], 1.1, f'{centered_axis_x[i]:.1f}', colors['original'], fontsize=12)
                add_annotation(ax_z, correct_axis_x[i], 0.1, f'{correct_axis_x[i]:.1f}', colors['centered'], fontsize=12)
                add_annotation(ax_z, incorrect_axis_x[i], -1.1, f'{incorrect_axis_x[i]:.1f}', colors['incorrect'], fontsize=12)

        add_annotation(ax_z, 0, 0.7, f'Centered (σ = {sigma:.1f})', colors['original'], fontsize=16)
        add_annotation(ax_z, 0, 0.0, 'Correct: (X-μ) ÷ σ', colors['centered'], fontsize=16)
        add_annotation(ax_z, 0, -0.7, 'Incorrect: (X-μ) × σ', colors['incorrect'], fontsize=16)

        if sigma > 1:
            warning_msg = f"MULTIPLYING BY σ MAKES IT WIDER! (σ² = {sigma*sigma:.2f})"
        else:
            warning_msg = f"MULTIPLYING BY σ MAKES IT NARROWER! (σ² = {sigma*sigma:.2f})"

        add_annotation(
            ax_main,
            0,
            0.45,
            warning_msg,
            colors['incorrect'],
            fontsize=22
        )

        add_annotation(
            ax_main,
            0,
            normal_pdf(0, 0, 1) * 1.1,
            "GOAL: σ = 1",
            colors['centered'],
            fontsize=20
        )

    return tuple(lines.values()) + tuple(text_objects.values()) + tuple(arrows) + tuple(fills) + tuple(annotations)


fps = 30
animation_duration = 30
ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                    interval=1000/fps, blit=True)

animation_filename = "z_score_incorrect_operations.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=fps, dpi=100, bitrate=12000,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
