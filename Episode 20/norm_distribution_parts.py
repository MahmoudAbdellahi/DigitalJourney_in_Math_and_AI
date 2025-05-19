import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
import scipy.stats as stats

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12), dpi=100)
fig.patch.set_facecolor('#121212')

ax = plt.subplot(111)
ax.set_facecolor('#121212')
ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
ax.tick_params(colors='white', labelsize=16)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

colors = {
    'exp_basic': '#98FB98',
    'exp_neg': '#FF9912',
    'exp_squared': '#FFA07A',
    'exp_neg_squared': '#FF6347',
    'exp_shifted': '#9370DB',
    'exp_scaled': '#00BFFF',
    'normal': '#FFFFFF',
    'highlight': '#FFFF00',
    'area': '#32CD32',
    'z_score': '#FF69B4',
    'text': '#FFFFFF',
    'point_trail': '#FFFF00'
}

x_full = np.linspace(-5, 5, 1000)

lines = {
    'exp_basic': ax.plot([], [], color=colors['exp_basic'], lw=3, label='e^x')[0],
    'exp_neg': ax.plot([], [], color=colors['exp_neg'], lw=3, label='e^(-x)')[0],
    'exp_squared': ax.plot([], [], color=colors['exp_squared'], lw=3, label='e^(x²)')[0],
    'exp_neg_squared': ax.plot([], [], color=colors['exp_neg_squared'], lw=3, label='e^(-x²)')[0],
    'exp_shifted': ax.plot([], [], color=colors['exp_shifted'], lw=3, label='e^(-(x-μ)²)')[0],
    'exp_scaled': ax.plot([], [], color=colors['exp_scaled'], lw=3, label='e^(-(x-μ)²/2σ²)')[0],
    'normal': ax.plot([], [], color=colors['normal'], lw=4, label='Normal PDF')[0],
    'moving_point': ax.plot([], [], 'o', color=colors['highlight'], ms=10)[0],
    'z_line': ax.plot([], [], '--', color=colors['z_score'], lw=2)[0],
    'z_point_orig': ax.plot([], [], 'o', color=colors['z_score'], ms=8)[0],
    'z_point_std': ax.plot([], [], 'o', color=colors['z_score'], ms=8)[0],
    'horizontal_line': ax.plot([], [], '--', color='white', lw=1.5, alpha=0.5)[0],
    'point_trail': ax.plot([], [], 'o', color=colors['point_trail'], ms=5, alpha=0.3)[0]
}

area_fill = None
area_labels = []
point_history_x = []
point_history_y = []

formula_text = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                       fontsize=20, color=colors['text'], ha='center',
                       bbox=dict(facecolor='#121212', alpha=0.8,
                                 edgecolor='white', boxstyle='round,pad=0.5'))

parameter_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                         fontsize=18, color=colors['text'], ha='left',
                         bbox=dict(facecolor='#121212', alpha=0.7,
                                   edgecolor='white', boxstyle='round,pad=0.5'))

ax.set_xlabel('Height (cm)', fontsize=18, color='white')
ax.set_ylabel('Probability Density', fontsize=18, color='white')

num_frames = 4500
num_steps = 10
frames_per_step = num_frames // num_steps
max_trail_points = 20

step_functions = [
    "f(x) = e^x",
    "f(x) = e^{-x}",
    "f(x) = e^{x^2}",
    "f(x) = e^{-x^2}",
    "f(x) = e^{-(x-μ)^2}",
    "f(x) = e^{-(x-μ)^2/2σ^2}",
    "f(x) = e^{-x^2/2σ^2} vs \\frac{1}{σ\\sqrt{2π}}e^{-x^2/2σ^2}",
    "f(x) = \\frac{1}{σ\\sqrt{2π}}e^{-(x-μ)^2/2σ^2}",
    "z = \\frac{x-μ}{σ}",
    "Normal Distribution Properties"
]


def normal_pdf(x, mu=0, sigma=1):
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    exp_term = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coef * exp_term


def init():
    global area_fill, area_labels, point_history_x, point_history_y

    for key, line in lines.items():
        line.set_data([], [])

    formula_text.set_text('')
    parameter_text.set_text('')

    if area_fill is not None:
        area_fill.remove()
        area_fill = None

    for label in area_labels:
        label.remove()
    area_labels = []

    point_history_x = []
    point_history_y = []

    return tuple(lines.values()) + (formula_text, parameter_text)


def dynamic_ylim_update(y_point_value=None):
    global point_history_y

    default_y_max = 2

    if y_point_value is not None:
        if point_history_y:
            y_max = max(1.2 * max(max(point_history_y), y_point_value), default_y_max)
        else:
            y_max = max(1.2 * y_point_value, default_y_max)
        ax.set_ylim(0, y_max)
    else:
        if point_history_y:
            y_max = max(1.2 * max(point_history_y), default_y_max)
            ax.set_ylim(0, y_max)
        else:
            ax.set_ylim(0, default_y_max)


def update_point_history(x, y):
    global point_history_x, point_history_y

    if x is not None and y is not None:
        point_history_x.append(x)
        point_history_y.append(y)

        if len(point_history_x) > max_trail_points:
            point_history_x.pop(0)
            point_history_y.pop(0)

        lines['point_trail'].set_data(point_history_x, point_history_y)


def animate(i):
    global area_fill, area_labels, point_history_x, point_history_y

    progress = i / (num_frames - 1)
    step = min(int(progress * num_steps), num_steps - 1)
    step_progress = (progress * num_steps) % 1

    # Check if we've moved to a new step
    if i % frames_per_step == 0:
        point_history_x = []
        point_history_y = []
        lines['point_trail'].set_data([], [])

    for key, line in lines.items():
        if key != 'point_trail':
            line.set_data([], [])

    if area_fill is not None:
        area_fill.remove()
        area_fill = None

    for label in area_labels:
        label.remove()
    area_labels = []

    formula_text.set_text(f'${step_functions[step]}$')
    parameter_text.set_text('')

    moving_point_x = -5 + 10 * (step_progress % 1)

    ax.set_xlim(-5, 5)

    if step == 0:
        y_exp = np.exp(x_full)
        lines['exp_basic'].set_data(x_full, y_exp)

        if -5 <= moving_point_x <= 5:
            y_point_exp = np.exp(moving_point_x)
            lines['moving_point'].set_data([moving_point_x], [y_point_exp])
            update_point_history(moving_point_x, y_point_exp)
            dynamic_ylim_update(y_point_exp)
        else:
            dynamic_ylim_update()

    elif step == 1:
        y_exp_neg = np.exp(-x_full)
        lines['exp_neg'].set_data(x_full, y_exp_neg)

        if -5 <= moving_point_x <= 5:
            y_point_exp_neg = np.exp(-moving_point_x)
            lines['moving_point'].set_data([moving_point_x], [y_point_exp_neg])
            update_point_history(moving_point_x, y_point_exp_neg)
            dynamic_ylim_update(y_point_exp_neg)
        else:
            dynamic_ylim_update()

    elif step == 2:
        y_exp_squared = np.exp(x_full**2)
        lines['exp_squared'].set_data(x_full, y_exp_squared)

        if -5 <= moving_point_x <= 5:
            y_point_exp_squared = np.exp(moving_point_x**2)
            lines['moving_point'].set_data([moving_point_x], [y_point_exp_squared])
            update_point_history(moving_point_x, y_point_exp_squared)

            mirror_x = -moving_point_x if moving_point_x != 0 else 0.01
            mirror_y = np.exp(mirror_x**2)
            lines['z_point_orig'].set_data([mirror_x], [mirror_y])

            if abs(moving_point_x) > 0.2:
                lines['z_line'].set_data([moving_point_x, mirror_x],
                                         [y_point_exp_squared, mirror_y])

            dynamic_ylim_update(y_point_exp_squared)
        else:
            dynamic_ylim_update()

    elif step == 3:
        y_exp_neg_squared = np.exp(-x_full**2)
        lines['exp_neg_squared'].set_data(x_full, y_exp_neg_squared)

        if -5 <= moving_point_x <= 5:
            y_point_exp_neg_squared = np.exp(-moving_point_x**2)
            lines['moving_point'].set_data([moving_point_x], [y_point_exp_neg_squared])
            update_point_history(moving_point_x, y_point_exp_neg_squared)

            mirror_x = -moving_point_x if moving_point_x != 0 else 0.01
            mirror_y = np.exp(-mirror_x**2)
            lines['z_point_orig'].set_data([mirror_x], [mirror_y])

            if abs(moving_point_x) < 0.1:
                lines['z_line'].set_data([-5, 5], [1, 1])
            elif abs(moving_point_x) > 0.2:
                lines['z_line'].set_data([moving_point_x, mirror_x],
                                         [y_point_exp_neg_squared, mirror_y])

            dynamic_ylim_update(y_point_exp_neg_squared)
        else:
            dynamic_ylim_update()

    elif step == 4:
        mu = -3 + 6 * step_progress

        y_shifted_correct = np.exp(-(x_full - mu)**2)
        y_shifted_wrong = np.exp(-(x_full + mu)**2)

        if step_progress < 0.5:
            lines['exp_shifted'].set_data(x_full, y_shifted_wrong)
            lines['exp_neg_squared'].set_data(x_full, y_shifted_correct)

            if -5 <= moving_point_x <= 5:
                y_point_shifted = np.exp(-(moving_point_x + mu)**2)
                lines['moving_point'].set_data([moving_point_x], [y_point_shifted])
                update_point_history(moving_point_x, y_point_shifted)
                dynamic_ylim_update(y_point_shifted)
            else:
                dynamic_ylim_update()

            formula_text.set_text(f'$f(x) = e^{{-(x+\\mu)^2}}$ vs $f(x) = e^{{-(x-\\mu)^2}}$')
        else:
            lines['exp_shifted'].set_data(x_full, y_shifted_correct)

            if -5 <= moving_point_x <= 5:
                y_point_shifted = np.exp(-(moving_point_x - mu)**2)
                lines['moving_point'].set_data([moving_point_x], [y_point_shifted])
                update_point_history(moving_point_x, y_point_shifted)
                dynamic_ylim_update(y_point_shifted)
            else:
                dynamic_ylim_update()

            formula_text.set_text(f'$f(x) = e^{{-(x-\\mu)^2}}$')

        parameter_text.set_text(f'μ = {mu:.2f} cm')

    elif step == 5:
        mu = 0
        sigma = 0.5 + 2.5 * step_progress

        y_scaled_correct = np.exp(-(x_full - mu)**2 / (2 * sigma**2))
        y_scaled_wrong = np.exp(-(x_full - mu)**2 * sigma**2 / 2)

        if step_progress < 0.5:
            lines['exp_neg_squared'].set_data(x_full, y_scaled_wrong)
            lines['exp_scaled'].set_data(x_full, y_scaled_correct)

            if -5 <= moving_point_x <= 5:
                y_point_scaled = np.exp(-(moving_point_x - mu)**2 * sigma**2 / 2)
                lines['moving_point'].set_data([moving_point_x], [y_point_scaled])
                update_point_history(moving_point_x, y_point_scaled)
                dynamic_ylim_update(y_point_scaled)
            else:
                dynamic_ylim_update()

            formula_text.set_text(f'$f(x) = e^{{-\\frac{{(x-\\mu)^2 \\cdot \\sigma^2}}{{2}}}}$ vs $f(x) = e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}$')
        else:
            lines['exp_scaled'].set_data(x_full, y_scaled_correct)

            if -5 <= moving_point_x <= 5:
                y_point_scaled = np.exp(-(moving_point_x - mu)**2 / (2 * sigma**2))
                lines['moving_point'].set_data([moving_point_x], [y_point_scaled])
                update_point_history(moving_point_x, y_point_scaled)
                dynamic_ylim_update(y_point_scaled)
            else:
                dynamic_ylim_update()

            formula_text.set_text(f'$f(x) = e^{{-\\frac{{(x-\\mu)^2}}{{2\\sigma^2}}}}$')

        parameter_text.set_text(f'σ = {sigma:.2f} cm')

    elif step == 6:
        sigma = 1

        if step_progress < 0.5:
            y_unnormalized = np.exp(-x_full**2 / (2 * sigma**2))
            lines['exp_scaled'].set_data(x_full, y_unnormalized)

            area = np.trapz(y_unnormalized, x_full)

            area_fill = ax.fill_between(x_full, 0, y_unnormalized,
                                        color=colors['area'], alpha=0.3)

            peak_value = np.exp(0)
            lines['z_point_orig'].set_data([0], [peak_value])
            update_point_history(0, peak_value)

            lines['horizontal_line'].set_data([-5, 5], [peak_value, peak_value])

            formula_text.set_text(f'$f(x) = e^{{-\\frac{{x^2}}{{2\\sigma^2}}}}$, Area = {area:.4f}')
            dynamic_ylim_update(peak_value)
        else:
            coef = 1 / (sigma * np.sqrt(2 * np.pi))
            y_normalized = coef * np.exp(-x_full**2 / (2 * sigma**2))
            lines['normal'].set_data(x_full, y_normalized)

            area_fill = ax.fill_between(x_full, 0, y_normalized,
                                        color=colors['area'], alpha=0.3)

            peak_value = coef * np.exp(0)
            lines['z_point_orig'].set_data([0], [peak_value])
            update_point_history(0, peak_value)

            lines['horizontal_line'].set_data([-5, 5], [peak_value, peak_value])

            formula_text.set_text(f'$f(x) = \\frac{{1}}{{\\sigma\\sqrt{{2\\pi}}}} e^{{-\\frac{{x^2}}{{2\\sigma^2}}}}$, Area = 1.0000')
            dynamic_ylim_update(peak_value)

    elif step == 7:
        mu_cycle = -2 + 4 * (step_progress % 0.5) if step_progress < 0.5 else 0
        sigma_cycle = 0.5 + (step_progress % 0.5) * 2 if step_progress >= 0.5 else 1

        y_normal = normal_pdf(x_full, mu_cycle, sigma_cycle)
        lines['normal'].set_data(x_full, y_normal)

        if -5 <= moving_point_x <= 5:
            y_point_normal = normal_pdf(moving_point_x, mu_cycle, sigma_cycle)
            lines['moving_point'].set_data([moving_point_x], [y_point_normal])
            update_point_history(moving_point_x, y_point_normal)
            dynamic_ylim_update(y_point_normal)
        else:
            dynamic_ylim_update()

        parameter_text.set_text(f'μ = {mu_cycle:.2f} cm, σ = {sigma_cycle:.2f} cm')

    elif step == 8:
        y_standard = normal_pdf(x_full)
        lines['normal'].set_data(x_full, y_standard)

        mu = 2 * np.sin(step_progress * 2 * np.pi)
        sigma = 1 + 0.5 * np.sin(step_progress * 4 * np.pi)

        y_nonstandard = normal_pdf(x_full, mu, sigma)
        lines['exp_shifted'].set_data(x_full, y_nonstandard)

        if -5 <= moving_point_x <= 5:
            y_point_nonstandard = normal_pdf(moving_point_x, mu, sigma)
            lines['moving_point'].set_data([moving_point_x], [y_point_nonstandard])
            update_point_history(moving_point_x, y_point_nonstandard)

            z_score = (moving_point_x - mu) / sigma

            if -5 <= z_score <= 5:
                y_point_standard = normal_pdf(z_score)
                lines['z_point_std'].set_data([z_score], [y_point_standard])
                lines['z_line'].set_data([moving_point_x, z_score],
                                         [y_point_nonstandard, y_point_standard])

            dynamic_ylim_update(max(y_point_nonstandard, y_point_standard if -5 <= z_score <= 5 else 0))
        else:
            dynamic_ylim_update()

        parameter_text.set_text(f'μ = {mu:.2f} cm, σ = {sigma:.2f} cm')

    elif step == 9:
        y_normal = normal_pdf(x_full)
        lines['normal'].set_data(x_full, y_normal)

        sub_step = int(step_progress * 3)
        peak_value = normal_pdf(0)
        dynamic_ylim_update(1.2 * peak_value)

        if sub_step == 0:
            region_start, region_end = -1, 1
            area_prob = 0.6827
            formula_text.set_text('$Standard\\ Normal\\ Distribution\\ (σ = 1\\ cm)$')

            region_mask = (x_full >= region_start) & (x_full <= region_end)
            region_x = x_full[region_mask]
            region_y = y_normal[region_mask]
            area_fill = ax.fill_between(region_x, 0, region_y, color=colors['area'], alpha=0.5)

            area_label = ax.text(0, 0.3, f"68.27%\n±1σ", ha='center', va='center',
                                 fontsize=18, color='black', weight='bold')
            area_labels.append(area_label)

        elif sub_step == 1:
            region_start, region_end = -2, 2
            area_prob = 0.9545
            formula_text.set_text('$Standard\\ Normal\\ Distribution\\ (σ = 1\\ cm)$')

            region_mask = (x_full >= region_start) & (x_full <= region_end)
            region_x = x_full[region_mask]
            region_y = y_normal[region_mask]
            area_fill = ax.fill_between(region_x, 0, region_y, color=colors['area'], alpha=0.5)

            area_label = ax.text(0, 0.3, f"95.45%\n±2σ", ha='center', va='center',
                                 fontsize=18, color='black', weight='bold')
            area_labels.append(area_label)

            outlier_text_left = ax.text(-3.5, 0.05, "Outliers\n(2.28%)", ha='center',
                                        fontsize=14, color='white', alpha=0.8)
            outlier_text_right = ax.text(3.5, 0.05, "Outliers\n(2.28%)", ha='center',
                                         fontsize=14, color='white', alpha=0.8)
            area_labels.extend([outlier_text_left, outlier_text_right])

        else:
            region_start, region_end = -3, 3
            area_prob = 0.9973
            formula_text.set_text('$Standard\\ Normal\\ Distribution\\ (σ = 1\\ cm)$')

            region_mask = (x_full >= region_start) & (x_full <= region_end)
            region_x = x_full[region_mask]
            region_y = y_normal[region_mask]
            area_fill = ax.fill_between(region_x, 0, region_y, color=colors['area'], alpha=0.5)

            area_label = ax.text(0, 0.3, f"99.73%\n±3σ", ha='center', va='center',
                                 fontsize=18, color='black', weight='bold')
            area_labels.append(area_label)

    if hasattr(lines['moving_point'], 'set_path_effects'):
        lines['moving_point'].set_path_effects([
            path_effects.Stroke(linewidth=12, foreground=colors['highlight'], alpha=0.5),
            path_effects.Normal()
        ])

    return_elements = list(lines.values()) + [formula_text, parameter_text]

    if area_fill is not None:
        return_elements.append(area_fill)

    for label in area_labels:
        return_elements.append(label)

    return tuple(return_elements)


fps = 30
animation_duration = 225
ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                    interval=1000/fps, blit=True)

animation_filename = "normal_distribution_animation.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=fps, dpi=100, bitrate=12000,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
