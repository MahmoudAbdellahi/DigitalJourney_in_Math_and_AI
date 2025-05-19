import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import stats
import matplotlib.patheffects as path_effects

# ====== CONFIGURATION FLAG ======
# Set this value to control how many figures to show:
# 1: Shows only the growth comparison
# 2: Shows only the derivatives
# 3: Shows all three figures
DISPLAY_MODE = 3  # Change this value to 1, 2, or 3
# ==============================

plt.style.use('dark_background')

if DISPLAY_MODE == 1:
    fig = plt.figure(figsize=(21, 21), dpi=100)
    fig.patch.set_facecolor('#121212')
    ax1 = fig.add_subplot(111)
    axes = [ax1]

elif DISPLAY_MODE == 2:
    fig = plt.figure(figsize=(21, 21), dpi=100)
    fig.patch.set_facecolor('#121212')
    ax2 = fig.add_subplot(111)
    axes = [ax2]

else:
    fig = plt.figure(figsize=(21, 21), dpi=100)
    fig.patch.set_facecolor('#121212')
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    axes = [ax1, ax2, ax3]

for ax in axes:
    ax.set_facecolor('#121212')
    ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
    ax.tick_params(colors='white', labelsize=18)

cubic_color = '#98FB98'
linear3x_color = '#FF9912'
exp2_color = '#FF6347'
expe_color = '#9370DB'
exp3_color = '#00BFFF'
glow_color = '#FFFF00'

x_full = np.linspace(-15, 15, 3000)

if DISPLAY_MODE == 1 or DISPLAY_MODE == 3:
    ax1.set_xlabel('x', fontsize=24, color='white')
    ax1.set_ylabel('f(x)', fontsize=24, color='white')
    ax1.set_title('Growth Comparison', fontsize=28, color='white', pad=20)

    cubic_line, = ax1.plot([], [], color=cubic_color, lw=4, label='f(x) = x³')
    linear3x_line, = ax1.plot([], [], color=linear3x_color, lw=4, label='f(x) = 3x')
    exp2_line, = ax1.plot([], [], color=exp2_color, lw=4, label='f(x) = 2ˣ')
    expe_line, = ax1.plot([], [], color=expe_color, lw=4, label='f(x) = eˣ')

    ax1.legend(loc='upper left', fontsize=20)

if DISPLAY_MODE == 2 or DISPLAY_MODE == 3:
    ax2.set_xlabel('x', fontsize=24, color='white')
    ax2.set_ylabel('Value', fontsize=24, color='white')
    ax2.set_title('Exponential Functions and Their Derivatives',
                  fontsize=28, color='white', pad=20)

    exp2_func_line, = ax2.plot([], [], color=exp2_color, lw=4, label='f(x) = 2ˣ')
    expe_func_line, = ax2.plot([], [], color=expe_color, lw=4, label='f(x) = eˣ')
    exp3_func_line, = ax2.plot([], [], color=exp3_color, lw=4, label='f(x) = 3ˣ')

    exp2_deriv_line, = ax2.plot([], [], color=exp2_color, lw=3, linestyle='--', alpha=0.7, label="f'(x) = ln(2)·2ˣ")
    expe_deriv_line, = ax2.plot([], [], color=expe_color, lw=3, linestyle='--', alpha=0.7, label="f'(x) = eˣ")
    exp3_deriv_line, = ax2.plot([], [], color=exp3_color, lw=3, linestyle='--', alpha=0.7, label="f'(x) = ln(3)·3ˣ")

    for line, color in [
        ('exp2_tangent', exp2_color),
        ('expe_tangent', expe_color),
        ('exp3_tangent', exp3_color)
    ]:
        line_obj, = ax2.plot([], [], color=color, lw=4)
        line_obj.set_path_effects([
            path_effects.Stroke(linewidth=6, foreground=glow_color, alpha=0.7),
            path_effects.Normal()
        ])
        globals()[line] = line_obj

    persistent_tangents = []
    for cycle in range(4):
        for func in ['exp2', 'expe', 'exp3']:
            color = globals()[f"{func}_color"]
            line, = ax2.plot([], [], color=color, lw=2, alpha=0.5)
            persistent_tangents.append((cycle, func, line))

    tangent_point2, = ax2.plot([], [], 'o', color=exp2_color, ms=8)
    tangent_pointe, = ax2.plot([], [], 'o', color=expe_color, ms=8)
    tangent_point3, = ax2.plot([], [], 'o', color=exp3_color, ms=8)

    ax2.legend(loc='upper left', fontsize=20)

if DISPLAY_MODE == 3:
    ax3.set_xlabel('Time (t)', fontsize=24, color='white')
    ax3.set_ylabel('Value', fontsize=24, color='white')
    ax3.set_title('Exponential Growth and Decay',
                  fontsize=28, color='white', pad=20)

    growth_line1, = ax3.plot([], [], color='#FF9912', lw=4, label='Growth: e^(t)')
    growth_line2, = ax3.plot([], [], color=expe_color, lw=4, label='Growth: e^(2t)')
    decay_line1, = ax3.plot([], [], color='#FF9912', lw=4, linestyle='--', label='Decay: e^(-t)')
    decay_line2, = ax3.plot([], [], color=expe_color, lw=4, linestyle='--', label='Decay: e^(-2t)')

    ax3.legend(loc='upper right', fontsize=20)

num_frames = 600


def calculate_tangent_line(x_point, func, deriv, x_range=0.5):
    y_point = func(x_point)
    slope = deriv(x_point)

    x_tangent = np.array([x_point - x_range, x_point + x_range])
    y_tangent = y_point + slope * (x_tangent - x_point)

    return x_tangent, y_tangent, y_point


def init():
    lines_to_return = []

    if DISPLAY_MODE == 1 or DISPLAY_MODE == 3:
        cubic_line.set_data([], [])
        linear3x_line.set_data([], [])
        exp2_line.set_data([], [])
        expe_line.set_data([], [])
        lines_to_return.extend([cubic_line, linear3x_line, exp2_line, expe_line])

    if DISPLAY_MODE == 2 or DISPLAY_MODE == 3:
        exp2_func_line.set_data([], [])
        expe_func_line.set_data([], [])
        exp3_func_line.set_data([], [])

        exp2_deriv_line.set_data([], [])
        expe_deriv_line.set_data([], [])
        exp3_deriv_line.set_data([], [])

        exp2_tangent.set_data([], [])
        expe_tangent.set_data([], [])
        exp3_tangent.set_data([], [])

        tangent_point2.set_data([], [])
        tangent_pointe.set_data([], [])
        tangent_point3.set_data([], [])

        for _, _, line in persistent_tangents:
            line.set_data([], [])

        lines_to_return.extend([
            exp2_func_line, expe_func_line, exp3_func_line,
            exp2_deriv_line, expe_deriv_line, exp3_deriv_line,
            exp2_tangent, expe_tangent, exp3_tangent,
            tangent_point2, tangent_pointe, tangent_point3
        ])
        lines_to_return.extend([line for _, _, line in persistent_tangents])

    if DISPLAY_MODE == 3:
        growth_line1.set_data([], [])
        growth_line2.set_data([], [])
        decay_line1.set_data([], [])
        decay_line2.set_data([], [])

        lines_to_return.extend([
            growth_line1, growth_line2, decay_line1, decay_line2
        ])

    return tuple(lines_to_return)


def safe_max(arr, default=0):
    """Safely get maximum of array, handling empty arrays"""
    if len(arr) > 0:
        return np.max(arr)
    return default


cycle_tangents = {cycle: {func: {'x': [], 'y': []} for func in ['exp2', 'expe', 'exp3']} for cycle in range(4)}


def animate(i):
    progress = i / (num_frames - 1)
    lines_to_return = []

    if DISPLAY_MODE == 1 or DISPLAY_MODE == 3:
        if progress < 0.15:
            current_limit = 0.5 + (progress/0.15) * 1.5
        else:
            adjusted_progress = (progress - 0.15) / 0.85
            current_limit = 2 + adjusted_progress * 8

        x_min = -current_limit
        x_max = current_limit

        mask = (x_full >= x_min) & (x_full <= x_max)
        x_current = x_full[mask]

        y_cubic = x_current**3
        y_linear3x = 3 * x_current
        y_exp2 = 2**x_current
        y_expe = np.exp(x_current)

        y_max_exp = max(safe_max(y_exp2), safe_max(y_expe))

        if progress < 0.15:
            ax1.set_ylim(-5, 10)
        else:
            ax1_y_limit = min(50000, max(10, y_max_exp * 1.1))
            y_min_cubic = min(safe_max(y_cubic, 0), -5)
            ax1.set_ylim(y_min_cubic, ax1_y_limit)

        ax1.set_xlim(x_min, x_max)

        cubic_line.set_data(x_current, y_cubic)
        linear3x_line.set_data(x_current, y_linear3x)
        exp2_line.set_data(x_current, y_exp2)
        expe_line.set_data(x_current, y_expe)

        lines_to_return.extend([cubic_line, linear3x_line, exp2_line, expe_line])

    if DISPLAY_MODE == 2 or DISPLAY_MODE == 3:
        def f_exp2(x): return 2**x
        def f_expe(x): return np.exp(x)
        def f_exp3(x): return 3**x

        def d_exp2(x): return np.log(2) * 2**x
        def d_expe(x): return np.exp(x)
        def d_exp3(x): return np.log(3) * 3**x

        num_cycles = 4
        cycle_time = 1.0 / num_cycles

        growth_fraction = 0.4
        tangent_fraction = 0.6

        cycle_index = int(progress / cycle_time)
        if cycle_index >= num_cycles:
            cycle_index = num_cycles - 1

        cycle_progress = (progress - cycle_index * cycle_time) / cycle_time

        in_growth_phase = cycle_progress < growth_fraction

        x_limits = [2, 4, 7, 10, 12]

        if in_growth_phase:

            growth_progress = cycle_progress / growth_fraction

            start_limit = 0 if cycle_index == 0 else x_limits[cycle_index - 1]
            end_limit = x_limits[cycle_index]

            current_x_limit = start_limit + growth_progress * (end_limit - start_limit)

            x_min = -current_x_limit
            x_max = current_x_limit

            mask = (x_full >= x_min) & (x_full <= x_max)
            x_current = x_full[mask]

            y_exp2 = 2**x_current
            y_expe = np.exp(x_current)
            y_exp3 = 3**x_current

            y_max_all = max(safe_max(y_exp2), safe_max(y_expe), safe_max(y_exp3))
            y_limit = min(50000, max(10, y_max_all * 1.1))

            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(-1, y_limit)

            exp2_func_line.set_data(x_current, y_exp2)
            expe_func_line.set_data(x_current, y_expe)
            exp3_func_line.set_data(x_current, y_exp3)

            exp2_tangent.set_data([], [])
            expe_tangent.set_data([], [])
            exp3_tangent.set_data([], [])
            tangent_point2.set_data([], [])
            tangent_pointe.set_data([], [])
            tangent_point3.set_data([], [])

            if cycle_index > 0:
                prev_cycle = cycle_index - 1
                prev_limit = x_limits[prev_cycle]

                prev_mask = (x_full >= -prev_limit) & (x_full <= prev_limit)
                x_prev = x_full[prev_mask]

                if len(x_prev) > 0:
                    y_exp2_prev = 2**x_prev
                    y_expe_prev = np.exp(x_prev)
                    y_exp3_prev = 3**x_prev

                    y_exp2_deriv_prev = np.log(2) * y_exp2_prev
                    y_expe_deriv_prev = y_expe_prev
                    y_exp3_deriv_prev = np.log(3) * y_exp3_prev

                    exp2_deriv_line.set_data(x_prev, y_exp2_deriv_prev)
                    expe_deriv_line.set_data(x_prev, y_expe_deriv_prev)
                    exp3_deriv_line.set_data(x_prev, y_exp3_deriv_prev)
                else:
                    exp2_deriv_line.set_data([], [])
                    expe_deriv_line.set_data([], [])
                    exp3_deriv_line.set_data([], [])
            else:
                exp2_deriv_line.set_data([], [])
                expe_deriv_line.set_data([], [])
                exp3_deriv_line.set_data([], [])

        else:
            tangent_progress = (cycle_progress - growth_fraction) / tangent_fraction

            current_x_limit = x_limits[cycle_index]

            x_min = -current_x_limit
            x_max = current_x_limit

            mask = (x_full >= x_min) & (x_full <= x_max)
            x_current = x_full[mask]

            if len(x_current) > 0:
                y_exp2 = 2**x_current
                y_expe = np.exp(x_current)
                y_exp3 = 3**x_current

                y_exp2_deriv = np.log(2) * y_exp2
                y_expe_deriv = y_expe
                y_exp3_deriv = np.log(3) * y_exp3

                y_max_all = max(safe_max(y_exp2), safe_max(y_expe), safe_max(y_exp3))
                y_limit = min(50000, max(10, y_max_all * 1.1))

                ax2.set_xlim(x_min, x_max)
                ax2.set_ylim(-1, y_limit)

                exp2_func_line.set_data(x_current, y_exp2)
                expe_func_line.set_data(x_current, y_expe)
                exp3_func_line.set_data(x_current, y_exp3)

                if cycle_index == 0:
                    start_x = x_min
                else:
                    start_x = -x_limits[cycle_index - 1]

                end_x = x_max

                tangent_x = start_x + tangent_progress * (end_x - start_x)

                if x_min <= tangent_x <= x_max:
                    tangent_size = 0.5

                    x_tan2, y_tan2, y_point2 = calculate_tangent_line(tangent_x, f_exp2, d_exp2, tangent_size)
                    x_tane, y_tane, y_pointe = calculate_tangent_line(tangent_x, f_expe, d_expe, tangent_size)
                    x_tan3, y_tan3, y_point3 = calculate_tangent_line(tangent_x, f_exp3, d_exp3, tangent_size)

                    exp2_tangent.set_data(x_tan2, y_tan2)
                    expe_tangent.set_data(x_tane, y_tane)
                    exp3_tangent.set_data(x_tan3, y_tan3)

                    tangent_point2.set_data([tangent_x], [y_point2])
                    tangent_pointe.set_data([tangent_x], [y_pointe])
                    tangent_point3.set_data([tangent_x], [y_point3])

                    cycle_tangents[cycle_index]['exp2']['x'].append(tangent_x)
                    cycle_tangents[cycle_index]['exp2']['y'].append(y_point2)
                    cycle_tangents[cycle_index]['expe']['x'].append(tangent_x)
                    cycle_tangents[cycle_index]['expe']['y'].append(y_pointe)
                    cycle_tangents[cycle_index]['exp3']['x'].append(tangent_x)
                    cycle_tangents[cycle_index]['exp3']['y'].append(y_point3)

                    deriv_mask = x_current <= tangent_x

                    if np.any(deriv_mask):
                        exp2_deriv_line.set_data(x_current[deriv_mask], y_exp2_deriv[deriv_mask])
                        expe_deriv_line.set_data(x_current[deriv_mask], y_expe_deriv[deriv_mask])
                        exp3_deriv_line.set_data(x_current[deriv_mask], y_exp3_deriv[deriv_mask])
                    else:
                        exp2_deriv_line.set_data([], [])
                        expe_deriv_line.set_data([], [])
                        exp3_deriv_line.set_data([], [])
                else:
                    exp2_tangent.set_data([], [])
                    expe_tangent.set_data([], [])
                    exp3_tangent.set_data([], [])
                    tangent_point2.set_data([], [])
                    tangent_pointe.set_data([], [])
                    tangent_point3.set_data([], [])

                    if tangent_x > x_max:
                        exp2_deriv_line.set_data(x_current, y_exp2_deriv)
                        expe_deriv_line.set_data(x_current, y_expe_deriv)
                        exp3_deriv_line.set_data(x_current, y_exp3_deriv)
                    else:
                        exp2_deriv_line.set_data([], [])
                        expe_deriv_line.set_data([], [])
                        exp3_deriv_line.set_data([], [])
            else:
                exp2_func_line.set_data([], [])
                expe_func_line.set_data([], [])
                exp3_func_line.set_data([], [])
                exp2_deriv_line.set_data([], [])
                expe_deriv_line.set_data([], [])
                exp3_deriv_line.set_data([], [])
                exp2_tangent.set_data([], [])
                expe_tangent.set_data([], [])
                exp3_tangent.set_data([], [])
                tangent_point2.set_data([], [])
                tangent_pointe.set_data([], [])
                tangent_point3.set_data([], [])

        for cycle, func_name, line in persistent_tangents:
            if cycle < cycle_index or (cycle == cycle_index and not in_growth_phase and tangent_progress > 0.5):
                x_data = cycle_tangents[cycle][func_name]['x']
                y_data = cycle_tangents[cycle][func_name]['y']

                if len(x_data) > 0 and len(y_data) > 0:
                    line.set_data(x_data, y_data)
                else:
                    line.set_data([], [])
            else:
                line.set_data([], [])

        lines_to_return.extend([
            exp2_func_line, expe_func_line, exp3_func_line,
            exp2_deriv_line, expe_deriv_line, exp3_deriv_line,
            exp2_tangent, expe_tangent, exp3_tangent,
            tangent_point2, tangent_pointe, tangent_point3
        ])
        lines_to_return.extend([line for _, _, line in persistent_tangents])

    if DISPLAY_MODE == 3:
        if progress < 0.15:
            current_limit = 0.5 + (progress/0.15) * 1.5
        else:
            adjusted_progress = (progress - 0.15) / 0.85
            current_limit = 2 + adjusted_progress * 8

        x_min_growth = -current_limit
        x_max_growth = current_limit

        mask = (x_full >= x_min_growth) & (x_full <= x_max_growth)
        x_current = x_full[mask]

        if len(x_current) > 0:
            growth_rate1 = 1.0
            growth_rate2 = 2.0

            growth_1 = np.exp(growth_rate1 * x_current)
            growth_2 = np.exp(growth_rate2 * x_current)
            decay_1 = np.exp(-growth_rate1 * x_current)
            decay_2 = np.exp(-growth_rate2 * x_current)

            if progress < 0.15:
                ax3.set_ylim(0, 10)
            else:
                max_growth = max(safe_max(growth_1), safe_max(growth_2))
                y_limit_decay = min(100, max(10, max_growth * 1.1))
                ax3.set_ylim(0, y_limit_decay)

            ax3.set_xlim(x_min_growth, x_max_growth)

            growth_line1.set_data(x_current, growth_1)
            growth_line2.set_data(x_current, growth_2)
            decay_line1.set_data(x_current, decay_1)
            decay_line2.set_data(x_current, decay_2)
        else:
            growth_line1.set_data([], [])
            growth_line2.set_data([], [])
            decay_line1.set_data([], [])
            decay_line2.set_data([], [])

        lines_to_return.extend([
            growth_line1, growth_line2, decay_line1, decay_line2
        ])

    return tuple(lines_to_return)


fps = 24
animation_duration = 25
ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                    interval=1000/fps, blit=True)

fig.suptitle('e ≈ 2.71828...', fontsize=42, color='white', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

animation_filename = "exponential_growth_animation.mp4"
ani.save(animation_filename, writer='ffmpeg', fps=fps, dpi=100, bitrate=12000,
         extra_args=['-pix_fmt', 'yuv420p'], savefig_kwargs={'facecolor': '#121212'})

print(f"Animation saved as {animation_filename}")
