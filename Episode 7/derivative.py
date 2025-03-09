import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy.stats import norm

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.antialiased': True,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'axes.facecolor': 'black',
    'figure.facecolor': 'black',
    'axes.edgecolor': '#666666',
    'grid.color': '#666666',
})


def create_derivative_animation():
    fig, ax = plt.subplots(figsize=(12, 8))

    def f(x):
        return x**2 + 1

    x = np.linspace(-2, 8, 400)
    y = f(x)

    ax.plot(x, y, color='#00ff88', label='f(x) = x² + 1')

    x_point = 2
    h_initial = 3
    h_final = 0.1
    num_frames = 1000

    h_values = np.geomspace(h_initial, h_final, num_frames)

    secant_line, = ax.plot([], [], color='#ff3366', linestyle='--', label='Secant Line')
    point_a, = ax.plot([], [], 'o', color='#00ffff', markersize=8, label='Point A')
    point_b, = ax.plot([], [], 'o', color='#00ffff', markersize=8, label='Point B')

    text_box_props = dict(boxstyle='round', facecolor='black', edgecolor='#666666', alpha=0.8)
    slope_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                         horizontalalignment='right',
                         verticalalignment='top',
                         color='white',
                         bbox=text_box_props)

    point_a_text = ax.text(0, 0, '', fontsize=10, color='white')
    point_b_text = ax.text(0, 0, '', fontsize=10, color='white')

    def init():
        secant_line.set_data([], [])
        point_a.set_data([], [])
        point_b.set_data([], [])
        point_a_text.set_text('')
        point_b_text.set_text('')
        return secant_line, point_a, point_b, slope_text, point_a_text, point_b_text

    def animate(frame):
        h = h_values[frame]

        x_a = x_point
        x_b = x_point + h
        y_a = f(x_a)
        y_b = f(x_b)

        slope = (y_b - y_a) / h

        x_line = np.array([x_a - 1, x_b + 1])
        y_line = slope * (x_line - x_a) + y_a

        secant_line.set_data(x_line, y_line)
        point_a.set_data([x_a], [y_a])
        point_b.set_data([x_b], [y_b])

        slope_text.set_text(f'h = {h:.3f}\nSlope = {slope:.3f}\n[f(x+h) - f(x)]/h = {slope:.3f}')

        point_a_text.set_position((x_a - 0.3, y_a + 2))
        point_a_text.set_text(f'A (x, f(x))\n({x_a}, {y_a:.1f})')

        point_b_text.set_position((x_b + 0.2, y_b + 2))
        point_b_text.set_text(f'B (x+h, f(x+h))\n({x_b:.1f}, {y_b:.1f})')

        return secant_line, point_a, point_b, slope_text, point_a_text, point_b_text

    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 70)
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('f(x)', color='white')
    ax.set_title('Limit Definition of Derivative\nAs h → 0, secant line approaches tangent line',
                 color='white', pad=20)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right')

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                         interval=400, blit=True)

    anim.save('derivative_4k.mp4',
              writer='ffmpeg',
              fps=60,
              dpi=300,
              bitrate=8000,
              extra_args=['-vcodec', 'libx264', '-crf', '17'])

    plt.close()


if __name__ == "__main__":
    create_derivative_animation()
