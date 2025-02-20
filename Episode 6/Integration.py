import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

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

fig, ax = plt.subplots(figsize=(12, 8))


def f(x):
    return x**2


x_curve = np.linspace(0, 3, 400)
y_curve = f(x_curve)


def update(frame):
    ax.clear()

    ax.plot(x_curve, y_curve, color='#00ff00', label='y = x²', linewidth=2)

    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.1, 9.1)
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('y', color='white')
    ax.set_title('Riemann Sum Approximation\nWatch dx get smaller!',
                 color='white', pad=20)

    n_rectangles = 2 + frame * 2
    dx = 3.0 / n_rectangles
    x_left = np.linspace(0, 3-dx, n_rectangles)

    total_area = 0

    for x in x_left:
        height = f(x)
        rect = plt.Rectangle((x, 0), dx, height,
                             facecolor='none',
                             edgecolor='#00ffff',
                             linewidth=2)
        ax.add_patch(rect)
        total_area += height * dx

    info_text = f'Number of rectangles: {n_rectangles}\nΔx = {dx:.4f}\nApprox Area = {total_area:.4f}'
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='black',
                      edgecolor='#666666',
                      alpha=0.8))

    return ax,


total_frames = 200
frame_interval = 100

anim = FuncAnimation(fig, update,
                     frames=total_frames,
                     interval=frame_interval,
                     repeat=True)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'riemann_sum_4k.mp4')
anim.save(output_path,
          writer='ffmpeg',
          fps=30,
          dpi=300,
          bitrate=8000,
          extra_args=['-vcodec', 'libx264', '-crf', '17'])

plt.close()
