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


def create_probability_animations():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.tight_layout(pad=4.0)

    x = np.linspace(120, 210, 400)

    mu, sigma = 165, 10
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)

    ax1.plot(x, pdf, color='#00ff88', label='PDF')
    ax2.plot(x, cdf, color='#00ffff', label='CDF')

    ax1.set_title('Probability Density Function (PDF)', color='white', pad=20)
    ax2.set_title('Cumulative Distribution Function (CDF)', color='white', pad=20)
    ax1.set_ylabel('Density', color='white')
    ax2.set_ylabel('Probability', color='white')
    ax2.set_xlabel('x', color='white')

    ax1.set_ylim(0, max(pdf) * 1.1)
    ax2.set_ylim(0, 1.1)

    line1, = ax1.plot([], [], color='#ff3366', label='Current x')
    point_on_cdf, = ax2.plot([], [], 'o', color='#ff3366',
                             markersize=8, label='Accumulated Probability')

    poly = Polygon([[x[0], 0]], color='#ff3366', alpha=0.3)
    ax1.add_patch(poly)

    area_text = ax1.text(0.5, 0.5, '', color='white', fontsize=24,
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)

    def init():
        line1.set_data([], [])
        point_on_cdf.set_data([], [])
        poly.set_xy([[x[0], 0]])
        area_text.set_text('')
        return line1, point_on_cdf, poly, area_text

    def animate(i):
        current_x = x[i]

        line1.set_data([current_x, current_x], [0, pdf[i]])
        point_on_cdf.set_data([current_x], [cdf[i]])

        vertices = np.vstack(([x[0], 0],
                              np.column_stack((x[:i+1], pdf[:i+1])),
                              [x[i], 0]))
        poly.set_xy(vertices)

        current_area = cdf[i]
        area_text.set_text(f'Area: {current_area:.3f}')

        return line1, point_on_cdf, poly, area_text

    ax1.legend()
    ax2.legend()

    plt.figtext(0.02, 0.02, 'The shaded area under the PDF equals the CDF value',
                color='white', fontsize=10,
                bbox=dict(facecolor='black', edgecolor='#666666', alpha=0.8))

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(x), interval=100, blit=True)

    anim.save('probability_4k.mp4',
              writer='ffmpeg',
              fps=60,
              dpi=300,
              bitrate=8000,
              extra_args=['-vcodec', 'libx264', '-crf', '17'])

    plt.close()


if __name__ == "__main__":
    create_probability_animations()
