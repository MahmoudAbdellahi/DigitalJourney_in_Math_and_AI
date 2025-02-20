import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.io import wavfile
import subprocess
import os
import tempfile
import sys

TOTAL_DURATION_SEC = 15
N_FRAMES = 300
FPS = 60
START_MS = 100
END_MS = 10


def generate_intervals():
    x = np.linspace(0, 1, N_FRAMES)
    intervals = START_MS * np.exp(-3 * x)

    intervals = np.maximum(intervals, END_MS)

    jump_indices = np.arange(30, N_FRAMES, 30)
    for idx in jump_indices:
        jump_amount = np.random.uniform(5, 15)
        intervals[idx:] += jump_amount
        intervals = np.minimum(intervals, START_MS)

    total_ms = np.sum(intervals)
    scale_factor = (TOTAL_DURATION_SEC * 1000) / total_ms
    intervals *= scale_factor

    intervals[-1] = END_MS
    return intervals


def create_click_sound(duration=0.015, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    frequency = 1000
    click = np.sin(2 * np.pi * frequency * t)
    click += 0.5 * np.sin(2 * np.pi * 400 * t)

    decay = np.exp(-10 * t/duration)
    click *= decay

    click = click / np.max(np.abs(click))
    return click


def create_audio_track(intervals, sample_rate=44100):
    total_duration = sum(intervals) / 1000

    n_samples = int(total_duration * sample_rate)
    audio_track = np.zeros(n_samples)

    click = create_click_sound(sample_rate=sample_rate)

    current_sample = 0
    for interval in intervals:
        samples_to_next = int((interval / 1000) * sample_rate)
        end_idx = current_sample + len(click)
        if end_idx <= len(audio_track):
            audio_track[current_sample:end_idx] += click
        current_sample += samples_to_next

    audio_track = audio_track / np.max(np.abs(audio_track))
    audio_track = (audio_track * 32767).astype(np.int16)

    return audio_track, sample_rate, total_duration


plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100, facecolor='#1C1C1C')
plt.subplots_adjust(wspace=0.4)

DICE_BORDER = '#404040'
DICE_FACE = '#2A2A2A'
DOT_COLOR = '#00FF00'
BAR_COLOR = '#4169E1'
THEORETICAL_LINE_COLOR = '#FF4500'

ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_facecolor('#1C1C1C')

plt.rcParams.update({'font.size': 24})
ax2.set_xlim(0.5, 6.5)
ax2.set_ylim(0, 0.5)
ax2.set_xlabel('Dice Value', color='#E0E0E0', fontsize=28)
ax2.set_ylabel('Probability', color='#E0E0E0', fontsize=28)
ax2.set_title('Distribution of Rolls', color='#E0E0E0', fontsize=32)
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.grid(True, alpha=0.2, color='#404040')
ax2.set_facecolor('#1C1C1C')

results = []


def draw_dice(number):
    ax1.clear()
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_facecolor('#1C1C1C')

    outer_square = plt.Rectangle((-1.2, -1.2), 2.4, 2.4, fill=True, color=DICE_BORDER)
    ax1.add_patch(outer_square)

    inner_square = plt.Rectangle((-1, -1), 2, 2, fill=True, color=DICE_FACE)
    ax1.add_patch(inner_square)

    dot_positions = {
        1: [(0, 0)],
        2: [(-0.5, 0.5), (0.5, -0.5)],
        3: [(-0.5, 0.5), (0, 0), (0.5, -0.5)],
        4: [(-0.5, 0.5), (-0.5, -0.5), (0.5, 0.5), (0.5, -0.5)],
        5: [(-0.5, 0.5), (-0.5, -0.5), (0, 0), (0.5, 0.5), (0.5, -0.5)],
        6: [(-0.5, 0.5), (-0.5, 0), (-0.5, -0.5), (0.5, 0.5), (0.5, 0), (0.5, -0.5)]
    }

    for x, y in dot_positions[number]:
        glow = plt.Circle((x, y), 0.25, color=DOT_COLOR, alpha=0.3)
        ax1.add_patch(glow)
        circle = plt.Circle((x, y), 0.2, color=DOT_COLOR)
        ax1.add_patch(circle)

    corner_positions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for x, y in corner_positions:
        circle = plt.Circle((x, y), 0.2, color=DICE_FACE)
        ax1.add_patch(circle)

    ax1.set_aspect('equal')
    ax1.axis('off')
    plt.suptitle(f'Current Roll: {number}', y=1.05, color='#E0E0E0', fontsize=36)


def update_histogram():
    ax2.clear()
    ax2.set_xlim(0.5, 6.5)
    ax2.set_ylim(0, 0.5)
    ax2.set_xlabel('Dice Value', color='#E0E0E0')
    ax2.set_ylabel('Probability', color='#E0E0E0')
    ax2.grid(True, alpha=0.2, color='#404040')
    ax2.set_facecolor('#1C1C1C')

    if results:
        values, counts = np.unique(results, return_counts=True)
        probabilities = counts / len(results)

        bars = ax2.bar(values, probabilities, alpha=0.8, color=BAR_COLOR)

        for bar, prob in zip(bars, probabilities):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{prob:.3f}', ha='center', va='bottom', color='#E0E0E0', fontsize=24)

        ax2.axhline(y=1/6, color=THEORETICAL_LINE_COLOR, linestyle='--',
                    alpha=0.8, label='Theoretical (1/6)', linewidth=2)
        ax2.legend(facecolor='#2A2A2A', edgecolor='#404040', labelcolor='#E0E0E0', fontsize=24)

    ax2.set_title(f'Distribution of Rolls (n={len(results)})', color='#E0E0E0')
    ax2.tick_params(colors='#E0E0E0')


def animate(frame):
    if frame < N_FRAMES:
        number = np.random.randint(1, 7)
        results.append(number)
        draw_dice(number)
        update_histogram()
    return fig,


def save_animation_with_audio(anim, filename, intervals, fps):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video = os.path.join(temp_dir, 'temp_video.mp4')
            temp_audio = os.path.join(temp_dir, 'temp_audio.wav')

            audio_track, sample_rate, total_duration = create_audio_track(intervals)

            actual_fps = N_FRAMES / total_duration

            wavfile.write(temp_audio, sample_rate, audio_track)

            writer = animation.FFMpegWriter(
                fps=actual_fps,
                metadata=dict(artist='Matplotlib'),
                bitrate=3000,
                codec='libx264',
                extra_args=[
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-threads', '0'
                ]
            )

            print(f"Saving video to temporary file: {temp_video}")
            anim.save(temp_video, writer=writer)

            if not os.path.exists(temp_video):
                print(f"Error: Failed to create temporary video file: {temp_video}")
                return

            output_dir = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.abspath(filename)
            print(f"Combining video and audio to: {output_path}")

            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', temp_audio,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                output_path
            ]

            print(f"Running command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print("FFmpeg successfully combined video and audio")
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e}")
                print(f"FFmpeg stderr: {e.stderr}")

                alt_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', temp_audio,
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-pix_fmt', 'yuv420p',
                    '-shortest',
                    output_path
                ]
                print(f"Trying alternative command: {' '.join(alt_cmd)}")
                subprocess.run(alt_cmd, check=True)

    except Exception as e:
        print(f"Error in save_animation_with_audio: {e}")
        import traceback
        traceback.print_exc()


try:
    subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FFmpeg is installed and working")
except (subprocess.SubprocessError, FileNotFoundError):
    print("ERROR: FFmpeg is not installed or not in the PATH. Please install FFmpeg.")
    sys.exit(1)

intervals = generate_intervals()

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES+1,
                               interval=1000/FPS, blit=True)

try:
    print(f"Starting animation save process...")
    save_animation_with_audio(anim, 'dark_uniform_dice_simulation.mp4', intervals, FPS)
    print("Animation has been saved to 'dark_uniform_dice_simulation.mp4' with synchronized audio")
except Exception as e:
    print(f"Failed to save animation: {e}")

    try:
        print("Attempting to save animation without audio as a fallback...")
        writer = animation.FFMpegWriter(
            fps=FPS,
            metadata=dict(artist='Matplotlib'),
            bitrate=3000,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23']
        )
        anim.save('dark_uniform_dice_simulation_noaudio.mp4', writer=writer)
        print("Animation without audio saved as 'dark_uniform_dice_simulation_noaudio.mp4'")
    except Exception as fallback_error:
        print(f"Fallback save also failed: {fallback_error}")
