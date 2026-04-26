# -*- coding: utf-8 -*-
"""
Vectorized tri banded solver and optional:
    - video f(x,v,t)
    - 9 equally-spaced snapshots 
    - mean voltage
    - activity
    - entropy
    - Fisher information

Created on Thu Apr 23 15:35:04 2026
@author: Nicolas Zadeh
"""

import os
import sys
import time
import shutil
from datetime import datetime

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, PowerNorm

# Parameters

# Initial condition
x10 = 6
v10 = 20
sigma = 1

# Reset
u_F = 9
u_R = 4
sigma_rho = 0.001

# Grid 
x_min = -31
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -40
v_max = 40
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

n = 50

# Electric parameters
b = 1
nu = 10
omega_0 = 0.5
tau = 5

a_0 = 0.5
a_1 = 0.5

# Time parameters
T = 10
Nt = 50001
delta_t = np.float64(T / (Nt - 1))

# Colormap / normalization - need to make big discrepancies in values appear
# without ruining legibility

colormap = "viridis"
powernorm = True
power_gamma = 0.5

# Fonts, make them appear TeX-like

USE_TEX = True

mpl.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Visualization-only optional cosmetic smoothing

USE_GAUSSIAN_SMOOTHING = False
sigma_x_vis = 0.8
sigma_v_vis = 0.8

# Snapshot export parameters

SAVE_SNAPSHOTS = True
NUM_SNAPSHOTS = 9
SNAPSHOT_DPI = 400

# Choice of saved plots

SAVE_ACTIVITY_PLOT = True
SAVE_ENTROPY_PLOT = True
SAVE_FISHER_PLOT = True
SAVE_EXPECTATION_PLOT = True
PLOT_DPI = 400

# Video export parameters

SAVE_VIDEO = True
target_duration = 10.0
fps = 30
VIDEO_DPI = 120

# Completion sounds

USE_SOUND = True

def play_beep(frequency=440.0, duration=0.3):
    if not USE_SOUND:
        return

    if sys.platform.startswith("win"):
        import winsound
        winsound.Beep(int(frequency), int(1000 * duration))
    else:
        # fallback: terminal bell
        print("\a", end="", flush=True)


def play_success_sound():
    play_beep(700, 0.2)
    time.sleep(0.005)
    play_beep(700, 0.1)
    time.sleep(0.05)
    play_beep(900, 0.8)


def play_failure_sound():
    play_beep(300, 0.2)
    time.sleep(0.05)
    play_beep(270,0.2)
    time.sleep(0.05)
    play_beep(240, 0.2)
    time.sleep(0.05)
    play_beep(225, 0.8)

# Output folders

base_dir = os.path.dirname(os.path.abspath(__file__))

results_dir = os.path.join(base_dir, "Results")
videos_dir = os.path.join(results_dir, "Videos")
snapshots_dir = os.path.join(results_dir, "Snapshots")
curves_dir = os.path.join(results_dir, "Curves")

os.makedirs(videos_dir, exist_ok=True)
os.makedirs(snapshots_dir, exist_ok=True)
os.makedirs(curves_dir, exist_ok=True)

# Utilities

def fmt_float_for_filename(x):
    return f"{x:.2e}".replace(".", "p").replace("-", "m").replace("+", "")

def locate_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    candidate_paths = [
        os.path.expanduser(r"~\bin\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def build_equally_spaced_indices(num_frames, num_selected):
    if num_selected <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, num_frames - 1, num_selected, dtype=int)
    return np.unique(idx)

def tex_num(x, digits=2):
    if abs(x) < 5e-15:
        x = 0.0
    return rf"${x:.{digits}f}$"

def tex_relevant_num(x, pos=None):
    if abs(x) < 5e-15:
        x = 0.0
    return rf"${x:g}$"

def tex_relevant_tick_formatter():
    return mticker.FuncFormatter(tex_relevant_num)

def apply_tex_ticks(ax):
    ax.xaxis.set_major_formatter(tex_relevant_tick_formatter())
    ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())

def apply_tex_colorbar_ticks(cbar):
    cbar.ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())
    cbar.update_ticks()

def tex_sci_num(x, pos=None):
    if abs(x) < 5e-15:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10**exp)
    return rf"${mant:.2g}\times 10^{{{exp}}}$"

def tex_sci_tick_formatter():
    return mticker.FuncFormatter(tex_sci_num)

def mass_of(f):
    return delta_x * delta_v * np.sum(f, dtype=np.float64)

def entropy_of(f):
    f_pos = f[f > 0.0]
    return np.float64(
        -delta_x * delta_v * np.sum(f_pos * np.log(f_pos), dtype=np.float64)
    )

def fisher_of(f):
    """
    Discrete Fisher information using the robust identity

        I(f) = 4 \int |\nabla sqrt(f)|^2 dx dv.

    """
    g = np.sqrt(np.maximum(f, 0.0))
    gx = np.gradient(g, delta_x, axis=0)
    gv = np.gradient(g, delta_v, axis=1)

    return np.float64(
        4.0 * delta_x * delta_v * np.sum(gx**2 + gv**2, dtype=np.float64)
    )

def mean_x_of(f):
    return np.float64(delta_x * delta_v * np.sum(x_col * f, dtype=np.float64))

def maybe_smooth(frame):
    if not USE_GAUSSIAN_SMOOTHING:
        return frame
    return gaussian_filter(
        frame,
        sigma=(sigma_x_vis, sigma_v_vis),
        mode="nearest"
    )

# Globals

Nx = Nv = None
delta_x = delta_v = None
dt_over_dx = dt_over_dv = None

x = v = None
x_col = v_row = None

f_initial = rho = None
i_F = j_0 = None

j_pos_x = j_neg_x = None
ab_row = None
J_full = None
x_interior_col = None
v_interior_row = None

# Initialization

def init_xv(n):
    global Nx, Nv, delta_x, delta_v, dt_over_dx, dt_over_dv
    global x, v, x_col, v_row
    global f_initial, rho, i_F, j_0
    global j_pos_x, j_neg_x
    global ab_row, J_full, x_interior_col, v_interior_row

    number_points_space = (n + 1) * size_x + 1
    number_points_velocity = (n + 1) * size_v + 1

    Nx = number_points_space - 2
    Nv = number_points_velocity - 2

    delta_x = np.float64(size_x / (Nx + 1))
    delta_v = np.float64(size_v / (Nv + 1))

    dt_over_dx = delta_t / delta_x
    dt_over_dv = delta_t / delta_v

    x = np.linspace(x_min, x_max, Nx + 2, dtype=np.float64)
    v = np.linspace(v_min, v_max, Nv + 2, dtype=np.float64)

    def index_x(point):
        return int((point - x_min) / delta_x)

    def index_v(point):
        return int((point - v_min) / delta_v)

    i_F = index_x(u_F)
    j_0 = index_v(0)

    x_col = x[:, None]
    v_row = v[None, :]

    x_interior_col = x[1:i_F, None]
    v_interior_row = v[None, :]

    j_pos_x = slice(j_0 + 1, Nv + 2)
    j_neg_x = slice(0, j_0)

    inv2s2_init = 1.0 / (2.0 * sigma**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init)
    f0 = np.multiply.outer(gx_init, gv_init)

    f0[0, j_0:] = 0.0
    f0[i_F, :j_0 + 1] = 0.0

    f_initial = f0 / (delta_x * delta_v * np.sum(f0, dtype=np.float64))
    f_initial *= 1.0 / (delta_x * delta_v * np.sum(f_initial, dtype=np.float64))

    print("Initial mass =", mass_of(f_initial))

    inv2s2_src = 1.0 / (2.0 * sigma_rho**2)
    gx_src = np.exp(-(x - u_R)**2 * inv2s2_src)
    gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src)
    rho0 = np.multiply.outer(gx_src, gv_src)

    rho0[0, j_0:] = 0.0
    rho0[i_F, :j_0 + 1] = 0.0

    rho = rho0 / (delta_x * delta_v * np.sum(rho0, dtype=np.float64))

    print("Source mass =", mass_of(rho))

    ab_row = np.zeros((3, Nv + 2), dtype=np.float64)
    J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

# Implicit matrix in v

def build_ab_row(N):
    alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

    ab_row.fill(0.0)
    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row

# Transport operator

def apply_B_2d(f, N):
    out = f.copy()

    if j_0 + 1 < Nv + 2:
        coeff_pos = (-v[j_pos_x] * dt_over_dx)[None, :]
        out[1:i_F + 1, j_pos_x] += coeff_pos * (
            f[1:i_F + 1, j_pos_x] - f[0:i_F, j_pos_x]
        )

    if j_0 > 0:
        coeff_neg = (-v[j_neg_x] * dt_over_dx)[None, :]
        out[0:i_F, j_neg_x] += coeff_neg * (
            f[1:i_F + 1, j_neg_x] - f[0:i_F, j_neg_x]
        )

    if i_F > 1:
        f_int = f[1:i_F, :]
        out_int = out[1:i_F, :]

        muv = -(omega_0**2) * x_interior_col - v_interior_row / tau + b * (nu + N)
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = ((tau * (-omega_0**2 * x[1:i_F] + b * (nu + N)) + V) / delta_v).astype(np.int64)
        jc = np.clip(jc, -1, Nv + 1)

        crit_mask = (J_full == jc[:, None]) | (J_full == (jc[:, None] + 1))

        out_int += np.where(
            crit_mask & (muv > 0.0),
            -(beta_v + delta_t / tau) * f_int,
            0.0
        )

        out_int += np.where(
            crit_mask & (muv < 0.0),
            (beta_v - delta_t / tau) * f_int,
            0.0
        )

        if Nv >= 1:
            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] > 0.0,
                beta_v[:, 1:Nv + 1] * (f_int[:, 1:Nv + 1] - f_int[:, 0:Nv]),
                0.0
            )

            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] < 0.0,
                beta_v[:, 1:Nv + 1] * (f_int[:, 2:Nv + 2] - f_int[:, 1:Nv + 1]),
                0.0
            )

        mu_bottom = -(omega_0**2) * x[1:i_F] - v[0] / tau + b * (nu + N)
        mu_top = -(omega_0**2) * x[1:i_F] - v[Nv + 1] / tau + b * (nu + N)

        out[1:i_F, 0] += -mu_bottom * dt_over_dv * f[1:i_F, 0]
        out[1:i_F, Nv + 1] += mu_top * dt_over_dv * f[1:i_F, Nv + 1]

    return out

# Implicit solve in v

def solve_A_rowwise(D2, N):
    fnew = np.empty_like(D2)

    fnew[0, :] = D2[0, :]
    fnew[i_F, :] = D2[i_F, :]

    if i_F > 1:
        ab = build_ab_row(N)
        rhs = D2[1:i_F, :].T
        sol = solve_banded((1, 1), ab, rhs)
        fnew[1:i_F, :] = sol.T

    return fnew

# Time step

def compute_activity(f):
    return np.float64(
        delta_v * np.sum(f[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2], dtype=np.float64)
        - delta_v * np.sum(f[0, 0:j_0] * v[0:j_0], dtype=np.float64)
    )

def step(f):
    N = compute_activity(f)
    D2 = apply_B_2d(f, N) + N * delta_t * rho
    fnew = solve_A_rowwise(D2, N)
    return fnew, N

# Plotting helpers

def build_common_norm(vmin, vmax):
    if powernorm:
        return PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)

def make_density_figure(frame2d, time_value, norm):
    fig, ax = plt.subplots(figsize=(6, 4))

    img = ax.imshow(
        frame2d.T,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], v[0], v[-1]],
        cmap=colormap,
        norm=norm,
        interpolation="nearest"
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 7))
    cbar.set_label(r"$f$", rotation=0, labelpad=15)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment("center")
    apply_tex_colorbar_ticks(cbar)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$", rotation=0, labelpad=15)
    ax.set_title(rf"$t = {time_value:g}\,$s")
    
    # To get proper scaling when the values don't get too spread out, comment
    # if necessary
    ax.set_ylim(-size_x/2, size_x/2)

    apply_tex_ticks(ax)
    plt.tight_layout()
    return fig, ax, img

def save_snapshot(frame2d, time_value, idx, outdir, stem, norm):
    fig, ax, img = make_density_figure(frame2d, time_value, norm)

    base = os.path.join(outdir, f"{stem}_snapshot_{idx+1:02d}_t{time_value:.3f}")
    pdf_name = base + ".pdf"
    fig.savefig(pdf_name, bbox_inches="tight")

    plt.close(fig)

def save_curve_plot(t, y, xlabel, ylabel, title, filename_base, sci_y="auto", hide_yticks=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y, linewidth=1.8)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.xaxis.set_major_formatter(tex_relevant_tick_formatter())

    y_finite = np.asarray(y, dtype=np.float64)
    y_finite = y_finite[np.isfinite(y_finite)]

    if y_finite.size == 0:
        use_scientific = False
    elif sci_y is True:
        use_scientific = True
    elif sci_y is False:
        use_scientific = False
    else:
        ymax = np.max(np.abs(y_finite))
        use_scientific = (ymax >= 1e3) or (0 < ymax < 1e-2)

    if use_scientific:
        ax.yaxis.set_major_formatter(tex_sci_tick_formatter())
    else:
        ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())
    
    if hide_yticks:
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both",
                       left=False, right=False, labelleft=False)
    
        text = ax.set_ylabel(ylabel, rotation=0, labelpad=10)
    
        fig = ax.figure
        fig.canvas.draw()
    
        bbox = text.get_window_extent()
        width_in_points = bbox.width * 72.0 / fig.dpi
    
        adaptive_labelpad = 5 + 0.6 * width_in_points
    
        ax.yaxis.labelpad = adaptive_labelpad
    else:
        ax.set_ylabel(ylabel, rotation=0, labelpad=20)

    plt.tight_layout()

    pdf_name = filename_base + ".pdf"
    fig.savefig(pdf_name, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

# Main

try:
    if (1 / tau**2 - 4 * omega_0**2) < 0:
        print("Oscillatory framework")
    else:
        raise RuntimeError("Non-oscillatory framework")

    np.set_printoptions(precision=25)

    start_time = time.time()
    init_xv(n)

    print(f"Nx = {Nx}, Nv = {Nv}, dx = {delta_x}, dv = {delta_v}, dt = {delta_t}")

    # Filename stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = (
        f"f_xv_"
        f"Nx{Nx}_Nv{Nv}_"
        f"T{T}_Nt{Nt}_"
        f"{timestamp}"
    )

    video_filename = os.path.join(videos_dir, file_stem + ".mp4")
    diag_base = os.path.join(curves_dir, file_stem)

    # Precompute selected indices

    snapshot_indices = build_equally_spaced_indices(Nt, NUM_SNAPSHOTS) if SAVE_SNAPSHOTS else np.array([], dtype=int)

    n_video_frames = max(2, int(round(target_duration * fps))) if SAVE_VIDEO else 0
    video_indices = build_equally_spaced_indices(Nt, n_video_frames) if SAVE_VIDEO else np.array([], dtype=int)

    selected_indices = np.unique(np.concatenate([snapshot_indices, video_indices]))
    selected_set = set(int(k) for k in selected_indices)

    print("Snapshot indices =", snapshot_indices.tolist())
    # print("Video indices    =", video_indices.tolist() if SAVE_VIDEO else [])

    # Allocate functionals of interest only

    times = np.linspace(0.0, T, Nt, dtype=np.float64)
    activities = np.empty(Nt, dtype=np.float64)
    masses = np.empty(Nt, dtype=np.float64)
    entropies = np.empty(Nt, dtype=np.float64)
    fishers = np.empty(Nt, dtype=np.float64)
    mean_x_values = np.empty(Nt, dtype=np.float64)

    # Store only selected frames
    stored_frames = {}

    # Initial state

    f = f_initial.copy()
    f_plot = maybe_smooth(f)
    vmax_global = np.max(f_plot)

    activities[0] = compute_activity(f)
    masses[0] = mass_of(f)
    entropies[0] = entropy_of(f)
    fishers[0] = fisher_of(f)
    mean_x_values[0] = mean_x_of(f)

    if 0 in selected_set:
        stored_frames[0] = f_plot.copy()

    # Time loop

    for k in range(1, Nt):
        f, N = step(f)
        f_plot = maybe_smooth(f)

        activities[k] = N
        masses[k] = mass_of(f)
        entropies[k] = entropy_of(f)
        fishers[k] = fisher_of(f)
        mean_x_values[k] = mean_x_of(f)

        frame_max = np.max(f_plot)
        if frame_max > vmax_global:
            vmax_global = frame_max

        if k in selected_set:
            stored_frames[k] = f_plot.copy()

        if (k % 10000 == 0) or (k == Nt - 1):
            print(
                f"k={k}/{Nt-1}, t={times[k]:.4f}, "
                f"N={activities[k]:.6e}, mass={masses[k]:.16f}, "
                f"S={entropies[k]:.16f}, I={fishers[k]:.16f}, "
                f"min(f)={np.min(f):.6e}"
            )

    print(f"Simulation done in {time.time() - start_time:.2f} s")

    # Common norm

    vmin = 0.0
    vmax = vmax_global

    if vmax <= 0:
        raise RuntimeError("All plotted values are non-positive; cannot build a meaningful colormap.")

    common_norm = build_common_norm(vmin=vmin, vmax=vmax)

    # Save snapshots

    if SAVE_SNAPSHOTS:
        print("Snapshot times   =", [float(times[idx]) for idx in snapshot_indices])
        for k_snap, idx in enumerate(snapshot_indices):
            save_snapshot(
                frame2d=stored_frames[int(idx)],
                time_value=times[int(idx)],
                idx=k_snap,
                outdir=snapshots_dir,
                stem=file_stem,
                norm=common_norm
            )

    # Save functionals of interest

    if SAVE_ACTIVITY_PLOT:
        save_curve_plot(
            t=times,
            y=activities,
            xlabel=r"$t$ (s)",
            ylabel=r"$N(t)$" + "\n\n" + r"(Hz)",
            # title=r"Activity",
            title=None,
            filename_base=diag_base + "_activity",
            sci_y="auto"
        )

    if SAVE_ENTROPY_PLOT:
        save_curve_plot(
            t=times,
            y=entropies,
            xlabel=r"$t$ (s)",
            ylabel=r"$S(t)$",
            # title=r"Entropy",
            title=None,
            filename_base=diag_base + "_entropy",
            sci_y=False,
            hide_yticks=True
        )

    if SAVE_FISHER_PLOT:
        save_curve_plot(
            t=times,
            y=fishers,
            xlabel=r"$t$ (s)",
            ylabel=r"$I(f)(t)$",
            # title=r"Fisher information",
            title=None,
            filename_base=diag_base + "_fisher",
            sci_y="auto",
            hide_yticks=True
        )

    if SAVE_EXPECTATION_PLOT:
        save_curve_plot(
            t=times,
            y=mean_x_values,
            xlabel=r"$t$ (s)",
            ylabel=r"$X(t)$" + "\n\n" + r"(volt)",
            # title=r"Mean potential",
            title=None,
            filename_base=diag_base + "_mean_x",
            sci_y=False
        )

    # Save video from stored selected frames only

    if SAVE_VIDEO:
        ffmpeg_path = locate_ffmpeg()

        if ffmpeg_path is None:
            print("\nWARNING: ffmpeg was not found.")
            print("Skipping MP4 export.")
            print("Snapshots and diagnostic plots were still saved.")
        else:
            mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
            print("Using ffmpeg at:", ffmpeg_path)

            frames_video = np.array([stored_frames[int(idx)] for idx in video_indices], dtype=np.float64)
            times_video = np.array([times[int(idx)] for idx in video_indices], dtype=np.float64)

            fig, ax, img = make_density_figure(
                frame2d=frames_video[0],
                time_value=times_video[0],
                norm=common_norm
            )

            def update(frame_idx):
                img.set_data(frames_video[frame_idx].T)
                ax.set_title(rf"$t = {times_video[frame_idx]:g}\,$s")
                return (img,)

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(frames_video),
                interval=1000 / fps,
                blit=False
            )

            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=-1,
                extra_args=[
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                ]
            )

            ani.save(video_filename, writer=writer, dpi=VIDEO_DPI)

            print("Video saved to:", os.path.abspath(video_filename))
            print(f"Chosen fps = {fps}, duration of approximatively {len(frames_video) / fps:.3f} s")

            plt.close(fig)

except Exception as e:
    print("Simulation failed:", e)
    play_failure_sound()
    raise

else:
    print("Simulation completed successfully")
    play_success_sound()