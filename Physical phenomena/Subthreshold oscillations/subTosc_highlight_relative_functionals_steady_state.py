# -*- coding: utf-8 -*-
"""
Stationary states and relative functionals

Pass 1:
    run until T_inf to compute f_inf, N_inf.

Pass 2:
    rerun only until T to produce relative functionals plots.

Constraint:
    T <= T_inf.

Relative entropy:
    H_rel(t) = -\int f_inf log(f(t) / f_inf)

Relative Fisher:
    I_rel(t) = 4 \int f_inf |\nabla sqrt(f(t)/f_inf)|²

Only relative functionals and the numerical steady state are stored.

Created on Fri Apr 24 23:10:54 2026
@author: Nicolas Zadeh
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, PowerNorm

# Parameters of the experiment

x10 = 6
v10 = 20
sigma = 1

u_F = 9
u_R = 4
sigma_rho = 0.001

b = 1
nu = 10
omega_0 = 0.5
tau = 5

a_0 = 0.5
a_1 = 0.5

# Construction of the grid

x_min = -31
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -40
v_max = 40
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

n = 50

# Discretization in time

T_inf = 50
T = 10

Nt = 50001
delta_t = np.float64(T / (Nt - 1))

if T > T_inf:
    raise RuntimeError("You must have T <= T_inf.")

Nt_inf = int(round(T_inf / delta_t)) + 1
T_inf_effective = delta_t * (Nt_inf - 1)

# Colormap / normalization

colormap = "viridis"
powernorm = True
power_gamma = 0.5

# Visualization-only optional smoothing

USE_GAUSSIAN_SMOOTHING = False
sigma_x_vis = 0.8
sigma_v_vis = 0.8

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

# Choice of saved plots

SAVE_STEADY_STATE_PLOT = True

SAVE_RELATIVE_ENTROPY_PLOT = True
SAVE_RELATIVE_FISHER_PLOT = True

PLOT_DPI = 400
DENSITY_DPI = 400

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
curves_dir = os.path.join(results_dir, "Curves")
reference_dir = os.path.join(results_dir, "Reference")
steady_state_dir = os.path.join(results_dir, "Steady_state")

os.makedirs(curves_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)
os.makedirs(steady_state_dir, exist_ok=True)

# Utilities

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


def maybe_smooth(frame):
    if not USE_GAUSSIAN_SMOOTHING:
        return frame

    return gaussian_filter(
        frame,
        sigma=(sigma_x_vis, sigma_v_vis),
        mode="nearest"
    )


def build_common_norm(vmin, vmax):
    if powernorm:
        return PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


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

# Functions computing important functionals

def mass_of(f):
    return delta_x * delta_v * np.sum(f, dtype=np.float64)


def entropy_of(f):
    f_pos = f[f > 0.0]
    return np.float64(
        -delta_x * delta_v * np.sum(f_pos * np.log(f_pos), dtype=np.float64)
    )


def fisher_of(f):
    """
    Discrete Fisher information using

        I(f) = 4 \int |\nabla sqrt(f)|² dx dv.
    """
    g = np.sqrt(np.maximum(f, 0.0))
    gx = np.gradient(g, delta_x, axis=0)
    gv = np.gradient(g, delta_v, axis=1)

    return np.float64(
        4.0 * delta_x * delta_v * np.sum(gx**2 + gv**2, dtype=np.float64)
    )


def mean_x_of(f):
    return np.float64(
        delta_x * delta_v * np.sum(x_col * f, dtype=np.float64)
    )


def relative_entropy_of(f, f_inf):
    """
    Numerical version of the relative entropy

        H_rel(t) = \int f_inf log(f_inf / f(t)).

    This expression relies on the fact that, by hypoellipticity,
    f > 0 inside the domain instantly.
    """
    mask = (f_inf > 0.0) & (f > 0.0)

    f_inf_mod = f_inf[mask]
    f_mod = f[mask]

    return np.float64(
        delta_x * delta_v * np.sum(
            f_inf_mod * np.log(f_inf_mod / f_mod),
            dtype=np.float64
        )
    )


def relative_fisher_of(f, f_inf):
    """
    Relative Fisher information:

        I_rel(t) = 4 \int f_inf |\nabla sqrt(f/f_inf)|².
    """
    mask = (f_inf > 0.0) & (f > 0.0)

    ratio = np.ones_like(f, dtype=np.float64)
    ratio[mask] = f[mask] / f_inf[mask]

    weight = np.zeros_like(f, dtype=np.float64)
    weight[mask] = f_inf[mask]

    g = np.sqrt(ratio)

    dg_dx = (g[2:, :] - g[:-2, :]) / (2.0 * delta_x)
    dg_dv = (g[:, 2:] - g[:, :-2]) / (2.0 * delta_v)

    return np.float64(
        4.0 * delta_x * delta_v * np.sum(
            weight[1:-1, 1:-1]
            * (
                dg_dx[:, 1:-1]**2
                + dg_dv[1:-1, :]**2
            ),
            dtype=np.float64
        )
    )


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


# Numerical solver

def build_ab_row(N):
    alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

    ab_row.fill(0.0)
    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row


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

        muv = (
            -(omega_0**2) * x_interior_col
            - v_interior_row / tau
            + b * (nu + N)
        )
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = (
            (
                tau * (-omega_0**2 * x[1:i_F] + b * (nu + N))
                + V
            )
            / delta_v
        ).astype(np.int64)

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
                beta_v[:, 1:Nv + 1]
                * (f_int[:, 1:Nv + 1] - f_int[:, 0:Nv]),
                0.0
            )

            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] < 0.0,
                beta_v[:, 1:Nv + 1]
                * (f_int[:, 2:Nv + 2] - f_int[:, 1:Nv + 1]),
                0.0
            )

        mu_bottom = (
            -(omega_0**2) * x[1:i_F]
            - v[0] / tau
            + b * (nu + N)
        )
        mu_top = (
            -(omega_0**2) * x[1:i_F]
            - v[Nv + 1] / tau
            + b * (nu + N)
        )

        out[1:i_F, 0] += -mu_bottom * dt_over_dv * f[1:i_F, 0]
        out[1:i_F, Nv + 1] += mu_top * dt_over_dv * f[1:i_F, Nv + 1]

    return out


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


def compute_activity(f):
    return np.float64(
        delta_v
        * np.sum(
            f[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2],
            dtype=np.float64
        )
        - delta_v
        * np.sum(
            f[0, 0:j_0] * v[0:j_0],
            dtype=np.float64
        )
    )


def step(f):
    N = compute_activity(f)
    D2 = apply_B_2d(f, N) + N * delta_t * rho
    fnew = solve_A_rowwise(D2, N)
    return fnew, N


# Plotting

def make_density_figure(frame2d, title, norm, colorbar_label=r"$f$"):
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
    cbar.set_label(colorbar_label, rotation=0, labelpad=15)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment("center")
    apply_tex_colorbar_ticks(cbar)
    
    # To get proper scaling when the values don't get too spread out, comment
    # out if necessary, has to be done a posteriori
    ax.set_ylim(-size_x/2,size_x/2)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$", rotation=0, labelpad=15)
    ax.set_title(title)

    apply_tex_ticks(ax)

    plt.tight_layout()
    return fig, ax, img


def save_steady_state_plot(f_inf, N_inf, filename_base):
    f_plot = maybe_smooth(f_inf)

    vmin = 0.0
    vmax = np.max(f_plot)

    if vmax <= 0.0:
        raise RuntimeError(
            "All steady-state plotted values are non-positive; "
            "cannot build a meaningful colormap."
        )

    norm = build_common_norm(vmin=vmin, vmax=vmax)
    Nmant, Nexp = f"{N_inf:.3e}".split("e")
    Nlatex = rf"{Nmant} \times 10^{{{int(Nexp)}}}"
    fig, ax, img = make_density_figure(
        frame2d=f_plot,
        # title=rf"Numerical steady state, $N_\infty = {Nlatex}\,$Hz",
        title=rf"$N_\infty = {Nlatex}\,$Hz",
        norm=norm,
        colorbar_label=r"$f_\infty$"
    )

    pdf_name = filename_base + ".pdf"
    fig.savefig(pdf_name, dpi=DENSITY_DPI, bbox_inches="tight")
    print("Steady-state PDF saved to:", os.path.abspath(pdf_name))
    apply_tex_ticks(ax)

    plt.close(fig)
    
def save_steady_state_3d_plot(f_inf, N_inf, filename_base):

    # Crop in v before plotting, has to be done a posteriori for scaling
    v_min_plot = -size_x / 2
    v_max_plot =  size_x / 2
    v_mask = (v >= v_min_plot) & (v <= v_max_plot)

    v_plot = v[v_mask]
    Z3 = maybe_smooth(f_inf)[:, v_mask]

    X3, V3 = np.meshgrid(x, v_plot, indexing="ij")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X3, V3, Z3,
        cmap=colormap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1
    )

    ax.set_xlabel(r"$x$", labelpad=8)
    ax.set_ylabel(r"$v$", labelpad=8)

    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$f_\infty$", rotation=0, labelpad=8)

    ax.set_xticks([x[0], x[-1]])
    ax.set_xticklabels([
        tex_relevant_num(x[0]),
        r"$u_{\rm F}$"
    ])

    ax.set_yticks([0.0])
    ax.set_yticklabels([
        r"$0$"
    ])

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(v_plot[0], v_plot[-1])

    ax.tick_params(axis="x", pad=0.5)
    ax.tick_params(axis="y", pad=0.5)
    ax.tick_params(axis="z", pad=2)

    try:
        ax.view_init(elev=6, azim=-20, roll=2)
    except TypeError:
        ax.view_init(elev=6, azim=-20)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.15)
    apply_tex_colorbar_ticks(cbar)

    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.08, top=0.95)

    pdf_name = filename_base + "_3D.pdf"
    fig.savefig(pdf_name, dpi=DENSITY_DPI, bbox_inches="tight")

    print("3D steady-state PDF saved to:", os.path.abspath(pdf_name))

    plt.close(fig)


def save_curve_plot(t, y, xlabel, ylabel, title, filename_base, sci_y="auto", hide_yticks=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y, linewidth=1.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, labelpad=15)
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

# Pass 1: compute f_inf := f(t = T_inf)

def compute_reference_state():
    print("Pass 1/2: computing numerical final state f_inf")
    print(f"T_inf requested  = {T_inf}")
    print(f"T_inf effective  = {T_inf_effective}")
    print(f"Nt_inf           = {Nt_inf}")

    f = f_initial.copy()
    N = compute_activity(f)

    for k in range(1, Nt_inf):
        f, N = step(f)

        if (k % 10000 == 0) or (k == Nt_inf - 1):
            print(
                f"[pass 1] k={k}/{Nt_inf-1}, t={k*delta_t:.4f}, "
                f"N={N:.6e}, mass={mass_of(f):.16f}, "
                f"min(f)={np.min(f):.6e}"
            )

    N_inf = compute_activity(f)
    f_inf = f.copy()

    print("Reference computed.")
    print("N_inf =", N_inf)
    print("Mass(f_inf) =", mass_of(f_inf))
    print("Min(f_inf) =", np.min(f_inf))
    print("Max(f_inf) =", np.max(f_inf))

    return f_inf, N_inf


# Pass 2: diagnostics and outputs up to T

def rerun_after_reference(f_inf, N_inf):
    print("Pass 2/2: recomputing trajectory and relative functionals")
    print(f"Plotting only up to t = {T}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = (
        f"f_xv_"
        f"Nx{Nx}_Nv{Nv}_"
        f"T{T}_Tinf{T_inf}_Nt{Nt}_Ntinf{Nt_inf}_"
        f"{timestamp}"
    )

    diag_base = os.path.join(curves_dir, file_stem)
    steady_base = os.path.join(steady_state_dir, file_stem + "_steady_state")

    reference_path = os.path.join(
        reference_dir,
        f"{file_stem}_reference_state.npz"
    )

    np.savez(
        reference_path,
        f_inf=f_inf,
        N_inf=N_inf,
        x=x,
        v=v,
        delta_x=delta_x,
        delta_v=delta_v,
        T=T,
        T_inf=T_inf,
        T_inf_effective=T_inf_effective,
        Nt=Nt,
        Nt_inf=Nt_inf,
        n=n,
        Nx=Nx,
        Nv=Nv,
        u_F=u_F,
        u_R=u_R,
        x_min=x_min,
        x_max=x_max,
        v_min=v_min,
        v_max=v_max,
        b=b,
        nu=nu,
        omega_0=omega_0,
        tau=tau,
        a_0=a_0,
        a_1=a_1,
    )

    print("Reference state saved to:", os.path.abspath(reference_path))

    if SAVE_STEADY_STATE_PLOT:
        save_steady_state_plot(
        f_inf=f_inf,
        N_inf=N_inf,
        filename_base=steady_base
    )

        save_steady_state_3d_plot(
        f_inf=f_inf,
        N_inf=N_inf,
        filename_base=steady_base
    )

    times = np.linspace(0.0, T, Nt, dtype=np.float64)

    relative_entropies = np.empty(Nt, dtype=np.float64)
    relative_fishers = np.empty(Nt, dtype=np.float64)

    f = f_initial.copy()

    relative_entropies[0] = relative_entropy_of(f, f_inf)
    relative_fishers[0] = relative_fisher_of(f, f_inf)

    for k in range(1, Nt):
        f, N = step(f)

        relative_entropies[k] = relative_entropy_of(f, f_inf)
        relative_fishers[k] = relative_fisher_of(f, f_inf)

        if (k % 10000 == 0) or (k == Nt - 1):
            print(
                f"[pass 2] k={k}/{Nt-1}, t={times[k]:.4f}, "
                f"Hrel={relative_entropies[k]:.6e}, "
                f"Irel={relative_fishers[k]:.6e}, "
                f"min(f)={np.min(f):.6e}"
            )

    if SAVE_RELATIVE_ENTROPY_PLOT:
        save_curve_plot(
            t=times,
            y=relative_entropies,
            xlabel=r"$t$ (s)",
            ylabel=r"$\mathcal{H}_{\mathrm{rel}}(t)$",
            title=None,
            # title=r"Relative entropy",
            filename_base=diag_base + "_relative_entropy",
            sci_y="auto",
            hide_yticks=True
        )

    if SAVE_RELATIVE_FISHER_PLOT:
        save_curve_plot(
            t=times,
            y=relative_fishers,
            xlabel=r"$t$ (s)",
            ylabel=r"$\mathcal{I}_{\mathrm{rel}}(t)$",
            title=None,
            # title=r"Relative Fisher information",
            filename_base=diag_base + "_relative_fisher",
            sci_y="auto",
            hide_yticks=True
        )

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
    print(f"T      = {T}")
    print(f"T_inf  = {T_inf}")
    print(f"Nt     = {Nt}")
    print(f"Nt_inf = {Nt_inf}")

    f_inf, N_inf = compute_reference_state()
    rerun_after_reference(f_inf, N_inf)

    print(f"Total execution time = {time.time() - start_time:.2f} s")

except Exception as e:
    print("Simulation failed:", e)
    play_failure_sound()
    raise

else:
    print("Simulation completed successfully")
    play_success_sound()