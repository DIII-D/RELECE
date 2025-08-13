import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from relece.cold_plasma import refraction
from relece.ray_refraction import ray_refraction


def produce_refraction_data(theta, w, alpha2, beta2, ray=False):
    """
    Generates two arrays of data for polar plotting based on input parameters.

    Parameters
    ----------
    theta : np.ndarray
        Array of angles, typically from 0 to 2*pi.
    w : float
        Wave frequency (rad/s).
    alpha2 : float
        Square of ratio between plasma frequency and wave frequency.
    beta2 : float
        Square of ratio between cyclotron frequency and wave frequency.
    ray : bool
        Whether to calculate n or nr.

    Returns
    -------
    ro : np.ndarray
        1/n(r) array for O mode.
    rx : np.ndarray
        1/n(r) array for X mode.

    """
    wpe = np.sqrt(alpha2) * w
    wce = np.sqrt(beta2) * w

    no = refraction(w, wpe, wce, theta, x_mode=False)
    nx = refraction(w, wpe, wce, theta, x_mode=True)

    if ray:
        ro = 1 / ray_refraction(no, w, wpe, wce, theta)
        rx = 1 / ray_refraction(nx, w, wpe, wce, theta)
    else:
        ro = 1 / no
        rx = 1 / nx

    return ro, rx


"""
Creates a figure with 8 subplots, each containing two polar plots.
"""
theta = np.linspace(0, 2 * np.pi, 501,
                    endpoint=False)[1:]  # Avoid singularities at 0
w = 1.0  # Relative frequency for demonstration

# Define 8 pairs of (alpha2, beta2) parameters for the 8 subplots
# These values are chosen to show a variety of shapes.
# params = [
#     (1/3, 3/4), (4/9, 1), (2/3, 3/2), (1, 9/4-1e-3)
# ]
params = [
    (2/9, 1/2), (1/4, 9/16), (4/15, 3/5), (4/13, 9/13)
]

# Create a figure and a 1x4 grid of subplots.
fig, axes = plt.subplots(1, 4, figsize=(
    16, 4), subplot_kw={'projection': 'polar'})
axes_flat = axes.ravel()

for i, ax in enumerate(axes_flat):
    alpha2, beta2 = params[i]

    ro, rx = produce_refraction_data(theta, w, alpha2, beta2, ray=True)
    ax.plot(theta, ro, label='O mode', color='b')
    # ax.plot(theta, rx, label='X mode', color='r', linestyle='--')

    ax.set_title(
        f'$\\alpha^2={str(Fraction(alpha2).limit_denominator())}$; '
        f'$\\beta^2={str(Fraction(beta2).limit_denominator())}$',
        va='bottom',
        pad=15
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_rticks([1])
    # ax.set_rlim(0, 1.5)
    ax.grid(True)

# Add a single legend for the entire figure
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',
           ncol=2, bbox_to_anchor=(0.5, -0.02))

# Adjust layout to prevent titles and labels from overlapping
# Adjust rect to make space for the legend
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
