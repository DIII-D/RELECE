import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from relece import plasma_engine

c = constants.c * 1e2  # cm/s
m_e = constants.m_e * 1e3  # g

# Plasma parameters
density = 3e19  # 1/m^3
temperature = 3000  # eV
magnetic_field = 1.66  # T
plasma = plasma_engine.ColdPlasma(density, magnetic_field, temperature)

# Wave parameters
#frequency = 2 * 0.98 * plasma.wce / (2 * np.pi)
frequency = 90e9
angle = np.pi / 4
mode = 'X'

f = plasma.distribution.f
w = 2 * np.pi * frequency
p_bar = plasma.distribution.p
theta = plasma.distribution.theta
n_par = plasma.refractive_index(frequency, mode, angle) * np.cos(angle)
y = plasma.wce / w
harmonic = 2

_, _, p_perp, p_par = plasma._get_resonance_ellipse(n_par, y, theta, harmonic)
p_ellipse = np.hypot(p_perp, p_par)
theta_ellipse = np.arctan2(p_perp, p_par)
gamma_ellipse = np.hypot(1, p_ellipse)
print(gamma_ellipse - n_par * p_par - harmonic * y)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
heatmap = ax.pcolormesh(theta, p_bar, f, shading='auto', cmap='plasma')
cbar = fig.colorbar(heatmap, ax=ax, pad=0.1)
cbar.set_label('f')
ax.plot(theta_ellipse, p_ellipse, color='cyan', lw=2, label='Resonance Ellipse')
ax.set_ylim(0, p_bar.max())
plt.show()

#print("Resonance ellipse: ", plasma.distribution.ev(p_perp, p_par))

emission = plasma.emission(frequency, mode, angle)
# plt.plot(emission)
# plt.show()
# absorption = np.zeros_like(frequencies)
# for i, frequency in enumerate(frequencies):
#     absorption[i] = plasma.absorption(frequency, mode, angle)

print(emission)
#print(absorption)
