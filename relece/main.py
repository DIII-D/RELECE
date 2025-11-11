import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from relece import plasma_engine

c = constants.c * 1e2  # cm/s
m_e = constants.m_e * 1e3  # g

# Plasma parameters
density = 3e19  # 1/m^3
temperature = 3000  # eV
magnetic_field = 1.61  # T
plasma = plasma_engine.ColdPlasma(density, magnetic_field, temperature)

# Wave parameters
frequencies = np.linspace(0.9, 1, 100, endpoint=False) * 2 * plasma.wce / (2 * np.pi)
#frequency = 2 * 0.98 * plasma.wce / (2 * np.pi)
frequency = 90e9
angle = np.pi / 10
mode = 'X'

f = plasma.distribution.f
w = 2 * np.pi * frequency
p_bar = plasma.distribution.p / (m_e * c)
theta = plasma.distribution.theta
n_par = plasma.refractive_index(frequency, mode, angle) * np.cos(angle)
y = plasma.wce / w
harmonic = 2

_, _, p_perp, p_par = plasma._get_resonance_ellipse(n_par, y, theta, harmonic)
gamma_ellipse = np.sqrt(1 + (p_perp**2 + p_par**2) / (m_e * c)**2)
k_par = n_par * w / c
v_par_ellipse = p_par / (gamma_ellipse * m_e)
resonance_condition = w - k_par * v_par_ellipse - harmonic * plasma.wce / gamma_ellipse
print(resonance_condition)
p_ellipse = np.hypot(p_perp, p_par)
p_bar_ellipse = p_ellipse / (m_e * c)
theta_ellipse = np.arctan2(p_perp, p_par)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
heatmap = ax.pcolormesh(theta, p_bar, f, shading='auto', cmap='plasma')
cbar = fig.colorbar(heatmap, ax=ax, pad=0.1)
cbar.set_label('f')
ax.plot(theta_ellipse, p_bar_ellipse)
plt.show()

#print("Resonance ellipse: ", plasma.distribution.ev(p_perp, p_par))

emission = plasma.emission(frequency, mode, angle)
# absorption = np.zeros_like(frequencies)
# for i, frequency in enumerate(frequencies):
#     absorption[i] = plasma.absorption(frequency, mode, angle)

#print(emission)
#print(absorption)
