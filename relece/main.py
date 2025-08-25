import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from relece import plasma_engine

c = constants.c / 1e2  # cm/s
m_e = constants.m_e * 1e3  # g

# Plasma parameters
density = 2e19  # 1/m^3
temperature = 3000  # eV
magnetic_field = 1.8  # T

# Wave parameters
frequency = 90e9  # Hz
angle = np.pi / 2
mode = 'X'

plasma = plasma_engine.ColdPlasma(density, magnetic_field, temperature)
f = plasma.distribution.f
w = 2 * np.pi * frequency
p = plasma.distribution.p / (m_e * c)
p /= np.sqrt(1 + p**2)
theta = plasma.distribution.theta
n_par = plasma.refractive_index(frequency, mode, angle) * np.cos(angle)
y = plasma.wce / (2 * np.pi * frequency)
harmonic = 2

_, _, p_perp, p_par = plasma._get_resonance_ellipse(n_par, y, theta, harmonic)
gamma = np.sqrt(1 + (p_perp**2 + p_par**2) / (m_e * c)**2)
print("This should be 0:", 1 - n_par * p_par / (gamma * m_e * c) - harmonic * y / gamma)
p_ellipse = np.sqrt(np.real(p_perp)**2 + np.real(p_par)**2)
print(p_perp)
theta_ellipse = np.arctan2(np.real(p_perp), np.real(p_par))
plt.plot(p_par, p_perp)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
heatmap = ax.pcolormesh(theta, p, f, shading='auto', cmap='plasma')
ax.plot(theta_ellipse, p_ellipse)
plt.show()

#emission = plasma.emission(frequency, mode, angle)
absorption = plasma.absorption(frequency, mode, angle)

#print(emission)
print(absorption)
