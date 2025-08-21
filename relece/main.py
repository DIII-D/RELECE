import numpy as np
from . import plasma as pl

# Plasma parameters
density = 2e13  # 1/cm^3
temperature = 3000  # eV
magnetic_field = 1.8  # T

# Wave parameters
frequency = 99e9  # Hz
angle = np.pi / 2
mode = 'X'

plasma = pl.ColdPlasma(density, magnetic_field, temperature)

emission = plasma.emission(frequency, mode, angle)
absorption = plasma.absorption(frequency, mode, angle)

print(emission)
print(absorption)
