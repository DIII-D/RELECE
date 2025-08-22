import numpy as np
from relece import plasma_engine

# Plasma parameters
density = 2e19  # 1/m^3
temperature = 3000  # eV
magnetic_field = 1.8  # T

# Wave parameters
frequency = 99e9  # Hz
angle = np.pi / 4
mode = 'X'

plasma = plasma_engine.ColdPlasma(density, magnetic_field, temperature)

emission = plasma.emission(frequency, mode, angle)
absorption = plasma.absorption(frequency, mode, angle)

print(emission)
print(absorption)
