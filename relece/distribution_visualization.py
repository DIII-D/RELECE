import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from matplotlib import colormaps
from relece.maxwellian import relativistic_maxwellian_distribution


vgrid, tgrid, f = relativistic_maxwellian_distribution(3000, enorm=10)
xgrid = vgrid * np.cos(tgrid) / c
ygrid = vgrid * np.sin(tgrid) / c

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xgrid, ygrid, f, cmap=colormaps['viridis'], rstride=1, cstride=1)

plt.show()
