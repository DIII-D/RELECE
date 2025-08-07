import tomllib
import numpy as np
from scipy import constants
from relece import utils

with open("config/params.toml", "rb") as f:
    params = tomllib.load(f)

# Generated values
w = 2*np.pi * params['f']
wpe = np.sqrt(params['ne'] * constants.e**2
              / (constants.m_e * constants.epsilon_0))  # Plasma freq (rad/s)
v_th = np.sqrt(params['Te']
               * constants.physical_constants['electron volt-joule relationship'][0]
               / constants.m_e)
gamma_th = 1 / np.sqrt(1 - (v_th / constants.c)**2)  # thermal Lorentz factor
wce = constants.e * params['Bt'] / (gamma_th * constants.m_e)
wr = wce / 2 * (1 + np.sqrt(1 + 4 * wpe**2/wce**2))  # RH cutoff (rad/s)
theta = eval(params['theta'])
n = np.sqrt(utils.refraction(w, wpe, wce, theta, params['x_mode']))
n_par = n * np.cos(theta)
n_perp = n * np.sin(theta)
k = utils.wavevector(n, w, theta)
k_par = k[2]
k_perp = k[0]
