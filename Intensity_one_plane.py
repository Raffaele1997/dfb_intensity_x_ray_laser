import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.widgets import Slider, TextBox
from scipy.signal import find_peaks
#import pandas as pd


N_1 = 4.43e-5
KAPPA = 9.96e-5 # nm^-1
WAVELENGTH_0 = 0.445 # nm
L = 1 # nm
initial_factor = 0.9999999

""""
# Physical and simulation parameters
N_1 = 4.43e-5
KAPPA = 9.96e-5 # nm^-1
WAVELENGTH_0 = 0.445 # nm
"""
# List of Length between 1 nm and 1000 nm and 10 nm steps Y axis
Lengths = np.arange(1, 1000, 10) # nm

#initial_factor = 0.999999999


division_factor = 2 #Change


# Functions for calculations

def propagation_constant_low_gain(KAPPA, WAVELENGTH):
    delta = N_1 * 2 * np.pi / (WAVELENGTH - WAVELENGTH_0)
    return np.sqrt(KAPPA**2 + (-1j * delta)**2)
"""Wrong one

def propagation_constant_low_gain(KAPPA, WAVELENGTH):
    delta = (N_1 * 2 * np.pi / WAVELENGTH) - (N_1 * 2 * np.pi / (WAVELENGTH - WAVELENGTH_0))
    return np.sqrt(KAPPA**2 + (-1j * delta)**2)"""


def R(z, L, WAVELENGTH):
    return np.sinh(propagation_constant_low_gain(KAPPA, WAVELENGTH)*(z + 0.5 * L))

def S(z, L, WAVELENGTH):
    return np.sinh(propagation_constant_low_gain(KAPPA, WAVELENGTH)*(z - 0.5 * L))

def E(z, L, WAVELENGTH):
    beta_0 = N_1*2*np.pi/WAVELENGTH
    return R(z, L, WAVELENGTH)*np.exp(-1j*beta_0)+S(z,L,WAVELENGTH)*np.exp(1j*beta_0)





# Compute intensity normalized at z = -L/division_factor
def compute_intensity(factor):
    WAVELENGTH = factor * WAVELENGTH_0
    intensity = np.abs(E(z, L, WAVELENGTH))
    normalization_factor = intensity[0] #at z = -L/2
    intensity /= normalization_factor
    return intensity


# -------------------------------------------------------------------------------------- #
# Factors for wavelength variation X-Axis
min_factor = 0.8
max_factor = 1.2
factors = np.linspace(min_factor,max_factor, 100)
# Compute intensity for all Lengths

#Creating Matrix with Length (Zeile) and Intensities (Spalte)
m = len(Lengths)
n = len(factors)

length_intensity_matrix = np.zeros((m,n))


#Filling up matrix with corresponding length
for m,L in enumerate(Lengths):
    z = np.linspace(-L/2, L/2, 1000)
    for n,factor in enumerate(factors):
        intensity = max(compute_intensity(factor))
        length_intensity_matrix[m][n] = intensity


fig, ax = plt.subplots()
# Plot with LogNorm
img = ax.imshow(
    length_intensity_matrix,
    extent=[
        factors.min() * WAVELENGTH_0,
        factors.max() * WAVELENGTH_0,
        Lengths.min(),
        Lengths.max()
    ],
    aspect='auto',
    origin='lower',
    cmap='YlOrBr',
    norm=LogNorm(vmin=length_intensity_matrix.min(),
                 vmax=length_intensity_matrix.max()),
    interpolation='gaussian'
)

# Colorbar with log ticks
cbar = fig.colorbar(img, ax=ax, format=LogFormatter(base=10, labelOnlyBase=False))
cbar.set_label(r'Normalized Intensity', fontsize=12)

ax.set_xlabel(r'Wavelength $\lambda$')
ax.set_ylabel('Amplyfing Length')
ax.set_title('Log-normalized Intensity Map')

plt.tight_layout()
plt.show()
#print(length_intensity_matrix)


# Convert to DataFrame and save as CSV
df = pd.DataFrame(length_intensity_matrix, index=Lengths, columns=(factors * WAVELENGTH_0))
df.index.name = 'Length (nm)'
df.columns.name = 'Wavelength (nm)'
df.to_csv('dfb_intensity_map.csv')








