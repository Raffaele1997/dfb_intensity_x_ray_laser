import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

# --- Physical Constants --- #
N_1 = 4e-5            # Amplitude of refractive index modulation
KAPPA = 9.96e-5          # Coupling coefficient [1/nm]
WAVELENGTH_0 = 0.445     # Central (Bragg) wavelength [nm]
division_factor = 2      # Used for defining normalization point

# --- Simulation Grid --- #
factors = np.linspace(0.8, 1.2, 100)                 # Wavelength scaling factors
wavelengths = factors * WAVELENGTH_0
lengths = np.arange(1, 1000, 10)                     # Gain region lengths [nm]

# --- Coupled-Wave Model --- #
def propagation_constant_low_gain(kappa: float, wavelength:float):
    delta = N_1 * 2 * np.pi / (wavelength - WAVELENGTH_0)
    return np.sqrt(kappa**2 + (-1j * delta)**2)

def R(z:float, L:float, wavelength:float)->float:
    return np.sinh(propagation_constant_low_gain(KAPPA, wavelength) * (z + 0.5 * L))

def S(z:float, L:float, wavelength:float)->float:
    return np.sinh(propagation_constant_low_gain(KAPPA, wavelength) * (z - 0.5 * L))

def E(z, L, wavelength)->float:
    beta_0 = N_1 * 2 * np.pi / wavelength
    return R(z, L, wavelength) * np.exp(-1j * beta_0) + S(z, L, wavelength) * np.exp(1j * beta_0)

# --- Intensity Computation --- #
def compute_intensity(L:float, factor:float)->float:
    wavelength = factor * WAVELENGTH_0
    z = np.linspace(-L/2, L/2, 1000)
    intensity = np.abs(E(z, L, wavelength))
    intensity /= intensity[0]  # Normalize at z = -L/2
    return np.max(intensity)

# --- Compute Normalized Intensity Map --- #
intensity_matrix = np.zeros((len(lengths), len(factors)))

for i, L in enumerate(lengths):
    for j, factor in enumerate(factors):
        intensity_matrix[i, j] = compute_intensity(L, factor)

# Normalize the entire matrix to its global maximum
intensity_matrix /= intensity_matrix.max()

# Set small values below threshold to zero
threshold = 0.005  # All values < 1% of max will be zeroed
intensity_matrix[intensity_matrix < threshold] = 0

# --- Visualization --- #
fig, ax = plt.subplots()
img = ax.imshow(
    intensity_matrix,
    extent=[wavelengths.min(), wavelengths.max(), lengths.min(), lengths.max()],
    aspect='auto',
    origin='lower',
    cmap='BuGn',
    norm=LogNorm(vmin=1e-2, vmax=1),  # Adjust vmin for better contrast
    interpolation='gaussian'
)

# Colorbar
cbar = fig.colorbar(img, ax=ax, format=LogFormatter(base=10))
cbar.set_label('Normalized Intensity (max = 1)', fontsize=12)

# Axis labels and title
ax.set_xlabel('Wavelength λ (nm)')
ax.set_ylabel('Gain Length $L$ (nm$^{-1}$)')
ax.set_title('Log-Normalized Intensity Map of DFB Laser')

plt.tight_layout()
plt.show()

# --- Save to CSV --- #
df = pd.DataFrame(intensity_matrix, index=lengths, columns=wavelengths)
df.index.name = 'Length (nm)'
df.columns.name = 'Wavelength (nm)'
df.to_csv('dfb_intensity_map.csv')
