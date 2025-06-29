import numpy as np

#if your IDE does not show "beautiful" graphs use this:
#----------------------------------------
#import matplotlib
#matplotlib.use('Qt5Agg') #For visualisation
#----------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter


# Import CSV file for MgO or any other material
path  = r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/MgO/mgo_planes_gain.csv'
mgo = pd.read_csv(path)

n_1 = mgo['Refractive_index_modulation']
kappa = mgo['kappa']
wavelength = mgo['Wavelengths']
planes_ = list(zip(mgo['h'], mgo['k'], mgo['l']))
planes = ["".join(str(x) for x in tpl) for tpl in planes_]

for det in range(len(planes)):
    N_1 = n_1[det]  # Amplitude of refractive index modulation
    KAPPA = kappa[det]  # Coupling coefficient [1/nm]
    WAVELENGTH_0 = wavelength[det]  # Central (Bragg) wavelength [nm]

    # --- Simulation Grid --- #
    factors = np.linspace(0.8, 1.2, 100)  # Wavelength scaling factors
    wavelengths = factors * WAVELENGTH_0
    x_dev = wavelengths - WAVELENGTH_0  # deviation from center wavelength
    lengths = np.arange(1, 1000, 10)  # Gain region lengths [nm]

    # --- Coupled-Wave Model --- #
    def propagation_constant_low_gain(kappa_val: float, wave: float):
        # Note: careful with division by zero if wave == WAVELENGTH_0;
        # here original formula uses (wavelength - WAVELENGTH_0) in denominator.
        delta = N_1 * 2 * np.pi / (wave - WAVELENGTH_0)
        return np.sqrt(kappa_val ** 2 + (-1j * delta) ** 2)

    def R(z: float, L: float, wave: float) -> complex:
        return np.sinh(propagation_constant_low_gain(KAPPA, wave) * (z + 0.5 * L))

    def S(z: float, L: float, wave: float) -> complex:
        return np.sinh(propagation_constant_low_gain(KAPPA, wave) * (z - 0.5 * L))

    def E(z, L, wave) -> complex:
        beta_0 = N_1 * 2 * np.pi / wave
        return R(z, L, wave) * np.exp(-1j * beta_0) + S(z, L, wave) * np.exp(1j * beta_0)

    # --- Intensity Computation --- #
    def compute_intensity(L: float, factor: float) -> float:
        wave = factor * WAVELENGTH_0
        # avoid exactly wave == WAVELENGTH_0 if denominator used;
        # in practice factors avoids exactly 1.0 if needed.
        z = np.linspace(-L / 2, L / 2, 1000)
        intensity = np.abs(E(z, L, wave))
        # Normalize at z = -L/2; ensure intensity[0] != 0
        if intensity[0] != 0:
            intensity /= intensity[0]
        return np.max(intensity)

    # --- Compute Normalized Intensity Map --- #
    intensity_matrix = np.zeros((len(lengths), len(factors)))
    for i, L in enumerate(lengths):
        for j, factor in enumerate(factors):
            intensity_matrix[i, j] = compute_intensity(L, factor)

    # Normalize the entire matrix to its global maximum
    max_val = intensity_matrix.max()
    if max_val != 0:
        intensity_matrix /= max_val

    # Set small values below threshold to zero
    threshold = 1e-4  # e.g., zero out values below 0.001 of max
    intensity_matrix[intensity_matrix < threshold] = 0

    # --- Visualization --- #
    fig, ax = plt.subplots(figsize=(8, 5))
    img = ax.imshow(
        intensity_matrix,
        extent=[x_dev.min(), x_dev.max(), lengths.min(), lengths.max()],
        aspect='auto',
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=0.9*threshold, vmax=1),
        interpolation='gaussian'
    )

    # Vertical line at deviation = 0 (at WAVELENGTH_0)
    ax.axvline(0, linestyle='--', color='gray')

    # Axis labels and title
    ax.set_xlabel(f'Detuning (nm) from Bragg Wavelength $λ_0=$ {WAVELENGTH_0:.3f} nm')
    ax.set_ylabel(r'Gain Length $L$ ({0:s})'.format("nm"))
    ax.set_title('Intensity Map of DFB Laser, plane ' + planes[det])

    # Secondary x-axis to show absolute wavelength
    def dev_to_abs(x_dev_val):
        return x_dev_val + WAVELENGTH_0
    def abs_to_dev(x_abs_val):
        return x_abs_val - WAVELENGTH_0

    secax = ax.secondary_xaxis('top', functions=(dev_to_abs, abs_to_dev))
    secax.set_xlabel('Absolute Wavelength $λ$(nm)')

    # Colorbar with log formatter
    cbar = fig.colorbar(img, ax=ax, format=LogFormatter(base=10))
    cbar.set_label('Normalized Intensity')

    plt.tight_layout()

    # Save or show
    # Show interactively
    plt.show()

    # --- Save to CSV and PNG --- #
    # DataFrame with index=lengths, columns=wavelengths (absolute)
    df = pd.DataFrame(intensity_matrix, index=lengths, columns=wavelengths)
    df.index.name = 'Length (nm)'
    df.columns.name = 'Wavelength (nm)'

    # Filenames
    png_path = f'/Users/raffaele/PycharmProjects/Bachelorthesis/Intensity Map/graph_results/dfb_intensity_map_{planes[det]}.png'
    csv_path = f'/Users/raffaele/PycharmProjects/Bachelorthesis/Intensity Map/csv_results/dfb_intensity_map_{planes[det]}.csv'

    # Save figure and CSV
    fig.savefig(png_path, dpi=300, format='png')
    df.to_csv(csv_path)

    plt.close(fig)


