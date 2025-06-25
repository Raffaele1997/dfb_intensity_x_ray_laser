import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from typing import Union

# =============================================================================
# DFB Laser Simulation
# This script simulates the distributed feedback laser intensity response
# as a function of wavelength detuning and gain region length.
#
# Usage:
# - Provide material and plane parameters as input.
# - Call DFBLaserSimulation.run() to compute the intensity matrix.
# - Use save_and_plot() to export CSV and PNG visualizations.
# =============================================================================

class DFBLaserSimulation:
    def __init__(self, plane_id: str, N_1: float, kappa: float, lambda_0: float) -> None:
        """
        Initialize the DFB laser simulation parameters.

        Parameters:
            plane_id (str): Identifier for the crystal plane (e.g., "111").
            N_1 (float): Amplitude of refractive index modulation.
            kappa (float): Coupling coefficient in 1/nm.
            lambda_0 (float): Bragg wavelength in nm.
        """
        self.plane_id = plane_id
        self.N_1 = N_1
        self.kappa = kappa
        self.lambda_0 = lambda_0

        self.factors = np.linspace(0.8, 1.2, 100)     # Scaling factors for wavelength
        self.lengths = np.arange(1, 1000, 10)         # Gain region lengths [nm]
        self.intensity_matrix = None

    def propagation_constant(self, wave: float) -> complex:
        """
        Calculate the complex propagation constant for a given wavelength.

        Parameters:
            wave (float): Wavelength in nm.

        Returns:
            complex: Complex propagation constant.
        """
        delta = self.N_1 * 2 * np.pi / (wave - self.lambda_0)
        return np.sqrt(self.kappa ** 2 + (-1j * delta) ** 2)

    def E_field(self, z: np.ndarray, L: float, wave: float) -> np.ndarray:
        """
        Calculate the total electric field inside the gain region.

        Parameters:
            z (np.ndarray): Array of spatial coordinates along the laser axis.
            L (float): Gain region length [nm].
            wave (float): Wavelength in nm.

        Returns:
            np.ndarray: Complex electric field distribution.
        """
        beta_0 = self.N_1 * 2 * np.pi / wave
        R = np.sinh(self.propagation_constant(wave) * (z + 0.5 * L))
        S = np.sinh(self.propagation_constant(wave) * (z - 0.5 * L))
        return R * np.exp(-1j * beta_0) + S * np.exp(1j * beta_0)

    def compute_intensity(self, L: float, factor: float) -> float:
        """
        Compute the maximum normalized intensity for a given gain length and wavelength factor.

        Parameters:
            L (float): Gain region length [nm].
            factor (float): Scaling factor applied to the Bragg wavelength.

        Returns:
            float: Maximum normalized field intensity.
        """
        wave = factor * self.lambda_0
        z = np.linspace(-L / 2, L / 2, 3000)
        intensity = np.abs(self.E_field(z, L, wave))
        return float(np.max(intensity / intensity[0])) if intensity[0] != 0 else 0.0

    def run(self) -> None:
        """
        Execute the simulation over the parameter grid and store the normalized intensity matrix.
        """
        self.intensity_matrix = np.zeros((len(self.lengths), len(self.factors)))
        for i, L in enumerate(self.lengths):
            for j, factor in enumerate(self.factors):
                self.intensity_matrix[i, j] = self.compute_intensity(L, factor)

        max_val = np.max(self.intensity_matrix)
        if max_val != 0:
            self.intensity_matrix /= max_val

        self.intensity_matrix[self.intensity_matrix < 1e-4] = 0

    def save_and_plot(self, base_path: str) -> None:
        """
        Save the simulation results to CSV and PNG plot.

        Parameters:
            base_path (str): Directory path to save output files.
        """
        x_dev = self.factors * self.lambda_0 - self.lambda_0
        wavelengths = self.factors * self.lambda_0

        fig, ax = plt.subplots(figsize=(8, 5))
        img = ax.imshow(
            self.intensity_matrix,
            extent=[x_dev.min(), x_dev.max(), self.lengths.min(), self.lengths.max()],
            aspect='auto',
            origin='lower',
            cmap='inferno',
            norm=LogNorm(vmin=0.9e-4, vmax=1),
            interpolation='gaussian'
        )

        ax.axvline(0, linestyle='--', color='gray')
        ax.set_xlabel(f'Detuning (nm) from Bragg Wavelength $\lambda_0=$ {self.lambda_0:.3f} nm')
        ax.set_ylabel('Gain Length $L$ (nm)')
        ax.set_title(f'Intensity Map of DFB Laser, plane {self.plane_id}')

        secax = ax.secondary_xaxis('top', functions=(lambda x: x + self.lambda_0, lambda x: x - self.lambda_0))
        secax.set_xlabel('Absolute Wavelength $\lambda$(nm)')

        cbar = fig.colorbar(img, ax=ax, format=LogFormatter(base=10))
        cbar.set_label('Normalized Intensity')

        plt.tight_layout()
        fig.savefig(f'{base_path}/graph_results/dfb_intensity_map_{self.plane_id}.png', dpi=300)

        df = pd.DataFrame(self.intensity_matrix, index=self.lengths, columns=wavelengths)
        df.index.name = 'Length (nm)'
        df.columns.name = 'Wavelength (nm)'
        df.to_csv(f'{base_path}/csv_results/dfb_intensity_map_{self.plane_id}.csv')

        plt.close(fig)


# =============================================================================
# Example usage (run this part separately or in a main script):
# =============================================================================

# import pandas as pd
# mgo = pd.read_csv("path/to/mgo_planes_gain.csv")
# n_1 = mgo['Refractive_index_modulation']
# kappa = mgo['kappa']
# wavelength = mgo['Wavelengths']
# planes_ = list(zip(mgo['h'], mgo['k'], mgo['l']))
# planes = ["".join(str(x) for x in tpl) for tpl in planes_]
#
# base_path = "./output"
# for det in range(len(planes)):
#     sim = DFBLaserSimulation(planes[det], n_1[det], kappa[det], wavelength[det])
#     sim.run()
#     sim.save_and_plot(base_path)
