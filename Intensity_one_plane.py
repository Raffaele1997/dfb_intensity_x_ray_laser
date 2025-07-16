import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from pathlib import Path



def intensity_calculator(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.lower()

    if 'material' not in df.columns:
        raise KeyError("Missing column 'material' in CSV.")

    for material in df['material'].unique():
        base_path = Path("/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes")
        material_path = base_path / material

        if not material_path.exists():
            print(f"Skipping '{material}' – folder not found.")
            continue

        intensity_dir = material_path / "Intensity"
        intensity_dir.mkdir(exist_ok=True)

        mat = df[df['material'] == material].iloc[0]
        N_1 = mat['refractive_index_modulation']
        KAPPA = mat['coupling_constant']
        WAVELENGTH_0 = mat['resonance_wavelength']
        plane = mat['plane']




        Lengths = np.arange(1, 1000, 10)
        factors = np.linspace(0.8, 1.2, 100)
        wavelengths = factors * WAVELENGTH_0
        delta_lambdas = wavelengths - WAVELENGTH_0  # <- this is the DETUNING axis

        def propagation_constant_low_gain(KAPPA, wavelength):
            delta = N_1 * 2 * np.pi / (wavelength - WAVELENGTH_0)
            return np.sqrt(KAPPA**2 + (-1j * delta)**2)

        def R(z, L, wavelength):
            return np.sinh(propagation_constant_low_gain(KAPPA, wavelength) * (z + 0.5 * L))

        def S(z, L, wavelength):
            return np.sinh(propagation_constant_low_gain(KAPPA, wavelength) * (z - 0.5 * L))

        def E(z, L, wavelength):
            beta_0 = N_1 * 2 * np.pi / wavelength
            return R(z, L, wavelength) * np.exp(-1j * beta_0) + S(z, L, wavelength) * np.exp(1j * beta_0)

        def compute_intensity(factor, z, L):
            wavelength = factor * WAVELENGTH_0
            intensity = np.abs(E(z, L, wavelength))
            return intensity / intensity[0]

        intensity_matrix = np.zeros((len(Lengths), len(delta_lambdas)))

        for i, L in enumerate(Lengths):
            z = np.linspace(-L / 2, L / 2, 1000)
            for j, factor in enumerate(factors):
                intensity_matrix[i, j] = compute_intensity(factor, z, L).max()

        # Normalize intensity_matrix to [0, 1]
        normalized_intensity = (intensity_matrix - intensity_matrix.min()) / (
                    intensity_matrix.max() - intensity_matrix.min())

        # Set small values below threshold to zero
        #threshold = 1e-3  # e.g., zero out values below 0.001 of max
        #normalized_intensity[normalized_intensity < threshold] = 0
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        img = ax.imshow(
            normalized_intensity,
            extent=[delta_lambdas.min(), delta_lambdas.max(), Lengths.min(), Lengths.max()],
            aspect='auto',
            origin='lower',
            cmap='YlOrBr',
            norm=LogNorm(vmin=1e-4, vmax=1), #Here you can adjust v_min e.g. 1e-4 or 1e-2
            interpolation='gaussian'
        )

        cbar = fig.colorbar(img, ax=ax, format=LogFormatter(base=10))
        cbar.set_label(f"Normalized intensity", fontsize=11)

        ax.set_xlabel(fr"Detuning $\Delta \lambda$ from $\lambda_0 = ${WAVELENGTH_0:.3g} nm")
        ax.set_ylabel(r'Amplifying Length $L$ (nm)')
        ax.set_title(f'{plane} Detuning Map – {material}')
        plt.tight_layout()



        # Save everything
        filename_csv = intensity_dir / f"dfb_intensity_map_{material}.csv"
        filename_pdf = intensity_dir / f"dfb_intensity_map_{material}.pdf"

        df_out = pd.DataFrame(intensity_matrix, index=Lengths, columns=delta_lambdas)
        df_out.index.name = 'Length (nm)'
        df_out.columns.name = 'Detuning (nm)'
        df_out.to_csv(filename_csv)

        plt.savefig(filename_pdf)
        plt.close(fig)

    print("Finished detuning maps.")


# Run
source = "/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/gain_vs_wavelengths.csv"
intensity_calculator(source)
