import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import os
import pandas as pd

import functions
"""
Obtain plots for several materials for all their atom planes to see which ones give the highest gain. 
Output are scatter plots of gain versus wavelength for the respective materials. they are saved as 
"<material>_planes_gain.pdf", and the corresponding tabular text files of [plane, wavelength, gain] are 
saved as "<material>_planes_gain.txt".
"""

# read data from the database, clean up, obtain form factors.
materials_data = functions.read_materials_data('data_for_different_materials.txt')
functions.remove_invalid_entries(materials_data)
functions.get_form_factors_local(materials_data, "formfactor_data")

output_dir = r"/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/BaO"

# define volume of the probe (in nm)
L_x = 100
L_y = 100
L_z = 100

L = np.array([L_x, L_y, L_z])




# same procedure as before but for a different material.
material_name = 'Li2O'
a = 4.65 #Anstrom
lattice_planes = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2], [4, 0, 0], [3, 3, 1], [4, 2, 0], [4, 2, 2], [5, 1, 1]])

d_spacing = np.zeros(shape = len(lattice_planes))

for i, plane in enumerate(lattice_planes):
    d_spacing[i] = (a/np.linalg.norm(plane))
wavelengths = 2 * d_spacing

gains_planes = np.empty(len(d_spacing))

for i, gi in enumerate(gains_planes):
    materials_data[material_name]['miller_indices'] = lattice_planes[i, :]
    materials_data[material_name]['wavelength'] = wavelengths[i]

    n = functions.find_max_n(materials_data, material_name)
    k = functions.coupling_constant_db(n, wavelengths[i])
    gains_planes[i] = functions.threshold_gain_db(L, k)



# for outputs: convert wavelength to nanometer
wavelengths *= 1e-1

# write to text file:
# Output file name
output_file_mgo = os.path.join(output_dir, 'Li2O_planes_gain.txt')

# Write to the file
with open(output_file_mgo, 'w') as f:    
    # Write the data
    for plane, lam, gain in zip(lattice_planes, wavelengths, gains_planes):
        # write the header
        f.write(f'# {material_name}: Lattice plane Wavelength Gain \n')

        # write the data
        f.write(f'{plane} {lam} {gain}\n')

plt.plot(wavelengths, gains_planes, 'r.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Gain estimate (nm$^{-1}$)')
plt.yscale('log')
plt.grid(linewidth = 0.5, color = 'grey')


for i in range(len(wavelengths)):
    if i == 0:
        plt.text(wavelengths[i] - 0.03, gains_planes[i] + 2, lattice_planes[i], fontsize=10)
    else:
        plt.text(wavelengths[i]+0.01, gains_planes[i]+2, lattice_planes[i], fontsize = 10)

plt.title('Gain estimate for different lattice planes of Li2O')
plt.savefig('Li2O/Li2O_planes_gain.pdf')
plt.show()

df = pd.DataFrame({
    'Wavelengths': wavelengths,
    'Gains Planes': gains_planes,
})

df['h'] =[lattice_planes[i][0] for i in  range(len(lattice_planes))]
df['k'] =[lattice_planes[i][1] for i in  range(len(lattice_planes))]
df['l'] =[lattice_planes[i][2] for i in  range(len(lattice_planes))]

df['Refractive_index_modulation'] = .0
df['Coupling_constant'] = .0

# Looking up n_1(111) from gain_vs_wavelength_list csv file
gain_vs_wavelengths = pd.read_csv(r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/gain_vs_wavelengths.csv')

# Extracting n_1(11) and saving it into df

n_111 = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Refractive_index_modulation"].values[0]
df.loc[0, 'Refractive_index_modulation'] = n_111

# Extracting coupling constant and saving it into df
kappa = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Coupling_constant"].values[0]
df.loc[0, 'Coupling_constant'] = kappa

# Extracting resonance wavelength and saving it into df
resonance_ = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Resonance_Wavelength"].values[0]
df.loc[0, 'Resonance_Wavelength'] = resonance_


#storing as csv
df.to_csv('Li2O/Li2O_planes_gain.csv')








