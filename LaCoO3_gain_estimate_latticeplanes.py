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
print(type(materials_data))
functions.remove_invalid_entries(materials_data)
functions.get_form_factors_local(materials_data, "formfactor_data")

output_dir = r"/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/LaCoO3"

# define volume of the probe in nm
L_x = 100
L_y = 100
L_z = 100

L = np.array([L_x, L_y, L_z])




# same procedure as before but for a different material.
material_name = 'LaCoO3'
a = 3.82 #Anstrom
lattice_planes = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [2, 1, 0],
    [2, 1, 1],
    [2, 2, 0],
    [2, 2, 1],
    [2, 2, 2],
    [3, 1, 0],
    [3, 1, 1],
    [3, 2, 1],
    [3, 2, 2],
    [3, 3, 1],
    [4, 1, 1],
    [4, 2, 0]
])

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
    print('Gain = ',gains_planes[i], ' Plane = ', lattice_planes[i])




# for outputs: convert wavelength to nanometer
wavelengths *= 1e-1

# write to text file:
# Output file name
output_file_mgo = os.path.join(output_dir, 'LaCoO3_planes_gain.txt')

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
#plt.yscale('log')
plt.grid(linewidth = 0.5, color = 'grey')


for i in range(len(wavelengths)):
    if i == 0:
        plt.text(wavelengths[i] - 0.03, gains_planes[i] + 1, lattice_planes[i], fontsize=10)
    else:
        plt.text(wavelengths[i]+0.01, gains_planes[i]+1, lattice_planes[i], fontsize = 10)
plt.title(r'Gain estimate for different lattice planes of La$_0.5$Sr$_0.5$CoO3')
plt.savefig('LaCoO3/LaCoO3_planes_gain.pdf')
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


#Saving it into gain_vs_wavelength.csv
#df = pd.read_csv(r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/gain_vs_wavelengths.csv')

# Calculating refractive index using refractive_index_db(materials_data, material_name, ksi) function
#n_111 = np.abs(functions.refractive_index_db(materials_data, material_name, 4.41E-10 ))

# Adding resonance wavelength for 111 plane and saving it into df
#lambd = (2*a/np.sqrt(3))*1E-1 #a in nm therefore *1E-1

#Adding coupling constant using coupling_constant_db(n, lambd)
#kappa = functions.coupling_constant_db(n_111, lambd)

#Gain_estimate
#gain = functions.threshold_gain_db(L, kappa)


#storing as csv
df.to_csv('LaCoO3/LaCoO3_planes_gain.csv')
"""
#Adding LaCoO3
new_row = {
    "material": "LaCoO3",
    "Refractive_index_modulation": n_111,
    "Coupling_constant": kappa,
    "Gain_estimate": gain ,
    "Resonance_Wavelength": lambd
}
df.loc[len(df)] = new_row

#saving df again as csv
df.to_csv(r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/gain_vs_wavelengths.csv')
#storing as csv
df.to_csv('LaCoO3/LaCoO3_planes_gain.csv', index = False)

"""






