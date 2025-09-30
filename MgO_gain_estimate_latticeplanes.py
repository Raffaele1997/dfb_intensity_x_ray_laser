import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import os
import pandas as pd
from panel.theme import Material

import functions
"""
Obtain plots for several materials for all their atom planes to see which ones give the highest gain. 
Output are scatter plots of gain versus wavelength for the respective materials. they are saved as 
"<material>_planes_gain.pdf", and the corresponding tabular text files of [plane, wavelength, gain] are 
saved as "<material>_planes_gain.txt".
"""

material_name = 'MgO'

# read data from the database, clean up, obtain form factors.
materials_data = functions.read_materials_data('data_for_different_materials.txt') #output dictionnary with units of Angstrom
functions.remove_invalid_entries(materials_data)
functions.get_form_factors_local(materials_data, "formfactor_data")

output_dir = f"/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/{material_name}"

# define volume of the probe in nm
L_x = 100 #nm
L_y = 100 #nm
L_z = 100 #nm

L = np.array([L_x, L_y, L_z])





#Lattice Parameter
a = 4.19 #Anstrom
lattice_planes = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2], [4, 0, 0], [3, 3, 1], [4, 2, 0], [4, 2, 2], [5, 1, 1]])

d_spacing = np.zeros(shape = len(lattice_planes))


for i, plane in enumerate(lattice_planes):
    d_spacing[i] = (a/np.linalg.norm(plane))


wavelengths = 2 * d_spacing

gains_planes = np.empty(len(d_spacing))

for i, gi in enumerate(gains_planes):
    materials_data[material_name]['miller_indices'] = lattice_planes[i, :]
    materials_data[material_name]['wavelength'] = wavelengths[i] #Angstrom

    n = functions.find_max_n(materials_data, material_name)
    k = functions.coupling_constant_db(n, wavelengths[i])
    gains_planes[i] = functions.threshold_gain_db(L, k)
    print(r"Gain  = ", gains_planes[i])



# for outputs: convert wavelength to nanometer
wavelengths *= 1e-1

# write to text file:
# Output file name
output_file_mgo = os.path.join(output_dir, f"{material_name}_planes_gain.txt")

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
plt.ylabel('Gain estimate (m$^{-1}$)')
plt.yscale('log')
plt.grid(linewidth = 0.5, color = 'grey')


for i in range(len(wavelengths)):
    if i == 0:
        plt.text(wavelengths[i] - 0.03, gains_planes[i] + 1, lattice_planes[i], fontsize=10)
    else:
        plt.text(wavelengths[i]+0.01, gains_planes[i]+1, lattice_planes[i], fontsize = 10)


plt.title(f"Gain estimate for different lattice planes of {material_name}")

savefig_filename = f"{material_name}/{material_name}_planes_gain.pdf"
plt.savefig(savefig_filename)
plt.show()

df = pd.DataFrame({
    'Wavelengths': wavelengths,
    'Gains Planes': gains_planes,
})

#Refractive index Modulation
delta_n = functions.find_max_n(materials_data, material_name)

#Dataframe erstellen

df['h'] =[lattice_planes[i][0] for i in  range(len(lattice_planes))]
df['k'] =[lattice_planes[i][1] for i in  range(len(lattice_planes))]
df['l'] =[lattice_planes[i][2] for i in  range(len(lattice_planes))]


#Structure Factor
#import csv file with informations about structure factor
mgo_path = r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/MgO/mgo_reflections.csv'
reflections = pd.read_csv(mgo_path)
reflections = reflections.set_index(['h', 'k', 'l'])['|F|']

# adding S(hkl) to df (attention it is defined as F in Vesta....
df['|F|'] = df.set_index(['h', 'k', 'l']).index.map(reflections)

#Adding Refractive index modulation and coupling constant, resonant wavelength, Refractive index modulation for 111 plane. dspacing 0 for distance 111. BUT STILL NOT OTHER PLANES!!
df['Refractive_index_modulation'] = 0.0
df['Coupling_constant'] = 0.0
df['Resonance_Wavelength'] = 0.0

#Looking up n_1(111) from gain_vs_wavelength_list csv file
gain_vs_wavelengths = pd.read_csv(r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/gain_vs_wavelengths.csv')

#Extracting n_1(11) and saving it into df
n_111 = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Refractive_index_modulation"].values[0]
df.loc[0, 'Refractive_index_modulation'] = n_111

#Extracting coupling constant and saving it into df
kappa = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Coupling_constant"].values[0]
df.loc[0, 'Coupling_constant'] = kappa*1E-9

#Extracting resonance wavelength and saving it into df
resonance_ = gain_vs_wavelengths[gain_vs_wavelengths["material"] == material_name]["Resonance_Wavelength"].values[0]
df.loc[0, 'Resonance_Wavelength'] = resonance_

#With n_1/n_1' = |F|/|F'| -> n_1'  = n_1 * |F'|/|F| we can fill up the rest of the values
for i in range(1, len(df['Refractive_index_modulation'])):
    df.loc[i, 'Resonance_Wavelength'] = 2*df.loc[i, 'Wavelengths']/np.sqrt(3)
    df.loc[i, 'Refractive_index_modulation'] = df.loc[0, 'Refractive_index_modulation']*df.loc[i,'|F|']/df.loc[0,'|F|']
    df.loc[i, 'Coupling_constant'] = df.loc[i, 'Refractive_index_modulation']*np.pi/df.loc[i,'Resonance_Wavelength']

print(df)
#storing as csv
df.to_csv('MgO/mgo_planes_gain.csv')


plt.plot(wavelengths, df['|F|'], 'r.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Structure Factor')
#plt.yscale('log')
plt.grid(linewidth = 0.5, color = 'grey')

for i in range(len(wavelengths)):
    plt.text(wavelengths[i], df['|F|'][i], lattice_planes[i], fontsize = 10)
plt.title(f"Gain estimate for different lattice planes of {material_name}")

plt.show()

plt.close()

#inverse S(hkl) or |F|
#plt.title('Inverse S(hkl)  vs Wavelength')
inverse_S = [1/s**2 for s in df['|F|']]
inverse_S[0] = 1/(13.0415**2)
plt.plot(wavelengths, inverse_S, 'r.')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'$(S_{hkl})^{-2}$')
plt.yscale('log')
plt.grid(linewidth = 0.5, color = 'grey')

for i in range(len(wavelengths)):
    plt.text(wavelengths[i]+0.001, inverse_S[i], lattice_planes[i], fontsize = 10)

savefig_filename = r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/MgO/s_hkl_inverse.pdf'
plt.savefig(savefig_filename)
plt.show()


#analysing refractive index from https://henke.lbl.gov/optical_constants/getdb2.html of MgO 0.15 to 0.5 nm in 100 steps
source_refractive = r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/MgO/mgo_wavelength_delta_beta.csv'

df_2 = pd.read_csv(source_refractive)

delta_inv_squared = [1/n**2 for n in df_2['Delta']]

plt.plot(df_2['Wavelength (nm)'], df_2['Delta'], 'r.')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Delta')
plt.grid(linewidth = 0.5, color = 'grey')

plt.show()


plt.plot(df_2['Wavelength (nm)'], delta_inv_squared, 'r.')
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'$\left(n^{-1}\right)^2$')
plt.grid(linewidth = 0.5, color = 'grey')

source_refractive = r'/Users/raffaele/PycharmProjects/Bachelorthesis/Gain_Planes/MgO/refractive_index.pdf'
plt.savefig(source_refractive)


plt.show()
#print(df.head())




