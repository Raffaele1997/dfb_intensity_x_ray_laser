import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as const 
from scipy.optimize import curve_fit 
import uncertainties.unumpy as unp 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import functools 
from io import StringIO
from urllib.parse import urljoin
from collections import Counter
import os

#imports Raffaele
from scipy.optimize import root


"""
To calculate material properties in the context of an x-ray lasing system making use of the coupling of Bragg reflected light within a crystal.

"""

###############################################################################
### to calculate atomic form factor and structure factor

# not needed in code as it is now, but works if you have the fit parameters for the respective element.
def atomic_form_factor(element, params, q_magnitude):
    """
    Calculate the atomic form factor for a given element and scattering vector.

    Parameters:
    element (str): chemical symbol of the element to be calculated
    params (list or something): library of constants needed to calculate the form factor, should contain every element you need
    q_magnitude (float): magnitude of reciprocal lattice vector

    Returns:
    Atomic form factor f.
    """

    # raise Error if the element is invalid
    if element not in params:
        raise ValueError(f"Element {element} is not found in the parameter library.")

    # Retrieve constants for given element
    a, b, c = params[element]

    # using these parameters, calculate the form factor
    f = sum(ai * np.exp(-bi * (q_magnitude / (4 * np.pi))**2) for ai, bi in zip(a, b)) + c

    return f

def calculate_q_vector(miller, unitcell_lengths, unitcell_angles):
    """
    Calculate the scattering vector from Miller indices and the unit cell parameters in order to use it in further calculations.

    Parameters:
    miller (float array): [h, k, l], Miller indices to be considered
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): [alpha, beta, gamma], unit cell angles in degrees

    Returns:
    q vector.
    """
    h, k, l = miller 
    a, b, c = unitcell_lengths
    alpha, beta, gamma = np.radians(unitcell_angles)

    # calculate reciprocal lattice volume
    V = a * b * c * np.sqrt(
        1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
        2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    a_star = b * c * np.sin(alpha) / V 
    b_star = a * c * np.sin(beta) / V 
    c_star = a * b * np.sin(gamma) / V 

    q = 2 * np.pi * np.array([h * a_star, k * b_star, l * c_star])

    return q 

# calculate structure factor via atom form factor function. not needed in the further files at the moment.
def structure_factor(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params):
    """
    Calculate the structure factor depending on material parameters of a crystal.

    Parameters:
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements

    Returns: 
    structure factor S(hkl)
    """
    h, k, l = miller 

    # calculate scattering vector q.
    q = calculate_q_vector(miller, unitcell_lengths, unitcell_angles)
    q_magnitude = np.linalg.norm(q)

    # calculate atomic form factors
    form_factors = np.array([atomic_form_factor(element, params, q_magnitude) for element in atomic_elements])

    # normalize form factors as described in Yariv paper
    form_factors *= len(atomic_elements) / np.sum(form_factors)

    # calculate dot product q.r for each atomic position r
    qr_dot_product = np.dot(atomic_positions, q)

    # calculate structure factor
    structure_factor = np.sum(fi * np.exp(- 2j * np.pi * (h * xi + k * yi + l * zi)) for fi, xi, yi, zi in zip(form_factors, atomic_positions[:,0], atomic_positions[:,1], atomic_positions[:,2]))

    return structure_factor

def structure_factor_db(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, form_factors):
    """
    Calculate the structure factor depending on material parameters of a crystal.

    Parameters:
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements
    form_factors (float array): atomic form factors of all elements of the material, in the same order as the respective positions and elements appear.

    Returns: 
    structure factor S(hkl)
    """
    h, k, l = miller 

    # normalize form factors as described in Yariv paper
    form_factors *= len(atomic_elements) / np.sum(form_factors)
    #print(form_factors)

    # calculate structure factor
    structure_factor = np.sum(fi * np.exp(- 2j * np.pi * (h * xi + k * yi + l * zi)) for fi, xi, yi, zi in zip(form_factors, atomic_positions[:,0], atomic_positions[:,1], atomic_positions[:,2]))

    #print(structure_factor)
    return structure_factor













##########################################################################
### refractive index calculation equations

def refractive_index(N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params):
    """
    Calculate the refractive index depending on material parameters of a crystal. 

    Parameters:
    N_0 (float): total electron density in one unit volume
    lamb (float): wavelength of considered light
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements

    Returns: 
    refractive index (modulated due to crystal lattice's periodic structure)
    """
    # calculate first the different components necessary for the stuff
    h, k, l = miller
    omega = 2 * np.pi * const.c / lamb
    N = len(atomic_elements)

    # calculate structure factor
    s = structure_factor(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params)
    print(f'Structure factor = {s}')
    print(f'-------------------------------------')

    # calculate modulation of index of refraction
    a_G = - N_0 * const.e**2 / (2 * N * omega**2 * const.m_e * const.epsilon_0) * s 
    print(f'a_G: {a_G}')
    print(f'-------------------------------------')

    a_0 = 5.45e-10
    ksi = a_0 / np.sqrt(3) # from the bragg condition ? what does this need to be? 
    print(ksi)

    # calculate phase term
    phase_term = np.exp(2j * np.pi * ksi * np.sqrt(3) / a_0)
    print(f'Phase Term: {np.real(s * phase_term)}')

    # calculate n
    n = a_G * phase_term 

    deltan = np.real(n)

    return deltan


# refractive index determination that makes use of the materials database. this is the one used in further code.
def refractive_index_db(materials_data, material_name, ksi):
    """
    Calculate the refractive index depending on material parameters of a crystal. 

    Parameters:
    materials_data (python dictionary): database with a selection of materials and their properties. should contain miller indices to be considered,
                                        resonant wavelength, unitcell dimensions, unitcell angles, and positions of the atoms in the unit cell as
                                        fractional coordinates, as well as atomic number.
    material_name (string): name of the material to be considered. has to be included in the materials database.
    ksi (float): distance r along the direction to be considered. refractive index will be determined at this distance.

    Returns: 
    refractive index (modulated due to crystal lattice's periodic structure)
    """

    # get material properties from the dictionary provided
    miller = materials_data[material_name]['miller_indices']
    omega = 2 * np.pi * const.c / (materials_data[material_name]['wavelength'] * 1e-10)
    unitcell_lengths = materials_data[material_name]['dimensions']
    unitcell_angles = materials_data[material_name]['angles']
    atomic_positions = get_atom_positions_as_numpy_array(materials_data, material_name)
    atomic_elements = get_atoms(materials_data, material_name)
    form_factors = materials_data[material_name]['atom_form_factor']
    N = len(atomic_elements)
    electrons = materials_data[material_name]['electrons']

    # calculate total electron density of the material
    atom_set = np.array(list(Counter(atomic_elements).values()))
    N_e = 0
    V = (unitcell_lengths[0] * 1e-10)**(3)
    
    for i in range(len(atom_set)):
        N_e += atom_set[i] * electrons[i]

    N_0 = N_e / V    

    # calculate structure factor
    s = structure_factor_db(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, form_factors)

    # calculate modulation of index of refraction
    a_G = - N_0 * const.e**2 / (2 * N * omega**2 * const.m_e * const.epsilon_0) * s 

    #unitcell_lengths[0] *= 1e-10
    #print(unitcell_lengths[0])
    # calculate phase term
    phase_term = np.exp(2j * np.pi * ksi * np.sqrt(3) / unitcell_lengths[0])

    # calculate n
    n = a_G * phase_term 
    #print(n)
    deltan = np.real(n)

    return deltan



def refractive_index_lam(materials_data, material_name, ksi, lambd):
    """
    Calculate the refractive index depending on material parameters of a crystal. 

    Parameters:
    materials_data (python dictionary): database with a selection of materials and their properties. should contain miller indices to be considered, 
                                        resonant wavelength, unitcell dimensions, unitcell angles, and positions of the atoms in the unit cell as
                                        fractional coordinates, as well as atomic number.
    material_name (string): name of the material to be considered. has to be included in the materials database.
    ksi (float): distance r along the direction to be considered. refractive index will be determined at this distance.
    lambd (float): consider material where the used wavelength differs from the ideal resonant wavelength of the system.

    Returns: 
    refractive index (modulated due to crystal lattice's periodic structure)
    """

    # get material properties from the dictionary provided
    miller = materials_data[material_name]['miller_indices']
    omega = 2 * np.pi * const.c / (lambd * 1e-10)
    unitcell_lengths = materials_data[material_name]['dimensions']
    unitcell_angles = materials_data[material_name]['angles']
    atomic_positions = get_atom_positions_as_numpy_array(materials_data, material_name)
    atomic_elements = get_atoms(materials_data, material_name)
    form_factors = materials_data[material_name]['atom_form_factor']
    N = len(atomic_elements)
    electrons = materials_data[material_name]['electrons']

    # calculate total electron density of the material
    atom_set = np.array(list(Counter(atomic_elements).values()))
    N_e = 0
    V = (unitcell_lengths[0] * 1e-10)**(3)
    
    for i in range(len(atom_set)):
        N_e += atom_set[i] * electrons[i]

    N_0 = N_e / V    

    # calculate structure factor
    s = structure_factor_db(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, form_factors)

    # calculate modulation of index of refraction
    a_G = - N_0 * const.e**2 / (2 * N * omega**2 * const.m_e * const.epsilon_0) * s 
    
    # calculate phase term
    phase_term = np.exp(2j * np.pi * ksi * np.sqrt(3) / unitcell_lengths[0])

    # calculate n
    n = a_G * phase_term 
    deltan = np.real(n)

    return deltan





###########################################################################
### coupling const. and threshold gain 

def coupling_constant(N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params):
    """
    Calculate the coupling constant of x-ray light interacting with a periodic lattice.

    Parameters:
    N_0 (float): total electron density in one unit volume
    lamb (float): wavelength of considered light
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements

    Returns: 
    Coupling constant kappa.
    """
    deltan = refractive_index(N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params)
    print(f'Delta N = {deltan}')

    kappa = 2 * np.pi * const.c / lamb * deltan / (2 * const.c)
    print(f'Coupling constant: {kappa}')

    return kappa

def coupling_constant_db(n, lambd):
    """
    Calculate the coupling constant using an already known refractive index modulation and already known wavelength lambda.

    It Represents Equation 3 of Yariv Paper with  omega =  2pi/(c*lambda)

    Attention. n ist im Zähler

    return unit is in 1/nm
    """
    lambd *= 1e-1 #converting Angstrom to nm
    kappa = 2 * np.pi * const.c / lambd * n / (2 * const.c)

    return kappa #1/nm





def threshold_gain(L, N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params):
    """
    Calculate the threshold gain constant of an x-ray lasing system, depening on material properties and wavelength.

    Parameters:
    L (float): sample side length in nm!!
    N_0 (float): total electron density in one unit volume
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements

    Returns: 
    Threshold gain g.

    From Equation 12
    """
    L_x, L_y, L_z = L
    V = L_x * L_y * L_z

    kappa = coupling_constant(N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params)
    g = (1/V) * (np.pi / kappa)**2

    return g

def threshold_gain_db(L, kappa):
    """
    Calculate the coupling constant using an already known coupling constant and already known system dimension L.
    """
    L_x, L_y, L_z = L
    V = L_x * L_y * L_z

    g = (1/V) * (np.pi / kappa)**2

    return g




##############################################################################
### functions used to compare with results of Yariv paper. not relevant for 
### further analysis.

def deltan_paper_eq(N_0, lambd, ksi, a_0):
    """
    Calculate the refractive index depending on material parameters of a crystal.
    Calculation only form one specific example, GaP, to compare with Yariv paper 
    results. Not to be used in further analysis; is not generally valid.

    Parameters:
    ksi (float): distance along the (111) direction, in meters
    N_0 (float): total electron density within one unit volume
    lambd (float): wavelength of the considered light
    a_0 (float): unit cell side length

    Returns: 
    refractive index (modulated due to crystal lattice's periodic structure)
    """

    omega = 2 * np.pi * const.c / lambd 

    deln = - N_0 * const.e**2 / (np.sqrt(8) * omega**2 * const.m_e * const.epsilon_0) * np.cos(2 * np.pi * np.sqrt(3) / a_0 * (ksi - a_0 / (8 * np.sqrt(3))))

    phase_del = np.sqrt(2) * np.cos(2 * np.pi * np.sqrt(3) / a_0 * (ksi - a_0 / (8 * np.sqrt(3))))

    print(- N_0 * const.e**2 / (np.sqrt(2) * omega**2 * const.m_e * const.epsilon_0) / 4)
    #print(f'other phase term: {phase_del}')
    return deln 


# same function as refractive_index, but taking the additional argument ksi, in order to plot it with respect to the distance along the (111) dimension
def refractive_index_ksi(ksi, N_0, lamb, miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params):
    """
    Calculate the refractive index depending on material parameters of a crystal.
    To find the refractive index with respect to some distance along the lattice direction.
    Use it to find the maximum value for the oscillating refractive index modulation.

    Parameters:
    ksi (float): distance along the (111) direction, in meters
    N_0 (float): total electron density within one unit volume
    miller (float array): [h, k, l], Miller indices
    unitcell_lengths (float array): [a, b, c], unit cell lengths
    unitcell_angles (float array): unit cell angles in degrees
    atomic_positions (np.ndarray): atomic positions in the unit cell (Nx3 shape)
    atomic_elements (list): List of symbols of the elements

    Returns: 
    refractive index (modulated due to crystal lattice's periodic structure)
    """
    # calculate first the different components necessary for the stuff
    h, k, l = miller
    omega = 2 * np.pi * const.c / lamb
    N = len(atomic_elements)

    # calculate structure factor
    s = structure_factor(miller, unitcell_lengths, unitcell_angles, atomic_positions, atomic_elements, params)

    # calculate modulation of index of refraction
    a_G = - N_0 * const.e**2 / (2* N * omega**2 * const.m_e * const.epsilon_0) * s

    a_0 = 5.45e-10

    print(a_G)

    # calculate phase term
    phase_term = np.exp(2j * np.pi * ksi * np.sqrt(3) / a_0) 

    # calculate n
    n = a_G * phase_term 

    deltan = np.real(n)

    return deltan


######################################################################################
### functions used to find the maximum index of reflection modulation.


def find_max_n(materials_data, material):
    """
    Function that takes material properties and gives back the maximum refractive 
    index modulation.

    Parameters:
    materials_data (dictionary): database with material properties
    material (string): symbol of the material that is to be analyzed

    Returns:
    maximum of index of refraction modulation
    """
    unitcell_l = materials_data[material]['dimensions']

    x = np.linspace(0, unitcell_l[0] * np.sqrt(3), 100)

    n_array = np.array([refractive_index_db(materials_data, material, xi) for xi in x])

    return np.max(n_array)


def find_max_n_lam(materials_data, material, lambd):
    """
    Function that takes material properties and gives back the maximum refractive 
    index modulation. Used in case the applied wavelength is not the resonant wavelength of the material.

    Parameters:
    materials_data (dictionary): database with material properties
    material (string): symbol of the material that is to be analyzed
    lambd (string): used wavelength in angstrom

    Returns:
    maximum of index of refraction modulation
    """

    unitcell_l = materials_data[material]['dimensions']

    x = np.linspace(0, unitcell_l[0] * np.sqrt(3), 100)

    n_array = np.array([refractive_index_lam(materials_data, material, xi, lambd) for xi in x])

    return np.max(n_array)



########################################################################
### read data file and convert into numpy objects; and all auxiliary 
### functions neccessary to use it efficiently

def read_materials_data(file_path):
    """
    Function that reads a text file containing each materials properties.
    Gives as output a list with the material names, miller indices, unit 
    cell dimensions and angles, number of constituent atoms, positions of
    atoms in unit cell.

    """
    # set up material dictionary and read from file
    materials = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    material = None 
    reading_positions = False 

    # iterate through the read document and extract all necessary information for each line
    for line in lines:
        line = line.strip()
        if 'Material:' in line:
            material = line.split(':',1)[1].strip()
            materials[material] = {
                'miller_indices': None,
                'dimensions': None,
                'angles': None,
                'wavelength': None,
                'electrons' : None,
                'atom_form_factors': [],
                'positions': []
            }
            reading_positions = False
        elif 'Miller:' in line:
            materials[material]['miller_indices'] = np.array(list(map(int, line.split(':')[1].strip().split())))
            reading_positions = False
        elif 'Dimensions:' in line:
            materials[material]['dimensions'] = np.array(list(map(float, line.split(':')[1].strip().split())))
            reading_positions = False
        elif 'Angles:' in line:
            materials[material]['angles'] = np.array(list(map(float, line.split(':')[1].strip().split())))
            reading_positions = False
        elif 'Wavelength:' in line:
            materials[material]['wavelength'] = float(line.split(':')[1].strip())
            reading_positions = False
        elif 'Electrons:' in line:
            materials[material]['electrons'] = np.array(list(map(float, line.split(':')[1].strip().split())))
            reading_positions = False
        elif 'Positions:' in line:
            reading_positions = True
        elif reading_positions and line:
            # see about the fractions?
            materials[material]['positions'].append(line.split())

    return materials


def get_atom_positions_as_numpy_array(material_data, material_name):
    """
    Function to obtain an array of the positions of the atom in a given 
    material. 
    Requires as input a dictionary of material data and the name of the material
    in question. 

    Returns a numpy array of atom positions.
    """

    # check first if the material is contained in the database
    if material_name in material_data:
        positions = material_data[material_name]['positions']
        position_array = np.array([[float(pos[1]), float(pos[2]), float(pos[3])] for pos in positions])
        return position_array
    else:
        print(f"Material {material_name} not found.")


def get_atoms(material_data, material_name):
    """
    Function that gives the atom configuration of a given material. Will 
    return order and names of atom in such a way that fits the position array.
    """

    if material_name in material_data:
        material_positions = material_data[material_name]['positions']
        atoms = [pos[0] for pos in material_positions]
        return atoms 
    else:
        print(f"Material {material_name} not found.")

def remove_invalid_entries(material_data, invalid_position=[2.0, 2.0, 2.0]):
    """
    Entries in the database where the material properties are not fully defined
    or where the entry is not yet ready are to be marked by setting the positions to [2 2 2]
    for all constituent atoms. This function can then be used to delete invalid entries.
    Takes as argument the material data dictionary. invalid_position can be changed,
    is [2 2 2] by default.
    """

    invalid_position = np.array(invalid_position)
    materials_to_remove = []

    for material, properties in material_data.items():
        positions = get_atom_positions_as_numpy_array(material_data, material)
        if np.all(positions == invalid_position, axis=1).any():
            materials_to_remove.append(material)

    for material in materials_to_remove:
        del material_data[material]




#################################################################################
### functions to try online downloading the atomic form factors from NIST (Henke).
### will take a while and NEEDS INTERNET CONNECTION TO RUN since it accesses the 
### NIST website directly.


@functools.cache
def atomicformfactor_nist(Z):
    """
    Function that takes an atom symbol as string as an argument. Accesses the Nist database,
    which in turn uses Henke data, to download directly the whole atomic form factor table.
    Requires internet connection. Otherwise it will throw a "connection timed out" type error.
    Please don't mess this one up, it took me ages to get it to work.
    """

    urltmplate = 'https://physics.nist.gov/cgi-bin/ffast/ffast.pl?gtype=4&Formula={Z}'
    url = urltmplate.format(Z=Z)
    session = requests.Session()

    r = session.get(url).content

    soup = BeautifulSoup(r, features="lxml")
    tabledata = soup.select('body > pre')[0].text.splitlines()[3:]
    tabledata = '\n'.join(tabledata)

    data = np.genfromtxt(StringIO(tabledata))
    eV = data[:,0] * 1e3
    f = data[:,1]

    form_factor_table = np.vstack((eV, f))

    print(f'got for {Z}')

    return form_factor_table 


def download_form_factors(materials_data, formfactor_data):
    """
    Calls atomicformfactor_nist for an entire dictionary of materials to find the value of 
    atomic form factor which is closest to the energy listed as resonant wavelength in the 
    material database. 
    Might take a minute to run, depending on internet connection. 
    Will save the atomic form factors as an array of one atomic form factor value per atom in the
    unit cell. Adds them under the key 'atomic_form_factor' to the dictionary as a numpy array.
    """

    processed_atoms = set()

    for mat, properties in materials_data.items():
        if not os.path.exists(formfactor_data):
            os.makedirs(formfactor_data)


        atoms = get_atoms(materials_data, mat)

        # some materials only have one atom type, dont need two form factor calculations
        if atoms[0] == atoms[-1]:
            # download from database
            table_1 = atomicformfactor_nist(atoms[0])  

            form_factor_data = table_1[0:2,:]


            if atoms[0] not in processed_atoms:
                output_file_path = os.path.join(formfactor_data, f"{atoms[0]}.txt")
                with open(output_file_path, 'a') as output_file:
                    np.savetxt(output_file, (table_1[0,:], table_1[1,:]))
                processed_atoms.add(atoms[0])
                print(f'processed {atoms[0]}')
        else:
            # download from database
            table_1 = atomicformfactor_nist(atoms[0])
            table_2 = atomicformfactor_nist(atoms[-1])

            form_factor_data_1 = table_1[0:2, :]
            form_factor_data_2 = table_2[0:2, :]

            if atoms[0] not in processed_atoms:
                output_file_path_1 = os.path.join(formfactor_data, f"{atoms[0]}.txt")
                with open(output_file_path_1, 'a') as output_file:
                    np.savetxt(output_file, (table_1[0,:], table_1[1,:]))
                processed_atoms.add(atoms[0])
                print(f'processed {atoms[0]}')
            
            if atoms[-1] not in processed_atoms:
                output_file_path_2 = os.path.join(formfactor_data, f"{atoms[-1]}.txt")
                with open(output_file_path_2, 'a') as output_file:
                    np.savetxt(output_file, (table_2[0,:], table_2[1,:]))
                processed_atoms.add(atoms[-1])
                print(f'processed {atoms[-1]}')

        #print(properties['atom_form_factor'])



def get_form_factors(materials_data):
    """
    Calls atomicformfactor_nist for an entire dictionary of materials to find the value of 
    atomic form factor which is closest to the energy listed as resonant wavelength in the 
    material database. 
    Might take a minute to run, depending on internet connection. 
    Will save the atomic form factors as an array of one atomic form factor value per atom in the
    unit cell. Adds them under the key 'atomic_form_factor' to the dictionary as a numpy array.
    """

    for mat, properties in materials_data.items():
        energy = const.h * const.c / (properties['wavelength'] * 1e-10 * const.electron_volt) 

        atoms = get_atoms(materials_data, mat)

        # some materials only have one atom type, dont need two form factor calculations
        if atoms[0] == atoms[-1]:
            count_1 = atoms.count(atoms[0])

            # download from database
            table_1 = atomicformfactor_nist(atoms[0])

            # find energy value closest to the wavelength we consider
            differences_1 = np.abs(table_1[0,:] - energy)
            min_index_1 = np.argmin(differences_1)

            form_factor_1 = np.full(count_1, table_1[1, min_index_1])
            properties['atom_form_factor'] = form_factor_1          
        else:
            count_1 = atoms.count(atoms[0])
            count_2 = atoms.count(atoms[-1])

            # download from database
            table_1 = atomicformfactor_nist(atoms[0])
            table_2 = atomicformfactor_nist(atoms[-1])


            # find energy value closest to the wavelength we consider
            differences_1 = np.abs(table_1[0,:] - energy)
            differences_2 = np.abs(table_2[0,:] - energy)

            min_index_1 = np.argmin(differences_1)
            min_index_2 = np.argmin(differences_2)

            form_factor_1 = np.full(count_1, table_1[1, min_index_1])
            form_factor_2 = np.full(count_2, table_2[1, min_index_2])

            properties['atom_form_factor'] = np.concatenate((form_factor_1, form_factor_2))

        #print(properties['atom_form_factor'])


def get_form_factors_local(materials_data, formfactor_data):
    """
    Takes the atomic form factors of each material in materials_data from the folder formfactor_data and adds 
    it to the dictionary under the key atom_form_factor. material information has to be contained in a text file
    in the folder mentioned before with the name of the atom symbol.
    """

    for mat, properties in materials_data.items():
        energy = const.h * const.c / (properties['wavelength'] * 1e-10 * const.electron_volt) 

        atoms = get_atoms(materials_data, mat)

        # some materials only have one atom type, dont need two form factor calculations
        if atoms[0] == atoms[-1]:
            count_1 = atoms.count(atoms[0])

            # download from database
            filepath = os.path.join(formfactor_data, f"{atoms[0]}.txt")

            with open(filepath, 'r') as file:
                data = np.loadtxt(file)
                energies = data[0]
                form_factors = data[1]

                # find closest value
                index = np.abs(energies - energy).argmin()
                closest_form_factor = form_factors[index]

                form_factor_1 = np.full(count_1, closest_form_factor)
                properties['atom_form_factor'] = form_factor_1 
        else:
            count_1 = atoms.count(atoms[0])
            count_2 = atoms.count(atoms[-1])

            form_factor_1 = np.empty(count_1)
            form_factor_2 = np.empty(count_2)

            # download from database
            filepath_1 = os.path.join(formfactor_data, f"{atoms[0]}.txt")
            filepath_2 = os.path.join(formfactor_data, f"{atoms[-1]}.txt")


            with open(filepath_1, 'r') as file:
                data = np.loadtxt(file)
                energies = data[0]
                form_factors = data[1]

                # find closest value
                index = np.abs(energies - energy).argmin()
                closest_form_factor = form_factors[index]

                form_factor_1 = np.full(count_1, closest_form_factor)

            with open(filepath_2, 'r') as file:
                data = np.loadtxt(file)
                energies = data[0]
                form_factors = data[1]

                # find closest value
                index = np.abs(energies - energy).argmin()
                closest_form_factor = form_factors[index]

                form_factor_2 = np.full(count_1, closest_form_factor)
            
            properties['atom_form_factor'] = np.concatenate((form_factor_1, form_factor_2))

#Function of RAFFAELE

def find_gamma(L, kappa, initial_guess=(0.127, 0)):
    """
    Finds the eigenvalue gamma for the transcendental equation:
    kappa = ±j * gamma * sinh(gamma * L).

    Parameters:
        L (float): Length of the medium.
        kappa (float): Known parameter in the equation.
        initial_guess (tuple): Initial guess for the real and imaginary parts of gamma.

    Returns:
        complex: The eigenvalue gamma as a Python complex number (x + jy).
    """
    # Define the system of equations
    def equations(vars):
        x, y = vars  # gamma = x + jy
        f1 = x * np.sinh(x * L) * np.cos(y * L) - y * np.cosh(x * L) * np.sin(y * L)
        f2 = x * np.cosh(x * L) * np.sin(y * L) + y * np.sinh(x * L) * np.cos(y * L) + kappa
        return [f1, f2]

    # Solve the equations
    solution = root(equations, initial_guess)

    # Check for success
    if not solution.success:
        raise ValueError(f"Root finding did not converge: {solution.message}")

    # Extract the solution
    x, y = solution.x
    return complex(x, y)  # Return gamma as a complex number


def plot_relative_intensity(gamma, L, z_range):
    """
    Computes and plots the expression:
    sinh(gamma * (z + 0.5 * L)) * sinh(gamma * (z + 0.5 * L)).conjugate() +
    sinh(gamma * (z - 0.5 * L)) * sinh(gamma * (z - 0.5 * L)).conjugate()
    for a range of z values.

    Parameters:
        gamma (complex): Complex eigenvalue gamma.
        L (float): Length of the medium.
        z_range (tuple): Range of z values as (z_min, z_max, num_points).

    Returns:
        None: Displays a plot of the computed values.
    """
    # Unpack z_range
    z_min, z_max, num_points = z_range

    # Generate z values
    z_values = np.linspace(z_min, z_max, num_points)

    # Compute the expression for each z value
    results = []
    for z in z_values:
        term1 = np.sinh(gamma * (z + 0.5 * L))
        term2 = np.sinh(gamma * (z - 0.5 * L))
        result = term1 * np.conjugate(term1) + term2 * np.conjugate(term2)
        results.append(result.real)  # Store the real part

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, results, label=f"γ = {gamma}, L = {L}", color='b')
    plt.title("Plot of Computed Expression")
    plt.xlabel("z")
    plt.ylabel("Value of Expression")
    plt.grid(True)
    plt.legend()
    plt.show()