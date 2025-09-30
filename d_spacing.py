import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#lattice parameter
a = 2 #nm

#d_spacing from planes

def d_spacing(lattice_planes):
    """

    :param lattice_planes: NP ARRAY
    :return: Array of d Spacing
    """
    for i, plane in enumerate(lattice_planes):
        d_space[i] = (a / np.linalg.norm(plane))
    return d_space


"""
lattice_planes = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2], [4, 0, 0], [3, 3, 1], [4, 2, 0], [4, 2, 2], [5, 1, 1]])

d_spacing = np.zeros(shape = len(lattice_planes))

for i, plane in enumerate(lattice_planes):
    d_spacing[i] = (a/np.linalg.norm(plane))
"""