import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import functions


# read data from the database, clean up, obtain form factors.
materials_data = functions.read_materials_data('data_for_different_materials.txt')
functions.remove_invalid_entries(materials_data)
functions.get_form_factors_local(materials_data, "formfactor_data")

output_dir = r"/Users/raffaele/PycharmProjects/Gain_Planes/HoN"

delta_n = functions.find_max_n(materials_data, "HoN")

print(delta_n)