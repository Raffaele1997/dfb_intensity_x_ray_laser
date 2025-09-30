import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import scipy.constants as const
import os

import functions
"""
To download atom form factors from Henke database via NIST website. Uses internet, will not run if there is no connection.
In order to download the data again, uncomment the functions.download... line, but if the folder with the data exists, this
is not necessary.
"""

materials_data = functions.read_materials_data('data_for_different_materials.txt')

#functions.download_form_factors(materials_data, "formfactor_data")

functions.get_form_factors_local(materials_data, "formfactor_data")

print(materials_data['CuO'])