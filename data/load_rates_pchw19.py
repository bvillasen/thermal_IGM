import numpy as np
from cosmo_constants import eV_to_ergs

file_name = 'data/data_pchw19.dat'
# file_name = 'data/data_pchw19_equilibrium.dat'

data = np.loadtxt( file_name ).T

z = data[0]
i_HI = data[1]
h_HI = data[2]
i_HeI = data[3]
h_HeI = data[4]
i_HeII = data[5]
h_HeII = data[6]

rates_pchw19 = {}
rates_pchw19['heating'] = {}  #eV / s
rates_pchw19['ionization'] = {}  # 1 / s
rates_pchw19['z'] = z
rates_pchw19['heating']['HI'] = h_HI     * eV_to_ergs
rates_pchw19['heating']['HeI'] = h_HeI   * eV_to_ergs
rates_pchw19['heating']['HeII'] = h_HeII * eV_to_ergs

rates_pchw19['ionization']['HI'] = i_HI
rates_pchw19['ionization']['HeI'] = i_HeI
rates_pchw19['ionization']['HeII'] = i_HeII