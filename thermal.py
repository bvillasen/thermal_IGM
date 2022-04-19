import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

thermal_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(thermal_dir)]
sys.path.extend(subDirectories)
from tools import *
from cosmology import Cosmology
from cosmo_constants import Myear, Mpc, K_b, M_p, Gcosmo    
from load_rates_pchw19 import rates_pchw19, rates_pchw19_eq
from load_grackle_rates_file import Load_Grackle_UVB_File
from temp_functions import Integrate_Evolution
from uvb_functions import Modify_UVB_Rates, Reaplace_Gamma_Parttial, Extend_Rates_Redshift
from data_functions import Write_Solution
from plot_functions import *

# proj_dir = data_dir + 'projects/thermal_history/'
# output_dir = proj_dir + 'data/ionization_history/'

proj_dir = data_dir + 'projects/thermal_history/'
input_dir = proj_dir + 'data/1024_50Mpc_modified_gamma_sigmoid/'
output_dir = input_dir

create_directory( output_dir )

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 10000
z_end = 0.
T_start = 5
X = 0.75984

# Set Number Densities
rho_gas_mean = cosmo.rho_gas_mean # kg cm^-3
rho_H = X*rho_gas_mean
rho_He = (1-X)*rho_gas_mean
n_H_comov = rho_H / M_p          # cm^-3
n_He_comov = rho_He / (4*M_p)    # cm^-3
a_start = 1. / (z_start + 1)
rho_cgs = rho_gas_mean * 1e3 / a_start**3 

H = cosmo.get_Hubble( a_start ) * 1000 / Mpc  # 1/sec


# Set photoheating and photoionization rates
# file_name = 'data/CloudyData_UVB_Puchwein2019_cloudy.h5'
# uvb_rates = Load_Grackle_UVB_File( file_name )
# max_delta_z = 0.1
# uvb_rates = Extend_Rates_Redshift( max_delta_z, uvb_rates, log=True )
# # uvb_parameters_v21 = {'scale_H':1.0, 'scale_He':1.0, 'delta_z_H':0.0, 'delta_z_He':0.0 } 
# uvb_parameters_V22 = { 'scale_He':0.47, 'scale_H':0.81, 'delta_z_He':0.25, 'delta_z_H':-0.09 } 
# uvb_rates = Modify_UVB_Rates( uvb_parameters_V22,  uvb_rates )

# alpha = 3.5
# file_name = f'data/UVB_rates_V22_modified_sigmoid_{alpha}.h5'
# uvb_rates = Load_Grackle_UVB_File( file_name )

file_name = input_dir + 'UVB_rates.h5'
uvb_rates = Load_Grackle_UVB_File( file_name )


uvb_parameters = { 'scale_He':1.0, 'scale_H':1.0, 'delta_z_He':0.0, 'delta_z_H':0.0 }
uvb_rates = Modify_UVB_Rates( uvb_parameters,  uvb_rates ) 

output_to_file = None
solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, output_to_file=output_to_file  )

output_file_name = output_dir + f'thermal_solution.h5'
Write_Solution( solution, output_file_name )

# Plot_Solution( output_dir, solution )