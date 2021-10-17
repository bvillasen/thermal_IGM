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
from load_rates_pchw19 import rates_pchw19
from load_grackle_rates_file import Load_Grackle_UVB_File
from temp_functions import Integrate_Evolution
from uvb_functions import Modify_UVB_Rates
from data_functions import Write_Solution
from plot_functions import *

use_mpi = True
if use_mpi:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_procs = comm.Get_size()
else:
  rank = 0
  n_procs = 1

output_dir = data_dir + 'tau_electron/'
if rank == 0:  create_directory( output_dir )

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 10000 * 1000
z_end = 0.
T_start = 1
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

param_vals = {}
param_vals[0] = [ 0.33, 0.54 ]
param_vals[1] = [ 0.75, 0.79 ]
param_vals[2] = [ 0.21, 0.38 ]
param_vals[3] = [ 0.02, 0.17 ]  


param_grid = Get_Parameters_Combination( param_vals )
n_models = len( param_grid )


# model_id = rank
model_id = 'HL'
# Load the Original Rates
grackle_file_name =  'data/CloudyData_UVB_Puchwein2019_cloudy.h5'
rates_P19 = Load_Grackle_UVB_File( grackle_file_name )

# p_vals = param_grid[model_id]
p_vals = [ 0.45, 0.77, 0.31, 0.1 ]
uvb_parameters = { 'scale_He':p_vals[0], 'scale_H':p_vals[1], 'delta_z_He':p_vals[2], 'delta_z_H':p_vals[3] } 
uvb_rates = Modify_UVB_Rates( uvb_parameters,  rates_P19 )

solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples  )

output_file_name = output_dir + f'solution_{model_id}.h5'
Write_Solution( solution, output_file_name )

