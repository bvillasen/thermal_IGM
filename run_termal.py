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
from uvb_functions import Modify_UVB_Rates, Reaplace_Gamma_Parttial
from data_functions import Write_Solution
from plot_functions import *

use_mpi = False
if use_mpi:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_procs = comm.Get_size()
else:
  rank = 0
  n_procs = 1


output_dir = data_dir + 'thermal_V22/'
# uvb_rates_file = 'data/UVB_rates_P19m.h5'
uvb_rates_file = 'data/UVB_rates_V22.h5'
print( f'UVB Rates File: {uvb_rates_file}' )
print( f'Output Dir:     {output_dir}' )
create_directory( output_dir, print_out=False )

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 10000
z_end = 4.
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
uvb_rates = Load_Grackle_UVB_File( uvb_rates_file )
uvb_parameters = {'scale_H':1.0, 'scale_He':1.0, 'delta_z_H':0.0, 'delta_z_He':0.0 } 
uvb_rates = Modify_UVB_Rates( uvb_parameters,  uvb_rates )

# Integrate the solution
integrator = 'bdf'
# integrator = 'rk4'
solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, output_to_file=None, integrator=integrator )

output_file_name = output_dir + f'solution_{integrator}_grackle.h5'
Write_Solution( solution, output_file_name )
