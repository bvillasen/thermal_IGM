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

grid_dir = data_dir + 'cosmo_sims/sim_grid/1024_P19m_np4_nsim400/'
sim_dirs = [ dir for dir in os.listdir(grid_dir)  if dir[0] == 'S' ]
sim_dirs.sort()
n_sims = len( sim_dirs )
sim_ids_local = split_indices( range(n_sims), rank, n_procs )
print( sim_ids_local )

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 10000 * 20
z_end = 2.
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

for sim_id in sim_ids_local:
  sim_dir = grid_dir + f'/reduced_files/{sim_dirs[sim_id]}/'
  output_dir = sim_dir + f'thermal_solution/'
  create_directory( output_dir )

  file_name = sim_dir + 'UVB_rates.h5'
  rates = Load_Grackle_UVB_File( file_name )

  uvb_parameters = { 'scale_He':1.0, 'scale_H':1.0, 'delta_z_He':0.0, 'delta_z_H':0.0 } 
  uvb_rates = Modify_UVB_Rates( uvb_parameters,  rates )

  solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples  )

  output_file_name = output_dir + f'solution.h5'
  Write_Solution( solution, output_file_name )
  Plot_Solution( output_dir, solution, file_name=f'solution_{model_id}.png' )
  Plot_HI_fraction( output_dir, solution=solution, input_file=None, solutions=None, file_name='HI_fraction.png', HI_data=None, data_labels=None )
