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

use_mpi = True
if use_mpi:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  n_procs = comm.Get_size()
else:
  rank = 0
  n_procs = 1


grid_name = '1024_wdmgrid_extended_beta'
grid_name = '1024_wdmgrid_cdm_extended_beta'
grid_dir = data_dir + f'cosmo_sims/sim_grid/{grid_name}/' 
root_dir = grid_dir + f'simulation_files/'
thermal_dir = grid_dir + 'thermal/'
if rank == 0: create_directory( thermal_dir )

sim_dirs = [ dir for dir in os.listdir(root_dir) if dir[0]=='S' ]
sim_dirs.sort()


n_simulations = len( sim_dirs )
sim_ids = range(n_simulations)
sim_ids_local = split_indices( sim_ids, rank, n_procs )
print( f'proc_id: {rank}  sim_ids_local:{sim_ids_local}')




for sim_id in sim_ids_local:
  sim_dir = root_dir + sim_dirs[sim_id] + '/'
  uvb_rates_file = sim_dir + 'UVB_rates.h5'
  output_dir     = thermal_dir + f'{sim_dirs[sim_id]}/'
  print( f'UVB Rates File: {uvb_rates_file}' )
  print( f'Output Dir:     {output_dir}' )
  create_directory( output_dir, print_out=False )

  # Initialize Cosmology
  z_start = 16
  cosmo = Cosmology( z_start )

  # Initialize parameters
  n_samples = 100000
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
  solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, output_to_file=None )
  
  # output_file_name = output_dir + 'solution.h5'
  # Write_Solution( solution, output_file_name )
  
  z      = solution['z']
  nH     = solution['n_H']
  nHII   = solution['n_HII']
  nHe    = solution['n_He']
  nHeIII = solution['n_HeIII']
  ion_frac_H  = 0.999
  ion_frac_He = 0.999
  
  HII_frac   = nHII / nH
  z_ion_H  = z[HII_frac   > ion_frac_H].max()
  
  
  # HeIII_frac = nHeIII / nHe
  # z_ion_He = z[HeIII_frac > ion_frac_He].max()
  
  
  # global_props = { 'ion_frac_H':ion_frac_H, 'ion_frac_He':ion_frac_He, 'z_ion_H':z_ion_H, 'z_ion_He':z_ion_He }
  global_props = { 'ion_frac_H':ion_frac_H, 'z_ion_H':z_ion_H,  }
  
  output_file_name = output_dir + 'global_properties.pkl'
  print( global_props)
  Write_Pickle_Directory( global_props, output_file_name )

  

