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


project_name = 'reduced_heating'
# project_name = 'zero_heat'

input_dir  = data_dir + f'modified_uvb_rates/{project_name}/uvb_models/'
output_dir = data_dir + f'modified_uvb_rates/{project_name}/thermal_solutions/'
create_directory( output_dir )

model_base_name = 'UVB_rates'

model_files = [ f for f in os.listdir(input_dir) if f.find(model_base_name)>=0 and f.find('.h5')>=0  ]
model_files.sort()

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 10000 * 10
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


for model_id, model_file_name in enumerate(model_files):

  file_name = input_dir + model_file_name
  rates = Load_Grackle_UVB_File( file_name )

  uvb_parameters = { 'scale_He':1.0, 'scale_H':1.0, 'delta_z_He':0.0, 'delta_z_H':0.0 } 
  uvb_rates = Modify_UVB_Rates( uvb_parameters,  rates )

  solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples  )

  output_file_name = output_dir + f'solution_{model_id}.h5'
  Write_Solution( solution, output_file_name )
  # Plot_Solution( output_dir, solution, file_name=f'solution_{model_id}.png' )
  # Plot_HI_fraction( output_dir, solution=solution, input_file=None, solutions=None, file_name='HI_fraction.png', HI_data=None, data_labels=None )
