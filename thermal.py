import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time as timer


thermal_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(thermal_dir)]
sys.path.extend(subDirectories)
from tools import *
from cosmology import Cosmology
from cosmo_constants import Myear, Mpc, K_b, M_p, Gcosmo    
from rk4 import RK4_step
from load_rates_pchw19 import rates_pchw19
from temp_functions import *


import matplotlib
# set some global options
matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
# matplotlib.rcParams['font.sans-serif'] = "Helvetica"
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

output_dir = '/home/bruno/Desktop/'

# Initialize Cosmology
z_start = 16
cosmo = Cosmology( z_start )

# Initialize parameters
n_samples = 20000
z_end = 0.
T_start = 0.1
X = 0.75984

# Set Number Densities
rho_gas_mean = cosmo.rho_gas_mean # kg cm^-3
rho_H = X*rho_gas_mean
rho_He = (1-X)*rho_gas_mean
n_H_comov = rho_H / M_p          # cm^-3
n_He_comov = rho_He / (4*M_p)    # cm^-3

# Create scale factor array
a_start = 1. / ( z_start + 1 )
a_end = 1. / ( z_end + 1 )
z_vals = np.linspace( z_end, z_start, n_samples)[::-1]
a_vals = 1/( z_vals + 1 )
da_vals = a_vals[1:] - a_vals[:-1]
dz_vals = z_vals[1:] - z_vals[:-1]

# Set photoheating and photoionization rates
uvb_rates = rates_pchw19


# Integrate Temperature evolution
n = len( da_vals )
time = 0 
current_a = a_start
temp_eq = T_start
temp_vals = [temp_eq]

fields = [ 'temperature', 'n_H', 'n_HI', 'n_HII', 'n_He', 'n_HeI', 'n_HeII', 'n_HeIII', 'n_e' ]
solution = {}
current_state = {}
for field in fields:
  solution[field] = []







n_min = 1e-60
current_state = {}
current_state['temperature'] = T_start
current_state['n_H']     = n_H_comov / a_start**3
current_state['n_HI']    = n_H_comov / a_start**3
current_state['n_HII']   = n_min
current_state['n_He']    = n_He_comov / a_start**3
current_state['n_HeI']   = n_He_comov / a_start**3
current_state['n_HeII']  = n_min
current_state['n_HeIII'] = n_min
current_state['n_e']     = 4*n_min

n_iter = 1000
start = timer.time()
print('Integrating Thermal Evolution...')
for i in range(n):

  H = cosmo.get_Hubble( current_a )
  current_z = 1/current_a - 1

  current_rates = interpolate_rates( current_z, uvb_rates )

  # #Get Ioniozation Fractions
  # n_H = n_H_comov / current_a**3
  # n_He = n_He_comov / current_a**3 
  # chemistry_eq = Get_Ionization_Fractions_Iterative( n_H, n_He, temp_eq, current_rates['ionization']) 

  append_state_to_solution( fields, current_state, solution )

  delta_a = da_vals[i]
  delta_z = dz_vals[i]
  dt = cosmo.get_dt( current_a, delta_a )

  state_array = get_state_array( fields, current_state )
  state_array = RK4_step( all_deriv, time, state_array, dt, cosmo=cosmo, uvb_rates=uvb_rates  )
  current_state = update_current_state( fields, state_array, current_state )
  # temp_eq = RK4_step( temp_deriv, time, temp_eq, dt, H=H, rates=current_rates, chemistry=chemistry_eq, z=current_z) 
  # temp_vals.append(temp_eq)

  time += dt
  current_a += delta_a
  
  if i%n_iter == 0:
    end = timer.time()
    delta = end - start
    print_str = 'z = {0:.3f}   T = {1:.2f}'.format(current_z, current_state['temperature']) 
    printProgress( i, n, delta, print_str )
    


append_state_to_solution( fields, current_state, solution )
for field in fields:
  solution[field] = np.array(solution[field])
printProgress( i, n, delta )
print('\nEvolution Fisished')



print('Plotting Results')



# Load Simulation Thermal History 
file_name = 'data/thermal_history_CHIPS_P19.txt'
th_pchw19 = np.loadtxt( file_name ).T 

file_name = 'data/global_statistics_pchw19.txt'
global_statistics = np.loadtxt( file_name ).T 
z, T0, rho_mean, HI_mean, HeI_mean, HeII_mean  = global_statistics
H_mean = X * rho_mean
He_mean = (1-X) * rho_mean
HII_mean = H_mean - HI_mean
HeIII_mean = He_mean - HeI_mean - HeII_mean

label_0 = 'Model'
label_1 = 'Simulation'


label_size = 17
legend_size = 13
fig_dpi = 300

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=True )
ax.plot( z_vals, solution['temperature'], label=label_0)
ax.plot( z, T0, label=label_1 )
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel( r'$z$', fontsize=label_size )
ax.set_ylabel( r'$T_0  \,\,\, [K]$', fontsize=label_size )
ax.legend(loc=1, frameon=False, fontsize=legend_size)
ax.tick_params(axis='both', which='major',  direction='in' )
ax.tick_params(axis='both', which='minor',  direction='in')

fig_name = output_dir + 'temp_evolution.png'
fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi )
print('Saved Figure: ' + fig_name)


fig, ax_l = plt.subplots(nrows=3, ncols=1, figsize=(8,4*3), sharex=True )
plt.subplots_adjust( hspace = 0.0, wspace=0.02)
ax = ax_l[0]
ax.plot( z_vals, solution['n_HII'] / solution['n_H'], lw=2, label=label_0 )
ax.plot( z,  HII_mean / H_mean, "--", label=label_1 )
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel( 'HII Fraction', fontsize=label_size )
# ax.set_xlabel( r'$z$' )
ax.legend(loc=1, frameon=False, fontsize=legend_size)
ax.tick_params(axis='both', which='major',  direction='in' )
ax.tick_params(axis='both', which='minor',  direction='in')

ax = ax_l[1]
ax.plot( z_vals, solution['n_HeII'] / solution['n_He'], lw=2, label=label_0 )
ax.plot( z,  HeII_mean / He_mean, "--", label=label_1 )
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel( 'HeII Fraction', fontsize=label_size )
# ax.set_xlabel( r'$z$' )
ax.legend(loc=1, frameon=False, fontsize=legend_size)
ax.tick_params(axis='both', which='major',  direction='in' )
ax.tick_params(axis='both', which='minor',  direction='in')

ax = ax_l[2]
ax.plot( z_vals, solution['n_HeIII'] / solution['n_He'], lw=2, label=label_0 )
ax.plot( z,  HeIII_mean / He_mean, "--", label=label_1 )
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel( 'HeIII Fraction', fontsize=label_size )
ax.set_xlabel( r'$z$', fontsize=label_size )
ax.legend(loc=1, frameon=False, fontsize=legend_size)
ax.tick_params(axis='both', which='major',  direction='in' )
ax.tick_params(axis='both', which='minor',  direction='in')


fig_name = output_dir + 'ionization_fractions.png'
fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi )
print('Saved Figure: ' + fig_name)
# 
# 
# # ncols = 2
# # nrows = 3
# # fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows), sharex=True )
# # 
# # 
# # ax = ax_l[0][0]
# # ax.plot( z_vals, rates_all['ionization']['HI'] )
# # ax.set_yscale('log')
# # 
# # ax = ax_l[1][0]
# # ax.plot( z_vals, rates_all['ionization']['HeI'] )
# # ax.set_yscale('log')
# # 
# # ax = ax_l[2][0]
# # ax.plot( z_vals, rates_all['ionization']['HeII'] )
# # ax.set_yscale('log')
# # 
# # 
# # 
# # ax = ax_l[0][1]
# # ax.plot( z_vals, rates_all['heating']['HI'] )
# # ax.set_yscale('log')
# # 
# # ax = ax_l[1][1]
# # ax.plot( z_vals, rates_all['heating']['HeI'] )
# # ax.set_yscale('log')
# # 
# # ax = ax_l[2][1]
# # ax.plot( z_vals, rates_all['heating']['HeII'] )
# # ax.set_yscale('log')
# # 
# # 
# # fig_name = output_dir + 'uvb_rates.png'
# # fig.savefig( fig_name, bbox_inches='tight', dpi=300 )
# # 
# # 
# # 
# 
# 
# 
# 
# 
# 
# # types = [ 'ionization', 'heating' ]
# # species = ['HI', 'HeI', 'HeII']
# # rates_all = {}
# # for type in types:
# #   rates_all[type] = {}
# #   for chem in species:
# #     rates_all[type][chem] = []
# # current_z = z_start
# # current_rates = interpolate_rates( current_z, uvb_rates )
# # rates_all = {}
# # for type in types:
# #   rates_all[type] = {}
# #   for chem in species:
# #     rates_all[type][chem] = []
# # 
# # for type in types:
# #   for chem in species:
# #     rates_all[type][chem].append(current_rates[type][chem])    
# # 
# 
# # for type in types:
# #   for chem in species:
# #     rates_all[type][chem].append(current_rates[type][chem])
