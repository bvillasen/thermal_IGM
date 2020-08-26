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
from temp_functions import interpolate_rates
from cooling_rates_Katz95 import *
from cooling_rates import *

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
n_samples = 50000
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
a_vals = np.linspace( a_start, a_end, n_samples)
da_vals = a_vals[1:] - a_vals[:-1]
z_vals = 1/a_vals - 1
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


def append_state_to_solution( fields, current_state, solution ):
  for field in fields:
    solution[field].append(current_state[field])

def get_state_array( fields, current_state ):
  state_array = np.array([ current_state[field] for field in fields ])
  return state_array

def update_current_state( fields, state_array, current_state ):
  n_arr = len(current_state)
  n_fields = len(fields)
  if n_arr != n_fields: print('ERROR Updating state')
  for i in range(n_fields):
    field = fields[i]
    current_state[field] = state_array[i]
  return current_state


def delta_n_hubble( n, time, current_a, cosmo ):
  a_dot = cosmo.a_deriv( time, current_a )
  dn_dt = - 3 * n * current_a**(-4) * a_dot
  dn_dt = 0
  return dn_dt 

def all_deriv( time, state_array, kargs=None ):
  cosmo = kargs['cosmo']
  uvb_rates = kargs['uvb_rates']
  temp, n_H, n_HI, n_HII, n_He, n_HeI, n_HeII, n_HeIII, n_e = state_array 
  n_min =  1e-60
  if n_HI < n_min: n_HI = n_min
  if n_HII < n_min : n_HII = n_min
  if n_HeI < n_min : n_HeI = n_min
  if n_HeII < n_min : n_HeII = n_min
  if n_HeIII < n_min : n_HeIII = n_min
  if n_e < n_min : n_e = n_min
  n_tot = n_HI + n_HII + n_HeI + n_HeII + n_HeIII + n_e
  # n_tot = n_H + n_He + n_e
  
  # Get Hubble 
  current_a = cosmo.get_current_a( time )
  current_z = 1./current_a - 1
  # print( current_z )
  H = cosmo.get_Hubble( current_a ) * 1000 / Mpc  # 1/sec
  dT_dt_hubble =   -2 * H * temp 
  
  # Get Current photoheating and photoionization rates
  current_rates = interpolate_rates( current_z, uvb_rates, log=False )
  photoheating_rates = current_rates['heating']
  photoionization_rates = current_rates['ionization']
  
  
  # Photoheating
  Q_phot_HI = n_HI * photoheating_rates['HI'] 
  Q_phot_HeI = n_HeI * photoheating_rates['HeI'] 
  Q_phot_HeII = n_HeII * photoheating_rates['HeII'] 
  dQ_dt_phot = Q_phot_HI + Q_phot_HeI + Q_phot_HeII 
  
  # Recopmbination Cooling
  Q_cool_rec_HII    = Cooling_Rate_Recombination_HII_Hui97( n_e, n_HII, temp )
  Q_cool_rec_HeII   = Cooling_Rate_Recombination_HeII_Hui97( n_e, n_HeII, temp )
  Q_cool_rec_HeII_d = Cooling_Rate_Recombination_dielectronic_HeII_Hui97( n_e, n_HeII, temp )
  Q_cool_rec_HeIII  = Cooling_Rate_Recombination_HeIII_Hui97( n_e, n_HeIII, temp )
  dQ_dt_cool_recomb = Q_cool_rec_HII + Q_cool_rec_HeII + Q_cool_rec_HeII_d + Q_cool_rec_HeIII 
  
  # Collisional Cooling
  Q_cool_collis_ext_HI = Cooling_Rate_Collisional_Excitation_e_HI_Hui97( n_e, n_HI, temp ) 
  Q_cool_collis_ext_HeII = Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( n_e, n_HeII, temp )
  Q_cool_collis_ion_HI = Cooling_Rate_Collisional_Ionization_e_HI_Katz95( n_e, n_HI, temp )
  Q_cool_collis_ion_HeI = Cooling_Rate_Collisional_Ionization_e_HeI_Katz95( n_e, n_HeI, temp )
  Q_cool_collis_ion_HeII = Cooling_Rate_Collisional_Ionization_e_HeII_Katz95( n_e, n_HeII, temp )
  dQ_dt_cool_collisional = Q_cool_collis_ext_HI + Q_cool_collis_ext_HeII + Q_cool_collis_ion_HI + Q_cool_collis_ion_HeI + Q_cool_collis_ion_HeII
  
  # Cooling Bremsstrahlung 
  dQ_dt_brem = Cooling_Rate_Bremsstrahlung_Katz95( n_e, n_HII, n_HeII, n_HeIII, temp )
  
  #Compton cooling off the CMB 
  dQ_dt_CMB = Cooling_Rate_Compton_CMB_MillesOstriker01( n_e, temp, current_z )
  # dQ_dt_CMB = Cooling_Rate_Compton_CMB_Katz95( n_e, temp, current_z ) 
  
  # Heating and Cooling Rates 
  dQ_dt  = dQ_dt_phot - dQ_dt_cool_recomb - dQ_dt_cool_collisional - dQ_dt_brem - dQ_dt_CMB
  
  # Photoionization Rates
  dn_dt_photo_HI = n_HI * photoionization_rates['HI']
  dn_dt_photo_HeI = n_HeI * photoionization_rates['HeI']
  dn_dt_photo_HeII = n_HeII * photoionization_rates['HeII']
  
  # Recombination Rates 
  T = temp
  dn_dt_recomb_HII = Recombination_Rate_HII_Hui97( T ) * n_HII * n_e  
  dn_dt_recomb_HeII = Recombination_Rate_HeII_Hui97( T ) * n_HeII * n_e
  dn_dt_recomb_HeIII = Recombination_Rate_HeIII_Katz95( T ) * n_HeIII * n_e
  dn_dt_recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( T ) * n_HeII * n_e
  
  # Collisional Ionization  Rates
  dn_dt_coll_HI = Collisional_Ionization_Rate_e_HI_Hui97( T ) * n_HI * n_e 
  dn_dt_coll_HeI = Collisional_Ionization_Rate_e_HeI_Abel97( T ) * n_HeI * n_e 
  dn_dt_coll_HeII = Collisional_Ionization_Rate_e_HeII_Abel97( T ) * n_HeII * n_e
  dn_dt_coll_HI_HI = Collisional_Ionization_Rate_HI_HI_Lenzuni91( T ) * n_HI * n_HI
  dn_dt_coll_HII_HI = Collisional_Ionization_Rate_HII_HI_Lenzuni91( T ) * n_HII * n_HI
  
  
  dn_dt_HI_p  = dn_dt_recomb_HII 
  dn_dt_HI_m  = dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI
  
  dn_dt_HII_p = dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI
  dn_dt_HII_m = dn_dt_recomb_HII 
  
  dn_dt_HeI_p = dn_dt_recomb_HeII + dn_dt_recomb_HeII_d
  dn_dt_HeI_m = dn_dt_photo_HeI + dn_dt_coll_HeI
  
  dn_dt_HeII_p = dn_dt_photo_HeI + dn_dt_coll_HeI + dn_dt_recomb_HeIII
  dn_dt_HeII_m = dn_dt_photo_HeII + dn_dt_coll_HeII + dn_dt_recomb_HeII + dn_dt_recomb_HeII_d
  
  dn_dt_HeIII_p = dn_dt_photo_HeII + dn_dt_coll_HeII 
  dn_dt_HeIII_m =  dn_dt_recomb_HeIII 
  
  dn_dt_e_p = dn_dt_photo_HI + dn_dt_photo_HeI + dn_dt_photo_HeII + dn_dt_coll_HI + dn_dt_coll_HeI + dn_dt_coll_HeII
  dn_dt_e_m = dn_dt_recomb_HII + dn_dt_recomb_HeII + dn_dt_recomb_HeII_d + dn_dt_recomb_HeIII
  
    
  d_T_dt = dT_dt_hubble +  2./(3*K_b*n_tot)*dQ_dt
  n_hubble_factor = -3 * H   
  d_nH_dt     = n_hubble_factor * n_H 
  d_nHe_dt    = n_hubble_factor * n_He  
  d_nHI_dt    = n_hubble_factor * n_HI    + dn_dt_HI_p - dn_dt_HI_m 
  d_nHII_dt   = n_hubble_factor * n_HII   + dn_dt_HII_p - dn_dt_HII_m
  d_nHeI_dt   = n_hubble_factor * n_HeI   + dn_dt_HeI_p - dn_dt_HeI_m
  d_nHeII_dt  = n_hubble_factor * n_HeII  + dn_dt_HeII_p - dn_dt_HeII_m
  d_nHeIII_dt = n_hubble_factor * n_HeIII + dn_dt_HeIII_p - dn_dt_HeIII_m
  d_ne_dt     = n_hubble_factor * n_e     + dn_dt_e_p - dn_dt_e_m
  
  return np.array([ d_T_dt, d_nH_dt,  d_nHI_dt, d_nHII_dt, d_nHe_dt,  d_nHeI_dt, d_nHeII_dt, d_nHeIII_dt,  d_ne_dt  ])

import timeit

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
