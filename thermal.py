import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5



thermal_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(thermal_dir)]
sys.path.extend(subDirectories)
from tools import create_directory
from cosmology import Cosmology
from cosmo_constants import Myear, Mpc, K_b, M_p, Gcosmo    
from load_rates_pchw19 import rates_pchw19, rates_pchw19_eq
from temp_functions import Integrate_Evolution


import matplotlib
# set some global options
matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
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


# Set photoheating and photoionization rates
uvb_rates = rates_pchw19

a_start = 1. / (z_start + 1)
rho_cgs = rho_gas_mean * 1e3 / a_start**3 
print( rho_cgs)

H = cosmo.get_Hubble( a_start ) * 1000 / Mpc  # 1/sec

scale_rates = ''
delta_z_rates = ''
solutions = []

if delta_z_rates == 'HI':
  delta_z_rates_list = [ -0.5, -0.25, 0.0, 0.25, 0.5, 0.75 ]
  delta_z_HeII = 0.0
  for delta_z_HI in delta_z_rates_list:
    print( "Delta_z HI: {0:0.1f}".format( delta_z_HI ))
    solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, delta_z_HI=delta_z_HI  )
    solutions.append(solution)

if delta_z_rates == 'HeII':
  delta_z_rates_list = [  -0.25, -0.12, 0.0, 0.12, 0.25, 0.4 ]
  delta_z_HeII = 0.0
  for delta_z_HeII in delta_z_rates_list:
    print( "Delta_z HeII: {0:0.1f}".format( delta_z_HeII ))
    solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, delta_z_HeII=delta_z_HeII  )
    solutions.append(solution)


# Solve the Thermal Evolution
if scale_rates == 'HI': 
  HI_rates_factor_list = [ 2.0, 1.5, 1.25, 1.0, 0.75,  0.5  ]
  HeI_rates_factor = 1.0
  HeII_rates_factor = 1.0
  for HI_rates_factor in HI_rates_factor_list:
    print( "HI factor: {0:0.1f}".format( HI_rates_factor ))
    solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, HI_rates_factor=HI_rates_factor, HeI_rates_factor=HeI_rates_factor, HeII_rates_factor=HeII_rates_factor,  )
    solutions.append(solution)

if scale_rates == 'HeI': 
  HeI_rates_factor_list = [ 2.0, 1.5, 1.25, 1.0, 0.75, 0.5    ]
  HI_rates_factor = 1.0
  HeII_rates_factor = 1.0
  for HeI_rates_factor in HeI_rates_factor_list:
    print( "HeI factor: {0:0.1f}".format( HeII_rates_factor ))
    solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, HI_rates_factor=HI_rates_factor, HeI_rates_factor=HeI_rates_factor, HeII_rates_factor=HeII_rates_factor,  )
    solutions.append(solution)

if scale_rates == 'HeII': 
  HeII_rates_factor_list = [ 2.0, 1.5, 1.25, 1.0, 0.75, 0.5    ]
  HI_rates_factor = 1.0
  HeI_rates_factor = 1.0
  for HeII_rates_factor in HeII_rates_factor_list:
    print( "HeII factor: {0:0.1f}".format( HeII_rates_factor ))
    solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, HI_rates_factor=HI_rates_factor, HeI_rates_factor=HeI_rates_factor, HeII_rates_factor=HeII_rates_factor,  )
    solutions.append(solution)

solution = Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples )
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



data_dir = '/home/bruno/Desktop/ssd_0/data/'
input_dir = data_dir + 'cosmo_sims/chemistry_test/output_files_zSmall/'
z_list, temp_list = [], []
for nSnap in range( 170 ):
  in_file_name = input_dir + '{0}.h5'.format(nSnap)
  inFile = h5.File( in_file_name, 'r' )
  current_z = inFile.attrs['Current_z'][0]
  dens = inFile['density'][...][0,0,0]
  temp = inFile['temperature'][...][0,0,0]
  z_list.append( current_z )
  temp_list.append( temp )
z_vals_grackle = np.array( z_list )
temp_vals_grackle = np.array( temp_list )  

input_dir = data_dir + 'cosmo_sims/chemistry_test/output_files_noZ/'
z_list, temp_list = [], []
for nSnap in range( 170 ):
  in_file_name = input_dir + '{0}.h5'.format(nSnap)
  inFile = h5.File( in_file_name, 'r' )
  current_z = inFile.attrs['Current_z'][0]
  dens = inFile['density'][...][0,0,0]
  temp = inFile['temperature'][...][0,0,0]
  z_list.append( current_z )
  temp_list.append( temp )
z_vals_grackle_noZ = np.array( z_list )
temp_vals_grackle_noZ = np.array( temp_list )  


label_0 = 'Bruno single cell'
label_1 = 'CHIPS.P19'


label_size = 17
legend_size = 13
fig_dpi = 300
border_width = 1


black_background = False

import pylab

C_0 = 'C1'
C_1 = pylab.cm.viridis(.7)
c_2 = pylab.cm.cool(.3)

text_color = 'black'
if black_background: text_color = 'white'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=True )
if black_background: fig.patch.set_facecolor('black')   
if black_background: ax.set_facecolor('k')
ax.plot( z, T0, label=label_1, C=C_1 )
ax.plot( solution['z'], solution['temperature'], label=label_0,  C=C_0)
ax.plot( z_vals_grackle, temp_vals_grackle, label=r'Grackle single cell $Z=10^{-6}$', c=c_2)
ax.plot( z_vals_grackle_noZ, temp_vals_grackle_noZ, '--', label=r'Grackle single cell $Z=0$', c='C4')
ax.set_xlabel( r'$z$', fontsize=label_size, color=text_color )
ax.set_ylabel( r'$T_0  \,\,\, [\mathrm{K}]$', fontsize=label_size, color=text_color )
leg=ax.legend(loc=1, frameon=False, fontsize=legend_size)
for text in leg.get_texts():
  plt.setp(text, color = text_color)
for spine in list(ax.spines.values()):
    spine.set_edgecolor(text_color)
ax.tick_params(axis='both', which='major',  direction='in', color=text_color, labelcolor=text_color )
ax.tick_params(axis='both', which='minor',  direction='in', color=text_color, labelcolor=text_color)
[sp.set_linewidth(border_width) for sp in ax.spines.values()]
ax.set_xlim( 0, 16 )
fig_name = output_dir + 'temp_evolution.png'
fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi, facecolor=fig.get_facecolor() )
print('Saved Figure: ' + fig_name)



colors = [ 'C0', 'C1', 'C2', 'C3', 'C4', 'C9']
if scale_rates in [ 'HI', 'HeI', 'HeII',]:
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=True )
  if black_background: fig.patch.set_facecolor('black')
  if black_background: ax.set_facecolor('k')   
  if scale_rates == 'HI':
    rates = HI_rates_factor_list
    name_label = r"$\beta_{\mathrm{HI}} = $"
  if scale_rates == 'HeI':
    rates = HeI_rates_factor_list
    name_label = r"$\beta_{\mathrm{HeI}} = $"
  if scale_rates == 'HeII':
    rates = HeII_rates_factor_list
    name_label = r"$\beta_{\mathrm{HeII}} = $"

  for i, solution in enumerate(solutions):
    rate_factor = rates[i]
    label = name_label + " {0:.2f}".format(rate_factor) 
    color = colors[i]
    ax.plot( solution['z'], solution['temperature'],  label=label, c=color )
  ax.set_xlabel( r'$z$', fontsize=label_size, color=text_color )
  ax.set_ylabel( r'$T_0  \,\,\, [\mathrm{K}]$', fontsize=label_size, color=text_color )
  leg = ax.legend(loc=1, frameon=False, fontsize=legend_size)
  for text in leg.get_texts():
    plt.setp(text, color = text_color)

  ax.tick_params(axis='both', which='major',  direction='in', color=text_color, labelcolor=text_color )
  ax.tick_params(axis='both', which='minor',  direction='in', color=text_color, labelcolor=text_color)
  [sp.set_linewidth(border_width) for sp in ax.spines.values()]
  for spine in list(ax.spines.values()):
      spine.set_edgecolor(text_color)
  fig_name = output_dir + 'temp_evolution_{0}_factor.png'.format(scale_rates)
  fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi, facecolor=fig.get_facecolor() )
  print('Saved Figure: ' + fig_name)


if delta_z_rates in [ 'HI', 'HeI', 'HeII',]:
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6), sharex=True )
  if black_background: fig.patch.set_facecolor('black')
  if black_background: ax.set_facecolor('k')   
  if delta_z_rates == 'HI':
    name_label = r"$\Delta z_{\mathrm{HI}} = $"
  if delta_z_rates == 'HeII':
    name_label = r"$\Delta z_{\mathrm{HeII}} = $"

  for i, solution in enumerate(solutions):
    delta_z = delta_z_rates_list[i]
    label = name_label + " {0:.2f}".format(delta_z) 
    color = colors[i]
    ax.plot( solution['z'], solution['temperature'],  label=label, c=color )
  ax.set_xlabel( r'$z$', fontsize=label_size, color=text_color )
  ax.set_ylabel( r'$T_0  \,\,\, [\mathrm{K}]$', fontsize=label_size, color=text_color )
  leg = ax.legend(loc=1, frameon=False, fontsize=legend_size)
  for text in leg.get_texts():
    plt.setp(text, color = text_color)

  ax.tick_params(axis='both', which='major',  direction='in', color=text_color, labelcolor=text_color )
  ax.tick_params(axis='both', which='minor',  direction='in', color=text_color, labelcolor=text_color)
  [sp.set_linewidth(border_width) for sp in ax.spines.values()]
  for spine in list(ax.spines.values()):
      spine.set_edgecolor(text_color)
  fig_name = output_dir + 'temp_evolution_{0}_delta_z.png'.format(delta_z_rates)
  fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi, facecolor=fig.get_facecolor() )
  print('Saved Figure: ' + fig_name)

# 
# 
# fig, ax_l = plt.subplots(nrows=3, ncols=1, figsize=(8,4*3), sharex=True )
# plt.subplots_adjust( hspace = 0.0, wspace=0.02)
# ax = ax_l[0]
# ax.plot( solution['z'], solution['n_HII'] / solution['n_H'], lw=2, label=label_0 )
# ax.plot( z,  HII_mean / H_mean, "--", label=label_1 )
# ax.set_ylim(-0.05, 1.05)
# ax.set_ylabel( 'HII Fraction', fontsize=label_size )
# # ax.set_xlabel( r'$z$' )
# ax.legend(loc=1, frameon=False, fontsize=legend_size)
# ax.tick_params(axis='both', which='major',  direction='in' )
# ax.tick_params(axis='both', which='minor',  direction='in')
# 
# ax = ax_l[1]
# ax.plot( solution['z'], solution['n_HeII'] / solution['n_He'], lw=2, label=label_0 )
# ax.plot( z,  HeII_mean / He_mean, "--", label=label_1 )
# ax.set_ylim(-0.05, 1.05)
# ax.set_ylabel( 'HeII Fraction', fontsize=label_size )
# # ax.set_xlabel( r'$z$' )
# ax.legend(loc=1, frameon=False, fontsize=legend_size)
# ax.tick_params(axis='both', which='major',  direction='in' )
# ax.tick_params(axis='both', which='minor',  direction='in')
# 
# ax = ax_l[2]
# ax.plot( solution['z'], solution['n_HeIII'] / solution['n_He'], lw=2, label=label_0 )
# ax.plot( z,  HeIII_mean / He_mean, "--", label=label_1 )
# ax.set_ylim(-0.05, 1.05)
# ax.set_ylabel( 'HeIII Fraction', fontsize=label_size )
# ax.set_xlabel( r'$z$', fontsize=label_size )
# ax.legend(loc=1, frameon=False, fontsize=legend_size)
# ax.tick_params(axis='both', which='major',  direction='in' )
# ax.tick_params(axis='both', which='minor',  direction='in')
# 
# 
# fig_name = output_dir + 'ionization_fractions.png'
# fig.savefig( fig_name, bbox_inches='tight', dpi=fig_dpi )
# print('Saved Figure: ' + fig_name)
# 
# 
