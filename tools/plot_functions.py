import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import pylab
import matplotlib
# set some global options
matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'



colors_data = [ 'C1', 'C2' ]
font_size = 16


def Plot_HI_fraction( output_dir, solution=None, input_file=None, solutions=None, file_name='HI_fraction.png', HI_data=None, data_labels=None ):
  if input_file: 
    file = h5.File( input_file, 'r' )
    solution = { key:file[key][...] for key in file }
    file.close

  if solution is not None:
    z = solution['z']
    temperature = solution['temperature']
    HI_frac   = solution['n_HI'] / solution['n_H']
  
  ncols, nrows = 1, 1
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,5*nrows), sharex=True)
  plt.subplots_adjust( hspace = 0.03, wspace=0.02)


  if input_file or solution is not None: ax.plot( z, HI_frac, label='Modified P19: One cell model ' )
  
  if solutions:
    for model_id in solutions:
      solution = solutions[model_id]
      z = solution['z']
      temperature = solution['temperature']
      HI_frac   = solution['n_HI'] / solution['n_H']
      if 'label' in solution: label = solution['label']
      else: label = ''
      ax.plot( z, HI_frac, lw=0.3, label=label )
  
  ax.set_yscale('log')
  
  if HI_data:
    for i in HI_data:
      data = HI_data[i]
      z_data = data['z']
      HI_frac_data = data['mean']
      color = colors_data[i]
      label = data_labels[i]
      if 'yerr' in data:
        yerr = data['yerr']
        ax.errorbar( z_data, HI_frac_data, yerr=yerr, color=color, label=label, fmt='o' )
      else: ax.scatter( z_data, HI_frac_data, color=color, label=label )
      
  
  ax.set_xlim( 2, 8)
  
  ax.legend( loc=2, frameon=False)
  ax.set_xlabel( r'$z$', fontsize=font_size)
  ax.set_ylabel( 'HI Fraction', fontsize=font_size)

  fig_name = output_dir + file_name
  fig.savefig( fig_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
  print('Saved Figure: ' + fig_name)

  

def Plot_Solution( output_dir, solution=None, input_file=None, file_name='solution.png', multiple=False ):
  
  
  if not solution: 
    file = h5.File( input_file, 'r' )
    solution = { key:file[key][...] for key in file }
    file.close()

  if not multiple: solutions = { 0: solution }
  else: solutions = solution
  
   


  ncols, nrows = 1, 3
  fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,3*nrows), sharex=True)
  plt.subplots_adjust( hspace = 0.03, wspace=0.02)
  
  for data_id in solutions:
    solution = solutions[data_id]
    
    z = solution['z']
    temperature = solution['temperature']
    HI_frac    = solution['n_HI'] / solution['n_H']
    HII_frac   = solution['n_HII'] / solution['n_H']
    HeIII_frac = solution['n_HeIII'] / solution['n_He']
    
    ls = '-'
    if 'line_style' in solution: ls = solution['line_style']
    
    label = ''
    if 'label' in solution: label = solution['label']

    ax = ax_l[0]
    ax.plot( z, temperature, ls=ls, label=label )

    ax = ax_l[1]
    ax.plot( z, HI_frac, ls=ls, label=label )

    ax = ax_l[2]
    ax.plot( z, HeIII_frac, ls=ls, label=label )
  
  ax = ax_l[0]
  ax.legend( frameon=False )


  fig_name = output_dir + file_name
  fig.savefig( fig_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
  print('Saved Figure: ' + fig_name)







