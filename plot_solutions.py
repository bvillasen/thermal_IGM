import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
thermal_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(thermal_dir)]
sys.path.extend(subDirectories)
from tools import *
from plot_functions import *

input_dir  = data_dir + 'chemistry_test/'
output_dir = data_dir + 'chemistry_test/figures/'
create_directory( output_dir, print_out=False )

solutions = {}

names = [ 'original', 'rk4', 'bdf', 'bdf_grackle' ]
line_styles = [ '-', '--', '--', '--' ]

for data_id,name in enumerate(names):
  file_name =  input_dir + f'solution_{name}.h5'
  file = h5.File( file_name, 'r' )
  solution = { key:file[key][...] for key in file }
  file.close()
  solutions[data_id] = solution
  solutions[data_id]['line_style'] = line_styles[data_id]
  solutions[data_id]['label'] = name

Plot_Solution( output_dir, solution=solutions, multiple=True )

