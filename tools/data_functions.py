import numpy as np
import h5py as h5


def Write_Solution( solution, output_file_name, n_stride=1 ):
  file = h5.File( output_file_name, 'w' )
  for key in solution:
    file.create_dataset( key, data=solution[key][::n_stride] )
  file.close() 
  print( f'Saved File: {output_file_name}' )
