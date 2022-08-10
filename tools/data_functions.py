import numpy as np
import h5py as h5


def Write_Solution( solution, output_file_name, n_stride=1, fields_to_write=None, print_out=True ):
  file = h5.File( output_file_name, 'w' )
  if fields_to_write is None: fields_to_write = solution.keys()
  for key in fields_to_write:
    file.create_dataset( key, data=solution[key][::n_stride] )
  file.close() 
  if print_out: print( f'Saved File: {output_file_name}' )
