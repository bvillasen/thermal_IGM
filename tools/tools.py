import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np

system = None
system = os.getenv('SYSTEM_NAME')
if not system:
  print( 'Can not find the system name')
  exit(-1)
print( f'System: {system}')

if system == 'Eagle':    data_dir = '/home/bruno/Desktop/data/'
if system == 'Tornado':  data_dir = '/home/bruno/Desktop/ssd_0/data/'
if system == 'Shamrock': data_dir = '/raid/bruno/data/'
if system == 'Lux':      data_dir = '/data/groups/comp-astro/bruno/'
if system == 'Summit':   data_dir = '/gpfs/alpine/csc434/scratch/bvilasen/'
if system == 'Mac_mini': data_dir = '/Users/bruno/Desktop/data/'
if system == 'MacBook':  data_dir = '/Users/bruno/Desktop/data/'

def split_array_mpi( array, rank, n_procs, adjacent=False ):
  n_index_total = len(array)
  n_proc_indices = (n_index_total-1) // n_procs + 1
  indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  if adjacent: indices_to_generate = np.array([ i + rank*n_proc_indices for i in range(n_proc_indices) ])
  else: indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  indices_to_generate = indices_to_generate[ indices_to_generate < n_index_total ]
  return array[indices_to_generate]

def split_indices( indices, rank, n_procs, adjacent=False ):
  n_index_total = len(indices)
  n_proc_indices = (n_index_total-1) // n_procs + 1
  indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  if adjacent: indices_to_generate = np.array([ i + rank*n_proc_indices for i in range(n_proc_indices) ])
  else: indices_to_generate = np.array([ rank + i*n_procs for i in range(n_proc_indices) ])
  indices_to_generate = indices_to_generate[ indices_to_generate < n_index_total ]
  return indices_to_generate

def Combine_List_Pair( a, b ):
  output = []
  for a_i in a:
    for b_i in b:
      if type(b_i) == list:
        add_in = [a_i] + b_i
      else:
        add_in = [ a_i, b_i]
      output.append( add_in )
  return output

def Get_Parameters_Combination( param_vals ):
  n_param = len( param_vals )
  indices_list = []
  for i in range(n_param):
    param_id = n_param - 1 - i
    n_vals =  len(param_vals[param_id]) 
    indices_list.append( [ x for x in range(n_vals)] )
  param_indx_grid = indices_list[0]
  for i in range( n_param-1 ):
    param_indx_grid = Combine_List_Pair( indices_list[i+1], param_indx_grid )
  param_combinations = []
  for param_indices in param_indx_grid:
    p_vals = [ ]
    for p_id, p_indx in enumerate(param_indices):
      p_vals.append( param_vals[p_id][p_indx] )
    param_combinations.append( p_vals )
  return param_combinations
  
def Load_Pickle_Directory( input_name ):
  import pickle
  print( f'Loading File: {input_name}')
  dir = pickle.load( open( input_name, 'rb' ) )
  return dir
  
def Write_Pickle_Directory( dir, output_name ):
  import pickle 
  f = open( output_name, 'wb' )
  pickle.dump( dir, f)
  print ( f'Saved File: {output_name}' )

def print_progress( i, n, time_start, extra_line="" ):
  import time
  time_now = time.time()
  time = time_now - time_start
  if i == 0: remaining = time *  n
  else: remaining = time * ( n - i ) / i

  hrs = remaining // 3600
  min = (remaining - hrs*3600) // 60
  sec = remaining - hrs*3600 - min*60
  etr = f'{hrs:02.0f}:{min:02.0f}:{sec:02.0f}'
  progres = f'{extra_line}Progress:   {i}/{n}   {i/n*100:.1f}%   ETR: {etr} '
  print_line_flush (progres )


def print_line_flush( terminalString ):
  terminalString = '\r' + terminalString
  sys.stdout. write(terminalString)
  sys.stdout.flush() 


def printProgress( current, total,  deltaTime, print_str='' ):
  terminalString = "\rProgress: "
  if total==0: total+=1
  percent = 100.*current/total
  nDots = int(percent/5)
  dotsString = "[" + nDots*"." + (20-nDots)*" " + "]"
  percentString = "{0:.0f}%".format(percent)
  ETR = deltaTime/(current+1)*(total - current)
  hours = int(ETR/3600)
  minutes = int(ETR - 3600*hours)//60
  seconds = int(ETR - 3600*hours - 60*minutes)
  ETRstring = "  ETR= {0}:{1:02}:{2:02}    ".format(hours, minutes, seconds)
  if deltaTime < 0.0001: ETRstring = "  ETR=    "
  terminalString  += dotsString + percentString + ETRstring
  terminalString += print_str
  sys.stdout. write(terminalString)
  sys.stdout.flush() 

def create_directory( dir, print_out=True ):
  if print_out: print(("Creating Directory: {0}".format(dir) ))
  indx = dir[:-1].rfind('/' )
  inDir = dir[:indx]
  dirName = dir[indx:].replace('/','')
  dir_list = next(os.walk(inDir))[1]
  if dirName in dir_list : 
    if print_out:print( " Directory exists")
  else:
    os.mkdir( dir )
    if print_out: print( " Directory created")

