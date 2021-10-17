import os
import numpy as np
import h5py as h5
from cosmo_constants import eV_to_ergs

def Load_Grackle_UVB_File( file_name ):
  file = h5.File( file_name, 'r' )

  rates = file['UVBRates']
  info = rates['Info'][...]
  z = rates['z'][...]

  rates_out = {}
  rates_out['z'] = z

  rates_out['ionization'] = {}
  chemistry = rates['Chemistry']
  rates_out['ionization']['HI']   = chemistry['k24'][...]
  rates_out['ionization']['HeI']  = chemistry['k26'][...]
  rates_out['ionization']['HeII'] = chemistry['k25'][...]

  rates_out['heating'] = {}
  heating = rates['Photoheating'] 
  rates_out['heating']['HI']   = heating['piHI'][...] * eV_to_ergs
  rates_out['heating']['HeI']  = heating['piHeI'][...] * eV_to_ergs
  rates_out['heating']['HeII'] = heating['piHeII'][...] * eV_to_ergs
  return rates_out
