import numpy as np



  


def Reaplace_Gamma_Parttial( z, gamma, change_z, change_gamma ):
  ind_sort = np.argsort( change_z )
  change_z = change_z[ind_sort]
  change_gamma = change_gamma[ind_sort]
  r_zmin, r_zmax =  change_z[0], change_z[-1]
  indices = np.where( (z>=r_zmin) * (z<=r_zmax) == True )
  original_z = z[indices] 
  gamma_replce = np.interp( original_z, change_z, change_gamma )
  gamma_new = gamma.copy()
  gamma_new[indices] = gamma_replce
  indx_last = indices[0][-1]
  gamma_new[indx_last] = np.sqrt( gamma_new[indx_last-1]*gamma_new[indx_last+1])
  return gamma_new

def Modify_UVB_Rates_extended( parameters, uvb_rates_in, print_out=False ):  
  # Copy Input Rates to local arrays  
  types = [ 'ionization', 'heating' ]
  species = ['HI', 'HeI', 'HeII']
  uvb_rates = {}
  uvb_rates['z'] = {}
  for chem in species:
    uvb_rates['z'][chem] = uvb_rates_in['z'].copy() 
  for type in types:
    uvb_rates[type] = {}
    for chem in species:
      uvb_rates[type][chem] = uvb_rates_in[type][chem].copy() 
  
  skip_parameters = [ 'wdm_mass' ]
  
  info = 'Rates for '
  for p_name in parameters.keys():
    if p_name in skip_parameters: continue
    p_val = parameters[p_name]
    info += f' {p_name}:{p_val}' 
    
    if p_name == 'scale_H_ion':
      uvb_rates['ionization']['HI']  *= p_val
      uvb_rates['ionization']['HeI'] *= p_val
    
    elif p_name == 'scale_H_Eheat':
      if 'scale_H_ion' in parameters: scale_H_ion = parameters['scale_H_ion']
      else:
        scale_H_ion = 1 
        print('WARNING: Using scale_H_ion = 1 for computing modified heating rates by rescaling the phoelectron energy')
      scale_H_heat = scale_H_ion * p_val
      uvb_rates['heating']['HI']  *= scale_H_heat
      uvb_rates['heating']['HeI'] *= scale_H_heat
    
    elif p_name == 'deltaZ_H':
      uvb_rates['z']['HI']  += p_val
      uvb_rates['z']['HeI'] += p_val
    
    else:
      print( f'ERROR: Modification not defined for parameter: {p_name}')
    
  if print_out: print( info )
  return uvb_rates



def Modify_UVB_Rates( parameters, uvb_rates_in ):  
  # Copy Input Rates to local arrays  
  types = [ 'ionization', 'heating' ]
  species = ['HI', 'HeI', 'HeII']
  uvb_rates = {}
  uvb_rates['z'] = {}
  for chem in species:
    uvb_rates['z'][chem] = uvb_rates_in['z'].copy() 
  for type in types:
    uvb_rates[type] = {}
    for chem in species:
      uvb_rates[type][chem] = uvb_rates_in[type][chem].copy() 
  
  HI_rates_factor = parameters['scale_H']
  HeII_rates_factor = parameters['scale_He']
  delta_z_HI = parameters['delta_z_H']
  delta_z_HeII = parameters['delta_z_He']
  
  # Rescalke the Photoionization and Photoheating rates
  uvb_rates['ionization']['HI']   *= HI_rates_factor
  uvb_rates['heating']['HI']      *= HI_rates_factor 
  uvb_rates['ionization']['HeI']  *= HI_rates_factor
  uvb_rates['heating']['HeI']     *= HI_rates_factor 
  uvb_rates['ionization']['HeII'] *= HeII_rates_factor
  uvb_rates['heating']['HeII']    *= HeII_rates_factor 
  uvb_rates['z']['HI'] += delta_z_HI
  uvb_rates['z']['HeI'] += delta_z_HI
  uvb_rates['z']['HeII'] += delta_z_HeII
  return uvb_rates
  
def interpolate_rates( current_z, rates, log=True ):
  rate_min = 1e-50

  current_rates = {}
  types = [ 'ionization', 'heating' ]
  species = ['HI', 'HeI', 'HeII']
  for type in types:
    current_rates[type] = {}
    for chem in species:
      z_vals = rates['z'][chem]
      values = rates[type][chem]
      if log: values = np.log10(values)
      if current_z > z_vals.max()  : current_val = rate_min
      elif current_z < z_vals.min() : current_val = values[0]
      else: 
        diff = np.abs( z_vals - current_z )
        indx = np.where( diff == diff.min() )
        z_interp = current_z
        current_val = np.interp( z_interp, z_vals, values )
        # current_val = values[indx]
        if log: current_val = 10**current_val
      current_rates[type][chem] = current_val
  # print( f"Rates HI:  {current_rates['ionization']['HI']:.6e}   {current_rates['heating']['HI']:.6e}   ")
  return current_rates