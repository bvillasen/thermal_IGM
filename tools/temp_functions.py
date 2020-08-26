import numpy as np
from cosmo_constants import Myear, Mpc, K_b, M_p   
from cooling_rates_Katz95 import *
from cooling_rates import *

rate_min = 1e-100
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
  dn_dt_coll_HeI_HI = Collisional_Ionization_Rate_HeI_HI_Lenzuni91( T ) * n_HeI * n_HI
  
  
  dn_dt_HI_p  = dn_dt_recomb_HII 
  dn_dt_HI_m  = dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI
  
  dn_dt_HII_p = dn_dt_photo_HI + dn_dt_coll_HI + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI
  dn_dt_HII_m = dn_dt_recomb_HII 
  
  dn_dt_HeI_p = dn_dt_recomb_HeII + dn_dt_recomb_HeII_d
  dn_dt_HeI_m = dn_dt_photo_HeI + dn_dt_coll_HeI
  
  dn_dt_HeII_p = dn_dt_photo_HeI + dn_dt_coll_HeI + dn_dt_recomb_HeIII
  dn_dt_HeII_m = dn_dt_photo_HeII + dn_dt_coll_HeII + dn_dt_recomb_HeII + dn_dt_recomb_HeII_d
  
  dn_dt_HeIII_p = dn_dt_photo_HeII + dn_dt_coll_HeII 
  dn_dt_HeIII_m =  dn_dt_recomb_HeIII 
  
  dn_dt_e_p = dn_dt_photo_HI + dn_dt_photo_HeI + dn_dt_photo_HeII + dn_dt_coll_HI + dn_dt_coll_HeI + dn_dt_coll_HeII + dn_dt_coll_HI_HI + dn_dt_coll_HII_HI + dn_dt_coll_HeI_HI
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


# def temp_deriv( time, temp, kargs=None ):
#   H = kargs['H'] * 1000 / Mpc  # 1/sec
#   photoheating_rates = kargs['rates']['heating']
#   chemistry = kargs['chemistry']
#   z = kargs['z']
#   n_HI = chemistry['n_HI']
#   n_HII = chemistry['n_HII']
#   n_HeI = chemistry['n_HeI']
#   n_HeII = chemistry['n_HeII']
#   n_HeIII = chemistry['n_HeIII']
#   n_e = chemistry['n_e']
#   n_tot = n_HI + n_HII + n_HeI + n_HeII + n_HeIII + n_e
# 
#   # Photoheating
#   Q_phot_HI = n_HI * photoheating_rates['HI'] 
#   Q_phot_HeI = n_HeI * photoheating_rates['HeI'] 
#   Q_phot_HeII = n_HeII * photoheating_rates['HeII'] 
#   dQ_dt_phot = Q_phot_HI + Q_phot_HeI + Q_phot_HeII 
#   # dQ_dt_phot *= 0.7
# 
#   # Recopmbination Cooling
#   Q_cool_rec_HII    = Cooling_Rate_Recombination_HII( n_e, n_HII, temp )
#   Q_cool_rec_HeII   = Cooling_Rate_Recombination_HeII( n_e, n_HeII, temp )
#   Q_cool_rec_HeII_d = Cooling_Rate_Recombination_dielectronic_HeII( n_e, n_HeII, temp )
#   Q_cool_rec_HeIII  = Cooling_Rate_Recombination_HeIII( n_e, n_HeIII, temp )
#   dQ_dt_cool_recomb = Q_cool_rec_HII + Q_cool_rec_HeII + Q_cool_rec_HeII_d + Q_cool_rec_HeIII 
# 
#   # Collisional Cooling
#   Q_cool_collis_ext_HI = Cooling_Rate_Collisional_Excitation_e_HI( n_e, n_HI, temp )
#   Q_cool_collis_ext_HeII = Cooling_Rate_Collisional_Excitation_e_HeII( n_e, n_HeII, temp )
#   Q_cool_collis_ion_HI = Cooling_Rate_Collisional_Ionization_e_HI( n_e, n_HI, temp )
#   Q_cool_collis_ion_HeI = Cooling_Rate_Collisional_Ionization_e_HeI( n_e, n_HeI, temp )
#   Q_cool_collis_ion_HeII = Cooling_Rate_Collisional_Ionization_e_HeII( n_e, n_HeII, temp )
#   dQ_dt_cool_collisional = Q_cool_collis_ext_HI + Q_cool_collis_ext_HeII + Q_cool_collis_ion_HI + Q_cool_collis_ion_HeI + Q_cool_collis_ion_HeII
# 
#   # Cooling Bremsstrahlung 
#   dQ_dt_brem = Cooling_Rate_Bremsstrahlung( n_e, n_HII, n_HeII, n_HeIII, temp )
# 
#   #Compton cooling off the CMB 
#   dQ_dt_CMB = Cooling_Rate_Compton_CMB( n_e, temp, z )
# 
#   dQ_dt = dQ_dt_phot - dQ_dt_cool_recomb - dQ_dt_cool_collisional - dQ_dt_brem - dQ_dt_CMB
#   dT_dt = -2 * H * temp  +  2./(3*K_b*n_tot)*dQ_dt
#   return dT_dt


 
def interpolate_rates( current_z, rates, log=True ):
  current_rates = {}
  z_vals = rates['z']
  types = [ 'ionization', 'heating' ]
  species = ['HI', 'HeI', 'HeII']
  for type in types:
    current_rates[type] = {}
    for chem in species:
      values = rates[type][chem]
      if log: values = np.log10(values)
      if (current_z > z_vals.max()) or (current_z < z_vals.min()) : current_val = rate_min
      else: 
        current_val = np.interp( current_z, z_vals, values )
        if log: current_val = 10**current_val
      current_rates[type][chem] = current_val
  return current_rates