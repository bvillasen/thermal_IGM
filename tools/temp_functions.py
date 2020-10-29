import numpy as np
import time as timer
from cosmo_constants import Myear, Mpc, K_b, M_p, M_p_cgs   
from cooling_rates_Katz95 import *
from cooling_rates import *
from rk4 import RK4_step
from tools import printProgress


rate_min = 1e-30


alpha_min = rate_min
Gamma_min = rate_min

fields = [ 'temperature', 'n_H', 'n_HI', 'n_HII', 'n_He', 'n_HeI', 'n_HeII', 'n_HeIII', 'n_e' ]

def Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates_in, cosmo, z_start, z_end, n_samples, HI_rates_factor=1.0, HeI_rates_factor=1.0, HeII_rates_factor=1.0, delta_z_HI=0.0, delta_z_HeII=0.0 ):

  # Create scale factor array
  a_start = 1. / ( z_start + 1 )
  a_end = 1. / ( z_end + 1 )
  z_vals = np.linspace( z_end, z_start, n_samples)[::-1]
  a_vals = 1/( z_vals + 1 )
  da_vals = a_vals[1:] - a_vals[:-1]
  dz_vals = z_vals[1:] - z_vals[:-1]

  # Integrate Temperature evolution
  n = len( da_vals )
  time = 0 
  current_a = a_start
  
  
  
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
  
  # Rescalke the Photoionization and Photoheating rates
  uvb_rates['ionization']['HI']   *= HI_rates_factor
  uvb_rates['heating']['HI']      *= HI_rates_factor 
  uvb_rates['ionization']['HeI']  *= HeI_rates_factor
  uvb_rates['heating']['HeI']     *= HeI_rates_factor 
  uvb_rates['ionization']['HeII'] *= HeII_rates_factor
  # uvb_rates['heating']['HeII']    *= HeII_rates_factor 
  uvb_rates['z']['HI'] += delta_z_HI
  uvb_rates['z']['HeI'] += delta_z_HI
  uvb_rates['z']['HeII'] += delta_z_HeII
  



    
  current_state, solution = initialize( a_start, T_start, n_H_comov, n_He_comov )

  n_iter = 100
  start = timer.time()
  print('Integrating Thermal Evolution...')
  for i in range(n):
    
    solve_equilibrium = False
    current_z = 1/current_a - 1
    


    append_state_to_solution( fields, current_state, solution )

    delta_a = da_vals[i]
    delta_z = dz_vals[i]
    dt = cosmo.get_dt( current_a, delta_a )

    state_array = get_state_array( fields, current_state )
    state_array = RK4_step( all_deriv, time, state_array, dt, cosmo=cosmo, uvb_rates=uvb_rates  )
    current_state = update_current_state( fields, state_array, current_state )
    
    if solve_equilibrium:
      H = cosmo.get_Hubble( current_a )
      current_rates_eq = interpolate_rates( current_z, uvb_rates_eq )
      n_H = n_H_comov / current_a**3
      n_He = n_He_comov / current_a**3 
      # #Get Ioniozation Fractions
      chemistry_eq = Get_Ionization_Fractions_Iterative( n_H, n_He, temp_eq, current_rates_eq['ionization']) 
      # temp_eq = RK4_step( temp_deriv, time, temp_eq, dt, H=H, rates=current_rates_eq, chemistry=chemistry_eq, z=current_z) 
      temp_eq_vals.append(temp_eq)

    time += dt
    current_a += delta_a
    
    if i%n_iter == 0:
      end = timer.time()
      delta = end - start
      print_str = 'z = {0:.3f}   T = {1:.2f}   n_HI = {2:.5f} '.format(current_z, current_state['temperature'], current_state['n_HI'])  
      printProgress( i, n, delta, print_str='' )
      # print( print_str )
      


  append_state_to_solution( fields, current_state, solution )
  for field in fields:
    solution[field] = np.array(solution[field])
  solution['z'] = z_vals
  printProgress( i, n, delta )
  print('\nEvolution Fisished')
  return solution

def initialize( a_start, T_start, n_H_comov, n_He_comov ):
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
  
  solution = {}
  for field in fields:
    solution[field] = []

  return current_state, solution

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
  
  dens = ( n_HI + n_HII + 4*( n_HeI + n_HeII + n_HeIII ) ) * M_p_cgs
  x_HI = n_HI / dens * M_p_cgs
  x_HII = n_HII / dens * M_p_cgs
  x_HeI = n_HeI / dens * M_p_cgs
  x_HeII = n_HeII / dens * M_p_cgs
  x_HeIII = n_HeIII / dens * M_p_cgs
  x_e = n_e / dens * M_p_cgs
  
  x_sum = x_HI + x_HII + x_HeI + x_HeII + x_HeIII + x_e
  
  dn_dt_HI    = dn_dt_HI_p    - dn_dt_HI_m
  dn_dt_HII   = dn_dt_HII_p   - dn_dt_HII_m
  dn_dt_HeI   = dn_dt_HeI_p   - dn_dt_HeI_m
  dn_dt_HeII  = dn_dt_HeII_p  - dn_dt_HeII_m
  dn_dt_HeIII = dn_dt_HeIII_p - dn_dt_HeIII_m
  dn_dt_e     = dn_dt_e_p     - dn_dt_e_m
  
  dx_dt = ( dn_dt_HI + dn_dt_HII + dn_dt_HeI + dn_dt_HeII + dn_dt_HeIII + dn_dt_e  ) / dens * M_p_cgs 
    
  d_T_dt = dT_dt_hubble +  2./(3*K_b*n_tot)*dQ_dt - T/x_sum*dx_dt
  # d_T_dt = dT_dt_hubble
  n_hubble_factor = -3 * H   
  d_nH_dt     = n_hubble_factor * n_H 
  d_nHe_dt    = n_hubble_factor * n_He  
  d_nHI_dt    = n_hubble_factor * n_HI    + dn_dt_HI 
  d_nHII_dt   = n_hubble_factor * n_HII   + dn_dt_HII
  d_nHeI_dt   = n_hubble_factor * n_HeI   + dn_dt_HeI
  d_nHeII_dt  = n_hubble_factor * n_HeII  + dn_dt_HeII
  d_nHeIII_dt = n_hubble_factor * n_HeIII + dn_dt_HeIII
  d_ne_dt     = n_hubble_factor * n_e     + dn_dt_e
  
  return np.array([ d_T_dt, d_nH_dt,  d_nHI_dt, d_nHII_dt, d_nHe_dt,  d_nHeI_dt, d_nHeII_dt, d_nHeIII_dt,  d_ne_dt  ])


 
def interpolate_rates( current_z, rates, log=True ):
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
  return current_rates
  


def Get_Ionization_Fractions_Iterative( n_H, n_He, T, Gamma_phot  ):


  #Recombination Rates
  alpha_HII = Recombination_Rate_HII_Katz95( T )
  alpha_HeII = Recombination_Rate_HeII_Katz95( T )
  alpha_HeIII = Recombination_Rate_HeIII_Katz95( T )
  alpha_d = Recombination_Rate_dielectronic_HeII_Katz95( T )

  print( alpha_HII )
  print( alpha_HeII )
  print( alpha_HeIII )
  print( alpha_d )

  #Ionization Rates
  #Collisional
  Gamma_e_HI = Collisional_Ionization_Rate_e_HI_Katz95( T )
  Gamma_e_HeI = Collisional_Ionization_Rate_e_HeI_Katz95( T )
  Gamma_e_HeII = Collisional_Ionization_Rate_e_HeII_Katz95( T )

  print( Gamma_e_HI )
  print( Gamma_e_HeI )
  print( Gamma_e_HeII )

  #Photoionization 
  Gamma_phot_HI   = Gamma_phot['HI']   #Photoionization of neutral hydrogen
  Gamma_phot_HeI  = Gamma_phot['HeI']  #Photoionization of neutral helium
  Gamma_phot_HeII = Gamma_phot['HeII'] #Photoionization of singly ionized helium

  print( Gamma_phot_HI )
  print( Gamma_phot_HeI )
  print( Gamma_phot_HeII )

  epsilon = 1e-2
  # initialize eletron fraction
  n_e = n_H # When no Radiation the electron number is not needed for computin ionization fractions
  iterate = True
  n_iter = 0
  #Compute Ionization Fractions
  while iterate:
    if n_iter > 0:
      vals_old = {}
      vals_old['HI']    = n_HI
      vals_old['HII']   = n_HII
      vals_old['HeI']   = n_HeI
      vals_old['HeII']  = n_HeII
      vals_old['HeIII'] = n_HeIII
      vals_old['e'] = n_e

    #Eq 33:
    n_HI  = alpha_HII * n_H / (  alpha_HII + Gamma_e_HI + Gamma_phot_HI/n_e   )
    n_HII = n_H - n_HI
    #Eq 35:
    n_HeII = n_He / ( 1 + (alpha_HeII + alpha_d)/(Gamma_e_HeI + Gamma_phot_HeI/n_e) + (Gamma_e_HeII + Gamma_phot_HeII/n_e)/alpha_HeIII )
    #Eq 36:
    n_HeI = n_HeII * ( alpha_HeII + alpha_d ) / (Gamma_e_HeI + Gamma_phot_HeI/n_e) 
    #Eq 37:
    n_HeIII = n_HeII * ( Gamma_e_HeII + Gamma_phot_HeII/n_e ) / alpha_HeIII
    #Eq 34:
    n_e  = n_HII + n_HeII + 2*n_HeIII

    if n_iter > 0:
      vals_new = {}
      vals_new['HI']    = n_HI
      vals_new['HII']   = n_HII
      vals_new['HeI']   = n_HeI
      vals_new['HeII']  = n_HeII
      vals_new['HeIII'] = n_HeIII
      vals_new['e'] = n_e 

      iterate = False
      for name in ['e' ]:
        v_old = vals_old[name]
        v_new = vals_new[name]
        diff = ( v_new - v_old ) / v_old
        # print( diff )
        if np.abs(diff) > epsilon:   iterate = True

    n_iter += 1

  print(' Chemistry Converged in {0} iterationms'.format(n_iter))

  ionization_frac = {}
  ionization_frac['n_HI'] = n_HI
  ionization_frac['n_HII'] = n_HII
  ionization_frac['n_HeI'] = n_HeI
  ionization_frac['n_HeII'] = n_HeII
  ionization_frac['n_HeIII'] = n_HeIII
  ionization_frac['n_e'] = n_e
  return ionization_frac

