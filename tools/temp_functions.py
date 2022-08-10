import numpy as np
import time as timer
from cosmo_constants import Myear, Mpc, K_b, M_p, M_p_cgs   
from cooling_rates_Katz95 import *
from cooling_rates import *
from rk4 import RK4_step
from tools import printProgress
from uvb_functions import interpolate_rates
import rates_grackle as gk





fields = [ 'temperature', 'n_H', 'n_HI', 'n_HII', 'n_He', 'n_HeI', 'n_HeII', 'n_HeIII', 'n_e' ]



def Integrate_Evolution( n_H_comov, n_He_comov, T_start, uvb_rates, cosmo, z_start, z_end, n_samples, output_to_file=None,
                         integrator='bdf', print_out=True ):
  
  
  
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
  
  solve_equilibrium = False
    
  current_state, solution = initialize( a_start, T_start, n_H_comov, n_He_comov )
  
  if output_to_file is not None:
    file_name = output_to_file['file_name']
    output_fields = output_to_file['fields']
    output_file = open( file_name, "w")
    print(f'Writing Fields: {output_fields}' )
    print(f'Output File: {file_name}' )
    output_to_file['file'] = output_file
    header = '#'
    for field in output_fields:
      header += f' {field}'
    output_file.write( header )

  n_iter = 100
  start = timer.time()
  if print_out:
    print( f'Integrator: {integrator}')
    print('Integrating Thermal Evolution...')
  for i in range(n):
    
    current_z = 1/current_a - 1
  
    append_state_to_solution( fields, current_state, solution )

    delta_a = da_vals[i]
    delta_z = dz_vals[i]
    dt = cosmo.get_dt( current_a, delta_a )
    
    print_str = 'da = {0:.3e}   dt = {1:.2e}  '.format(delta_a, dt) 
    # print( print_str )
    
    print_str = 'z = {0:.3f}   T = {1:.2f}   n_HI = {2:.5f} '.format(current_z, current_state['temperature'], current_state['n_HI']) 
    # print( print_str )

    state_array = get_state_array( fields, current_state )
    if integrator == 'rk4': state_array = RK4_step( all_deriv, time, state_array, dt, cosmo=cosmo, uvb_rates=uvb_rates, output_to_file=output_to_file  )
    if integrator == 'bdf': state_array = BDF_step( state_array, time, dt, cosmo=cosmo, uvb_rates=uvb_rates, output_to_file=output_to_file  )
    # break
    current_state = update_current_state( fields, state_array, current_state )
  
    print_str = 'z = {0:.3f}   T = {1:.2f}   n_HI = {2:.5f} '.format(current_z, current_state['temperature'], current_state['n_HI']) 
    # print( print_str )
  
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
  
    if i%n_iter == 0 and print_out:
      end = timer.time()
      delta = end - start
      n_H   = current_state['n_H']
      n_HII = current_state['n_HII']
      x_HII = n_HII / n_H
      print_str = 'z = {0:.3f}   T = {1:.2f}   x_HII = {2:.5f} '.format(current_z, current_state['temperature'], x_HII)  
      printProgress( i, n, delta, print_str=print_str )
      # print( print_str )
  
  append_state_to_solution( fields, current_state, solution )
  for field in fields:
    solution[field] = np.array(solution[field])
  solution['z'] = z_vals
  if print_out:
    printProgress( i, n, delta )
    print('\nEvolution Fisished')
  
  if output_to_file:
    output_file.close()
    if print_out:print( f'Saved File: {file_name}')
  
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
  
def Update_BDF( n, C, D, dt ):
  n_new = ( C * dt + n ) / ( 1 + D * dt )
  return n_new 
  
use_grackle_rates = True

def BDF_step( state_array, time, dt, **kargs ):
  # print('Starting BDF step')
  cosmo = kargs['cosmo']
  uvb_rates = kargs['uvb_rates']
  output_to_file = kargs['output_to_file']  
  init_temp, init_n_H, init_n_HI, init_n_HII, init_n_He, init_n_HeI, init_n_HeII, init_n_HeIII, init_n_e = state_array 
  temp, n_H, n_HI, n_HII, n_He, n_HeI, n_HeII, n_HeIII, n_e = state_array 
  n_min =  1e-60
  if n_HI < n_min: n_HI = n_min
  if n_HII < n_min : n_HII = n_min
  if n_HeI < n_min : n_HeI = n_min
  if n_HeII < n_min : n_HeII = n_min
  if n_HeIII < n_min : n_HeIII = n_min
  if n_e < n_min : n_e = n_min
  n_tot = n_HI + n_HII + n_HeI + n_HeII + n_HeIII + n_e
  
  # Get Hubble 
  current_a = cosmo.get_current_a( time )
  current_z = 1./current_a - 1
  H = cosmo.get_Hubble( current_a ) * 1000 / Mpc  # 1/sec
  
  # Get Current photoheating and photoionization rates
  current_rates = interpolate_rates( current_z, uvb_rates, log=False )
  photoheating_rates = current_rates['heating']
  photoionization_rates = current_rates['ionization']

  # Photoheating
  Q_phot_HI = n_HI * photoheating_rates['HI'] 
  Q_phot_HeI = n_HeI * photoheating_rates['HeI'] 
  Q_phot_HeII = n_HeII * photoheating_rates['HeII'] 
  dQ_dt_phot = Q_phot_HI + Q_phot_HeI + Q_phot_HeII 
  
  if not use_grackle_rates:
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
  
  if use_grackle_rates:
    # Recopmbination Cooling
    Q_cool_rec_HII    = gk.cool_reHII_rate(temp)*n_HII*n_e 
    Q_cool_rec_HeII   = gk.cool_reHeII1_rate(temp)*n_HeII*n_e 
    Q_cool_rec_HeII_d = gk.cool_reHeII2_rate(temp)*n_HeII*n_e  
    Q_cool_rec_HeIII  = gk.cool_reHeIII_rate(temp)*n_HeIII*n_e  
    dQ_dt_cool_recomb = Q_cool_rec_HII + Q_cool_rec_HeII + Q_cool_rec_HeII_d + Q_cool_rec_HeIII 
    
    # Collisional Cooling
    Q_cool_collis_ext_HI   = gk.cool_ceHI_rate(temp)*n_HI*n_e
    Q_cool_collis_ext_HeI  = gk.cool_ceHeI_rate(temp)*n_HeI*n_e 
    Q_cool_collis_ext_HeII = gk.cool_ceHeII_rate(temp)*n_HeII*n_e
    Q_cool_collis_ion_HI   = gk.cool_ciHI_rate(temp)*n_HI*n_e 
    Q_cool_collis_ion_HeI  = gk.cool_ciHeI_rate(temp)*n_HeI*n_e 
    Q_cool_collis_ion_HeIS = gk.cool_ciHeIS_rate(temp)*n_HeI*n_e
    Q_cool_collis_ion_HeII = gk.cool_ciHeII_rate(temp)*n_HeII*n_e
    dQ_dt_cool_collisional = Q_cool_collis_ext_HI + Q_cool_collis_ext_HeI + Q_cool_collis_ext_HeII + Q_cool_collis_ion_HI + Q_cool_collis_ion_HeI + Q_cool_collis_ion_HeIS + Q_cool_collis_ion_HeII
      
    dQ_dt_brem = gk.cool_brem_rate(temp)*n_HI*n_HeII*n_HeIII
    
    dQ_dt_CMB = Cooling_Rate_Compton_CMB_MillesOstriker01( n_e, temp, current_z ) 
  
  # Heating and Cooling Rates 
  dQ_dt  = dQ_dt_phot - dQ_dt_cool_recomb - dQ_dt_cool_collisional - dQ_dt_brem - dQ_dt_CMB

  # Photoionization Rates
  photo_HI   = photoionization_rates['HI']
  photo_HeI  = photoionization_rates['HeI'] 
  photo_HeII = photoionization_rates['HeII'] 
  
  if not use_grackle_rates:
    # Get Recombination Rates
    recomb_HII    = Recombination_Rate_HII_Hui97( temp )   
    recomb_HeII   = Recombination_Rate_HeII_Hui97( temp )  
    recomb_HeIII  = Recombination_Rate_HeIII_Katz95( temp ) 
    recomb_HeII_d = Recombination_Rate_dielectronic_HeII_Hui97( temp ) 

    # Get Collisional Ionization  Rates
    coll_HI     = Collisional_Ionization_Rate_e_HI_Hui97( temp )          
    coll_HeI    = Collisional_Ionization_Rate_e_HeI_Abel97( temp )  
    coll_HeII   = Collisional_Ionization_Rate_e_HeII_Abel97( temp ) 
    coll_HI_HI  = Collisional_Ionization_Rate_HI_HI_Lenzuni91( temp ) 
    coll_HII_HI = Collisional_Ionization_Rate_HII_HI_Lenzuni91( temp ) 
    coll_HeI_HI = Collisional_Ionization_Rate_HeI_HI_Lenzuni91( temp )  

  if use_grackle_rates:
    # Get Recombination Rates
    recomb_HII    = gk.recomb_HII(temp)   
    recomb_HeII   = gk.recomb_HeII(temp)  
    recomb_HeIII  = gk.recomb_HeIII(temp) 
    recomb_HeII_d = 0 
    
    # Get Collisional Ionization  Rates
    coll_HI     = gk.coll_i_HI(temp)      
    coll_HeI    = gk.coll_i_HeI(temp)  
    coll_HeII   = gk.coll_i_HeII(temp) 
    coll_HI_HI  = gk.coll_i_HI_HI(temp) 
    coll_HII_HI = 0 
    coll_HeI_HI = gk.coll_i_HI_HeI(temp)  
    

  dens = ( n_HI + n_HII + 4*( n_HeI + n_HeII + n_HeIII ) ) * M_p_cgs
  x_HI = n_HI / dens * M_p_cgs
  x_HII = n_HII / dens * M_p_cgs
  x_HeI = n_HeI / dens * M_p_cgs
  x_HeII = n_HeII / dens * M_p_cgs
  x_HeIII = n_HeIII / dens * M_p_cgs
  x_e = n_e / dens * M_p_cgs
  x_sum = x_HI + x_HII + x_HeI + x_HeII + x_HeIII + x_e
  
  # 1. Update HI
  C_HI = recomb_HII*n_HII*n_e
  D_HI = photo_HI + coll_HI*n_e + coll_HI_HI*n_HI + coll_HII_HI*n_HII + coll_HeI_HI*n_HeI
  n_HI = Update_BDF( n_HI, C_HI, D_HI, dt )
  
  # 2. Update HII
  C_HII = photo_HI*n_HI + coll_HI*n_HI*n_e + coll_HI_HI*n_HI*n_HI + coll_HII_HI*n_HII*n_HI + coll_HeI_HI*n_HeI*n_HI
  D_HII = recomb_HII*n_e
  n_HII = Update_BDF( n_HII, C_HII, D_HII, dt )
  
  # 3. Update electron
  C_e_phot = photo_HI*n_HI + photo_HeI*n_HeI + photo_HeII*n_HeII
  C_e_coll = coll_HI*n_HI*n_e + coll_HeI*n_HeI*n_e + coll_HeII*n_HeII*n_e + coll_HI_HI*n_HI*n_HI * coll_HII_HI*n_HII*n_HI * coll_HeI_HI*n_HeI*n_HI
  C_e = C_e_phot + C_e_coll
  D_e = recomb_HII*n_HII + recomb_HeII*n_HeII + recomb_HeII_d*n_HeII + recomb_HeIII*n_HeIII
  n_e = Update_BDF( n_e, C_e, D_e, dt )
  
  # 4. Update HeI
  C_HeI = recomb_HeII*n_HeII*n_e + recomb_HeII_d*n_HeII*n_e
  D_HeI = photo_HeI + coll_HeI*n_e
  n_HeI = Update_BDF( n_HeI, C_HeI, D_HeI, dt )
  
  # 5. Update HeII
  C_HeII = photo_HeI*n_HeI + coll_HeI*n_HeI*n_e + recomb_HeIII*n_HeIII*n_e
  D_HeII = photo_HeII + coll_HeII*n_e + recomb_HeII*n_e + recomb_HeII_d*n_e
  n_HeII = Update_BDF( n_HeII, C_HeII, D_HeII, dt )
  
  # 6. Update HeIII
  C_HeIII = photo_HeII*n_HeII + coll_HeII*n_HeII*n_e
  D_HeIII = recomb_HeIII*n_e
  n_HeIII = Update_BDF( n_HeIII, C_HeIII, D_HeIII, dt )
  
  dx_dt = ( C_e - D_e*n_e ) / dens * M_p_cgs
  
  # 7. Update temperature
  # d_T_dt = dT_dt_hubble +  2./(3*K_b*n_tot)*dQ_dt - T/x_sum*dx_dt
  dT_dt =  2./(3*K_b*n_tot)*dQ_dt - temp/x_sum*dx_dt
  temp += dT_dt*dt
  
  # 8. Add expansion
  temp    -= 2 * H * init_temp * dt
  n_H     -= 3 * H * init_n_H * dt
  n_He    -= 3 * H * init_n_He * dt
  n_HI    -= 3 * H * init_n_HI * dt
  n_HII   -= 3 * H * init_n_HII * dt
  n_HeI   -= 3 * H * init_n_HeI * dt
  n_HeII  -= 3 * H * init_n_HeII * dt
  n_HeIII -= 3 * H * init_n_HeIII * dt
  n_e     -= 3 * H * init_n_e * dt
  
  # Return the updated state
  state_array = np.array([ temp, n_H, n_HI, n_HII, n_He, n_HeI, n_HeII, n_HeIII, n_e ])
  return state_array
  
  
def all_deriv( time, state_array, kargs=None ):
  cosmo = kargs['cosmo']
  uvb_rates = kargs['uvb_rates']
  n_step = kargs['n_step'] 
  output_to_file = kargs['output_to_file']
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
    
  # Fraction of Recombination to photoionization rates
  # rates_fraction = dn_dt_recomb_HII / dn_dt_photo_HI
  # print( f'z: {current_z}  Recombination / Photoionization HI: {rates_fraction}'  )
  if n_step == 0 and output_to_file:
    outfile = output_to_file['file']
    fields = output_to_file['fields']
    for field in fields:
      if field == 'z': outfile.write( f'{current_z:.4f} ' )
      elif field == 'photoionization_HI': outfile.write( f'{dn_dt_photo_HI:.4e} ' )
      elif field == 'recombination_HI': outfile.write( f'{dn_dt_recomb_HII:.4e} ' )
      else: print( f'ERROR: Invalid output filed: {field}' )
    outfile.write( '\n')
  
  
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
  
  # print( f"Phot Heat   {dQ_dt_phot:.4e}  ")    
  # print( f"Net Heat:   {dQ_dt:.4e}  ")  
  # print( f"N_tot:   {n_tot:.4e}  ")    
  # print( f"dT_dt:   {2./(3*K_b*n_tot)*dQ_dt:.4e}  ")  
    
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

