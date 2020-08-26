import sys
import numpy as np


alpha_min = 1e-60
Gamma_min = 1e-60


#Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)
def Cooling_Rate_Collisional_Excitation_e_HI_Katz95( n_e, n_HI, temp ):
  temp_5 = temp / 1e5
  Lambda_e_HI = 7.50e-19 * np.exp(-118348.0 / temp) * n_e*n_HI/( 1 + temp_5**(0.5) )
  return Lambda_e_HI

def Cooling_Rate_Collisional_Excitation_e_HeII_Katz95( n_e, n_HeII, temp ):
  temp_5 = temp / 1e5
  Lambda_e_HeII = 5.54e-17 * temp**(-0.397) * np.exp(-473638.0 / temp) * n_e*n_HeII/( 1 + temp_5**(0.5) )
  return Lambda_e_HeII


#Colisional ionization  of HI, HeI and HeII
def Cooling_Rate_Collisional_Ionization_e_HI_Katz95( n_e, n_HI, temp ):
  temp_5 = temp / 1e5
  Lambda_e_HI = 1.27e-21 * temp**(0.5) * np.exp(-157809.1 / temp) * n_e*n_HI/( 1 + temp_5**(0.5) )
  return Lambda_e_HI


def Cooling_Rate_Collisional_Ionization_e_HeI_Katz95( n_e, n_HeI, temp ):
  temp_5 = temp / 1e5
  Lambda_e_HeI = 9.38e-22 * temp**(0.5) * np.exp(-285335.4 / temp) * n_e*n_HeI/( 1 + temp_5**(0.5) )
  return Lambda_e_HeI

def Cooling_Rate_Collisional_Ionization_e_HeII_Katz95( n_e, n_HeII, temp ):
  temp_5 = temp / 1e5
  Lambda_e_HeII = 4.95e-22 * temp**(0.5) * np.exp(-631515.0/ temp) * n_e*n_HeII/( 1 + temp_5**(0.5) )
  return Lambda_e_HeII


def Collisional_Ionization_Rate_e_HI_Katz95( temp ):
  temp_5 = temp / 1e5
  Gamma_e_HI = 5.85e-11 * temp**(0.5) *  np.exp(-157809.1 / temp) / ( 1 + temp_5**(0.5) )
  if Gamma_e_HI < Gamma_min: Gamma_e_HI = Gamma_min
  return Gamma_e_HI

def Collisional_Ionization_Rate_e_HeI_Katz95( temp ):
  temp_5 = temp / 1e5
  Gamma_e_HeI = 2.38e-11 * temp**(0.5) *  np.exp(-285335.4 / temp) / ( 1 + temp_5**(0.5) )
  if Gamma_e_HeI < Gamma_min: Gamma_e_HeI = Gamma_min
  return Gamma_e_HeI

def Collisional_Ionization_Rate_e_HeII_Katz95( temp ):
  temp_5 = temp / 1e5
  Gamma_e_HeII = 5.688e-12 * temp**(0.5) *  np.exp(-631515.0 / temp) / ( 1 + temp_5**(0.5) )
  if Gamma_e_HeII < Gamma_min: Gamma_e_HeII = Gamma_min
  return Gamma_e_HeII


#Standard Recombination of HII, HeII and HeIII

def Cooling_Rate_Recombination_HII_Katz95( n_e, n_HII, temp ):
  temp_3 = temp / 1e3
  temp_6 = temp / 1e6
  Lambda_HII = 8.70e-27 * temp**(0.5) * temp_3**(-0.2) * ( 1 + temp_6**0.7 )**(-1) * n_e * n_HII 
  return Lambda_HII
  
def Cooling_Rate_Recombination_HeII_Katz95( n_e, n_HeII, temp ):
  Lambda_HeII = 1.55e-26 * temp**(0.3647) * n_e * n_HeII 
  return Lambda_HeII


def Cooling_Rate_Recombination_HeIII_Katz95( n_e, n_HeIII, temp ):
  temp_3 = temp / 1e3
  temp_6 = temp / 1e6
  Lambda_HeIII = 3.48e-26 * temp**(0.5) * temp_3**(-0.2) * ( 1 + temp_6**0.7 )**(-1) * n_e * n_HeIII 
  return Lambda_HeIII
    


def Recombination_Rate_HII_Katz95( temp ):
  temp_3 = temp / 1e3
  temp_6 = temp / 1e6
  alpha_HII = 8.4e-11 * temp**(-0.5) * temp_3**(-0.2) * ( 1 + temp_6**0.7 )**(-1)
  if alpha_HII < alpha_min: alpha_HII = alpha_min
  return alpha_HII
 
def Recombination_Rate_HeII_Katz95( temp ):
 alpha_HeII = 1.5e-10 * temp**(-0.6353)
 if alpha_HeII < alpha_min: alpha_HeII = alpha_min
 return alpha_HeII

def Recombination_Rate_HeIII_Katz95( temp ):
  temp_3 = temp / 1e3
  temp_6 = temp / 1e6
  alpha_HeIII = 3.36e-10 * temp**(-0.5) * temp_3**(-0.2) * ( 1 + temp_6**0.7 )**(-1)
  if alpha_HeIII < alpha_min: alpha_HeIII = alpha_min
  return alpha_HeIII
  
 
 
#Dielectronic recombination of HeII
def Cooling_Rate_Recombination_dielectronic_HeII_Katz95( n_e, n_HeII, temp ):
 Lambda_d = 1.24e-13 * temp**(-1.5) * np.exp( -470000.0/temp ) * ( 1 + 0.3 * np.exp( -94000.0/temp ) ) * n_e * n_HeII
 return Lambda_d


def Recombination_Rate_dielectronic_HeII_Katz95( temp ):
 alpha_d = 1.9e-3 * temp**(-1.5) * np.exp( -470000.0/temp ) * ( 1 + 0.3 * np.exp( -94000.0/temp ) )
 if alpha_d < alpha_min: alpha_d = alpha_min
 return alpha_d


#Free-Free emission (Bremsstrahlung) 
def gaunt_factor( log10_T ):
  gff = 1.1 + 0.34 * np.exp( -(5.5 - log10_T)**2 / 3.0   )
  return gff 

def Cooling_Rate_Bremsstrahlung_Katz95( n_e, n_HII, n_HeII, n_HeIII, temp ):
  gff = gaunt_factor( np.log10(temp) )
  Lambda_bmst = 1.42e-27 * gff * temp**(0.5) * ( n_HII + n_HeII + 4*n_HeIII ) * n_e
  return Lambda_bmst



#Compton cooling off the CMB 
def Cooling_Rate_Compton_CMB_Katz95( n_e, temp, z ):
  # Ikeuchi & Ostriker 1986
  return 5.41e-36 * n_e * temp * ( 1 + z )**4


# def Get_Ionization_Fractions_Iterative( n_H, n_He, T, Gamma_phot  ):
# 
# 
#   #Recombination Rates
#   alpha_HII = Recombination_Rate_HII( T )
#   alpha_HeII = Recombination_Rate_HeII( T )
#   alpha_HeIII = Recombination_Rate_HeIII( T )
#   alpha_d = Recombination_Rate_dielectronic_HeII( T )
# 
#   # print( alpha_HII )
#   # print( alpha_HeII )
#   # print( alpha_HeIII )
#   # print( alpha_d )
# 
#   #Ionization Rates
#   #Collisional
#   Gamma_e_HI = Collisional_Ionization_Rate_e_HI( T )
#   Gamma_e_HeI = Collisional_Ionization_Rate_e_HeI( T )
#   Gamma_e_HeII = Collisional_Ionization_Rate_e_HeII( T )
# 
#   # print( Gamma_e_HI )
#   # print( Gamma_e_HeI )
#   # print( Gamma_e_HeII )
# 
#   #Photoionization 
#   Gamma_phot_HI   = Gamma_phot['HI']   #Photoionization of neutral hydrogen
#   Gamma_phot_HeI  = Gamma_phot['HeI']  #Photoionization of neutral helium
#   Gamma_phot_HeII = Gamma_phot['HeII'] #Photoionization of singly ionized helium
# 
#   epsilon = 1e-1
#   # initialize eletron fraction
#   n_e = n_H # When no Radiation the electron number is not needed for computin ionization fractions
#   iterate = True
#   n_iter = 0
#   #Compute Ionization Fractions
#   while iterate:
#     if n_iter > 0:
#       vals_old = {}
#       vals_old['HI']    = n_HI
#       vals_old['HII']   = n_HII
#       vals_old['HeI']   = n_HeI
#       vals_old['HeII']  = n_HeII
#       vals_old['HeIII'] = n_HeIII
#       vals_old['e'] = n_e
# 
#     #Eq 33:
#     n_HI  = alpha_HII * n_H / (  alpha_HII + Gamma_e_HI + Gamma_phot_HI/n_e   )
#     n_HII = n_H - n_HI
#     #Eq 35:
#     n_HeII = n_He / ( 1 + (alpha_HeII + alpha_d)/(Gamma_e_HeI + Gamma_phot_HeI/n_e) + (Gamma_e_HeII + Gamma_phot_HeII/n_e)/alpha_HeIII )
#     #Eq 36:
#     n_HeI = n_HeII * ( alpha_HeII + alpha_d ) / (Gamma_e_HeI + Gamma_phot_HeI/n_e) 
#     #Eq 37:
#     n_HeIII = n_HeII * ( Gamma_e_HeII + Gamma_phot_HeII/n_e ) / alpha_HeIII
#     #Eq 34:
#     n_e  = n_HII + n_HeII + 2*n_HeIII
# 
#     if n_iter > 0:
#       vals_new = {}
#       vals_new['HI']    = n_HI
#       vals_new['HII']   = n_HII
#       vals_new['HeI']   = n_HeI
#       vals_new['HeII']  = n_HeII
#       vals_new['HeIII'] = n_HeIII
#       vals_new['e'] = n_e 
# 
#       iterate = False
#       for name in ['HI',  'HeI', 'HeII',  'e' ]:
#         v_old = vals_old[name]
#         v_new = vals_new[name]
#         diff = ( v_new - v_old ) / v_old
#         if np.abs(diff) > epsilon: 
#           # print( diff )
#           iterate = True
# 
#     n_iter += 1
# 
#   print(' Chemistry Converged in {0} iterationms'.format(n_iter))
# 
#   ionization_frac = {}
#   ionization_frac['n_HI'] = n_HI
#   ionization_frac['n_HII'] = n_HII
#   ionization_frac['n_HeI'] = n_HeI
#   ionization_frac['n_HeII'] = n_HeII
#   ionization_frac['n_HeIII'] = n_HeIII
#   ionization_frac['n_e'] = n_e
#   return ionization_frac
# 
# 
# def Get_Ionization_Fractions( n_H, n_He, T  ):
#   n_e = n_H # When no Radiation the electron number is not needed for computin ionization fractions
#   y = n_He / n_H
# 
#   #Recombination Rates
#   alpha_HII = Recombination_Rate_HII( T )
#   alpha_HeII = Recombination_Rate_HeII( T )
#   alpha_HeIII = Recombination_Rate_HeIII( T )
#   alpha_d = Recombination_Rate_dielectronic_HeII( T )
# 
#   #Ionization Rates
#   #Collisional
#   Gamma_e_HI = Collisional_Ionization_Rate_e_HI( T )
#   Gamma_e_HeI = Collisional_Ionization_Rate_e_HeI( T )
#   Gamma_e_HeII = Collisional_Ionization_Rate_e_HeII( T )
#   #Photoionization 
#   Gamma_phot_HI = 0 #Photoionization of neutral hydrogen
#   Gamma_phot_HeI = 0 #Photoionization of neutral helium
#   Gamma_phot_HeII = 0 #Photoionization of singly ionized helium
# 
#   #Compute Ionization Fractions
#   #Eq 33:
#   n_HI  = alpha_HII * n_H / (  alpha_HII + Gamma_e_HI + Gamma_phot_HI/n_e   )
#   n_HII = n_H - n_HI
#   #Eq 35:
#   n_HeII = y * n_H / ( 1 + (alpha_HeII + alpha_d)/(Gamma_e_HeI + Gamma_phot_HeI/n_e) + (Gamma_e_HeII + Gamma_phot_HeII/n_e)/alpha_HeIII )
#   #Eq 36:
#   n_HeI = n_HeII * ( alpha_HeII + alpha_d ) / (Gamma_e_HeI + Gamma_phot_HeI/n_e) 
#   #Eq 37:
#   n_HeIII = n_HeII * ( Gamma_e_HeII + Gamma_phot_HeII/n_e ) / alpha_HeIII
#   #Eq 34:
#   n_e = n_HII + n_HeII + 2*n_HeIII 
# 
#   ionization_frac = {}
#   ionization_frac['HI'] = n_HI
#   ionization_frac['HII'] = n_HII
#   ionization_frac['HeI'] = n_HeI
#   ionization_frac['HeII'] = n_HeII
#   ionization_frac['HeIII'] = n_HeIII
#   ionization_frac['e'] = n_e
#   return ionization_frac
# 
# def Get_Cooling_Rates( n_H, n_He, T  ):
#   #Compute ioniozation fractions
#   ionization_frac = Get_Ionization_Fractions( n_H, n_He, T )
# 
#   #Compute Cooling Rates
#   n_HI    = ionization_frac['HI']
#   n_HII   = ionization_frac['HII']
#   n_HeI   = ionization_frac['HeI']
#   n_HeII  = ionization_frac['HeII']
#   n_HeIII = ionization_frac['HeIII']
#   n_e     = ionization_frac['e']
# 
#   #Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)
#   Lambda_exitation_HI = Cooling_Rate_Collisional_Excitation_e_HI( n_e, n_HI, T )  
#   Lambda_exitation_HeII = Cooling_Rate_Collisional_Excitation_e_HeII( n_e, n_HeII, T )
# 
#   #Colisional ionization  of HI, HeI and HeII
#   Lambda_ionization_HI = Cooling_Rate_Collisional_Ionization_e_HI( n_e, n_HI, T )
#   Lambda_ionization_HeI = Cooling_Rate_Collisional_Ionization_e_HeI( n_e, n_HeI, T )
#   Lambda_ionization_HeII = Cooling_Rate_Collisional_Ionization_e_HeII( n_e, n_HeII, T )
# 
#   #Standard Recombination of HII, HeII and HeIII
#   Lambda_recombination_HII = Cooling_Rate_Recombination_HII(  n_e, n_HII, T )
#   Lambda_recombination_HeII = Cooling_Rate_Recombination_HeII(  n_e, n_HeII, T )
#   Lambda_recombination_HeIII = Cooling_Rate_Recombination_HeIII(  n_e, n_HeIII, T )
#   Lambda_recombination_dielectronic_HeII = Cooling_Rate_Recombination_dielectronic_HeII(  n_e, n_HeII, T )
# 
#   #Free-Free (Bremsstrahlung)
#   Lambda_bremst = Cooling_Rate_Bremsstrahlung( n_e, n_HII, n_HeII, n_HeIII, T )
# 
#   cooling_rates = { 'Excitation':{}, 'Ionization':{}, 'Recombination':{}, 'Bremsstrahlung':{} }
#   cooling_rates['Excitation']['HI'] = Lambda_exitation_HI
#   cooling_rates['Excitation']['HeII'] = Lambda_exitation_HeII
#   cooling_rates['Ionization']['HI'] = Lambda_ionization_HI
#   cooling_rates['Ionization']['HeI'] = Lambda_ionization_HeI
#   cooling_rates['Ionization']['HeII'] = Lambda_ionization_HeII
#   cooling_rates['Recombination']['HII'] = Lambda_recombination_HII
#   cooling_rates['Recombination']['HeII'] = Lambda_recombination_HeII
#   cooling_rates['Recombination']['HeIII'] = Lambda_recombination_HeIII
#   cooling_rates['Recombination']['HeII_dielectronic'] = Lambda_recombination_dielectronic_HeII
#   cooling_rates['Bremsstrahlung'] = Lambda_bremst
#   return cooling_rates
  
  