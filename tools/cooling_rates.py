import sys
import numpy as np
from cosmo_constants import K_b

alpha_min = 1e-30
Gamma_min = 1e-30

ev_to_K = 1.160451812e4
K_to_ev = 1./ev_to_K

#Colisional excitation of neutral hydrogen (HI) and singly ionized helium (HeII)
def Collisional_Ionization_Rate_e_HI_Abel97( temp ):
  temp = temp * K_to_ev
  ln_T = np.log(temp)
  k1 = np.exp( -32.71396786 + 13.536556 * ln_T - 5.73932875 * ln_T**2 + 1.56315498 * ln_T**3 -
       0.2877056 * ln_T**4 + 3.48255977e-2 * ln_T**5 - 2.63197617e-3 * ln_T**6 +
       1.11954395e-4 * ln_T**7 - 2.03914985e-6 * ln_T**8 ) # cm^3 s^-1
  return k1

def Recombination_Rate_HII_Abel97( temp ):
  temp = temp * K_to_ev
  ln_T = np.log(temp)
  k2 = np.exp( -28.6130338 - 0.72411256 * ln_T - 2.02604473e-2 * ln_T**2 -
      2.38086188e-3 * ln_T**3 - 3.21260521e-4 * ln_T**4 - 1.42150291e-5 * ln_T**5 +
      4.98910892e-6 * ln_T**6 + 5.75561414e-7 * ln_T**7 - 1.85676704e-8 * ln_T**8 -
      3.07113524e-9 * ln_T**9 ) # cm 3 s -1 .
  return k2 
  
def Collisional_Ionization_Rate_e_HeI_Abel97( temp ):
  temp = temp * K_to_ev
  ln_T = np.log(temp)
  k3 = np.exp( -44.09864886 + 23.91596563 * ln_T - 10.7532302 * ln_T**2 + 3.05803875 * ln_T**3 -
      0.56851189 * ln_T**4 + 6.79539123e-2 * ln_T**5 - 5.00905610e-3 * ln_T**6 +
      2.06723616e-4 * ln_T**7 - 3.64916141e-6 * ln_T**8 ) #cm 3 s -1 .
  return k3
  
def Collisional_Ionization_Rate_e_HeII_Abel97( temp ):
    temp = temp * K_to_ev
    ln_T = np.log(temp)    
    k5 = np.exp( -68.71040990 + 43.93347633 * ln_T - 18.4806699 * ln_T**2 + 4.70162649 * ln_T**3 -
        0.76924663 * ln_T**4 + 8.113042e-2 * ln_T**5 - 5.32402063e-3 * ln_T**6 +
        1.97570531e-4 * ln_T**7 - 3.16558106e-6 * ln_T**8 ) # cm 3 s -1 .  
    return k5
    
def Collisional_Ionization_Rate_HI_HI_Lenzuni91( temp ):
  k = 1.2e-17 * temp**(1.2) * np.exp(-157800 / temp )
  return k
  
def Collisional_Ionization_Rate_HII_HI_Lenzuni91( temp ):
  k = 9e-31 * temp**3
  return k

def Collisional_Ionization_Rate_HeI_HI_Lenzuni91( temp ):
  k = 1.75e-17 * temp**(1.3) * np.exp(-157800 / temp )
  return k
  
  
  
def Recombination_Rate_HII_Hui97( temp ):
  T_thr_HI = 157807
  lamda_HI = 2 * T_thr_HI / temp
  alpha = 1.269e-13 * lamda_HI**1.503 / ( 1 + (lamda_HI/0.522)**0.470 )**1.923
  return alpha
  
def Recombination_Rate_HeII_Hui97( temp ):
  T_thr_HeI = 285335
  lamda_HeI = 2 * T_thr_HeI / temp
  alpha = 3e-14 * lamda_HeI**0.654
  return alpha
  
def Recombination_Rate_HeIII_Hui97( temp ):
  T_thr_HeII = 631515
  lamda_HeII = 2 * T_thr_HeII / temp
  alpha = 2 * 1.269e-13 * lamda_HeII**1.503 / ( 1 + (lamda_HeII/0.522)**0.470 )**1.923
  return alpha
  

def Cooling_Rate_Recombination_HII_Hui97( n_e, n_HII, temp ):
  T_thr_HI = 157807
  lambda_HI = 2 * T_thr_HI / temp
  Lambda_HII = 1.778e-29 * temp * lambda_HI**1.965 / ( 1 + (lambda_HI/0.541)**0.502 )**2.697 * n_e * n_HII 
  return Lambda_HII
  
def Cooling_Rate_Recombination_HeII_Hui97( n_e, n_HII, temp ):
  T_thr_HeI = 285335
  lamda_HeI = 2 * T_thr_HeI / temp
  Lambda_HII = K_b * temp * 3e-14 * lamda_HeI**0.654 * n_e * n_HII 
  return Lambda_HII
  
def Cooling_Rate_Recombination_HeIII_Hui97( n_e, n_HII, temp ):
  T_thr_HeII = 631515
  lambda_HeII = 2 * T_thr_HeII / temp
  Lambda_HII = 8 * 1.778e-29 * temp * lambda_HeII**1.965 / ( 1 + (lambda_HeII/0.541)**0.502 )**2.697 * n_e * n_HII 
  return Lambda_HII
  
def Recombination_Rate_dielectronic_HeII_Hui97( temp ):
  T_thr_HeI = 285335
  T_thr_HeII = 631515
  lambda_HeI = 2 * T_thr_HeI / temp
  alpha = 1.9e-3 * temp**(-3./2) * np.exp( -0.75*lambda_HeI/2 ) * ( 1 + 0.3 * np.exp(-0.15 * lambda_HeI / 2 ) )
  return alpha
    
    
def Cooling_Rate_Recombination_dielectronic_HeII_Hui97( n_e, n_HeII, temp ):
  T_thr_HeI = 285335
  T_thr_HeII = 631515
  lambda_HeI = 2 * T_thr_HeI / temp
  Lambda = 0.75 * K_b * temp * 1.9e-3 * temp**(-3./2) * np.exp( -0.75*lambda_HeI/2 ) * ( 1 + 0.3 * np.exp(-0.15 * lambda_HeI / 2 ) ) * n_e * n_HeII
  return Lambda
  
def Collisional_Ionization_Rate_e_HI_Hui97( temp ):
  T_thr_HI = 157807
  lambda_HI = 2 * T_thr_HI / temp
  k = 21.11 * temp**(-3/2) * np.exp(-lambda_HI/2) * lambda_HI**(-1.089) / ( 1 + (lambda_HI/0.354)**0.874 )**1.101
  return k

def Cooling_Rate_Collisional_Excitation_e_HI_Hui97( n_e, n_HI, temp ):
  T_thr_HI = 157807
  lambda_HI = 2 * T_thr_HI / temp
  k = 7.5e-19 * np.exp( -0.75 * lambda_HI / 2 ) / ( 1 + (temp/1e5)**0.5 )
  return k * n_e * n_HI

def Cooling_Rate_Collisional_Excitation_e_HeII_Hui97( n_e, n_HeII, temp ):
  T_thr_HeII = 631515
  lambda_HeII = 2 * T_thr_HeII / temp
  k = 5.54e-17 * (1/temp)**0.397 * np.exp( -0.75 * lambda_HeII / 2 ) / ( 1 + (temp/1e5)**0.5 )
  return k * n_e * n_HeII

def Cooling_Rate_Compton_CMB_Peebles93( n_e, temp, current_z, cosmo ):
  # dQ_dt = 6.35e-41 * cosmo.Omega_b * cosmo.h**2 * X_e * ( 1 + z )**7 * ( 2.726*(1+z) - temp ) 
  dQ_dt = 6.35e-41 * cosmo.Omega_b * cosmo.h**2 * X_e * ( 1 + z )**7 * ( 2.726*(1+z) - temp ) 



#Compton cooling off the CMB 
def Cooling_Rate_Compton_CMB_MillesOstriker01( n_e, temp, z ):
  # M. Coleman Miller and Eve C. Ostriker 2001 (https://iopscience.iop.org/article/10.1086/323321/fulltext/)
  T_3 = temp / 1e3
  T_cm_3 = 0 # Don't know this value
  return 5.6e-33 * n_e * ( 1 + z )**4 * ( T_3 - T_cm_3 ) 
