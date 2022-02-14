import numpy as np

tevk = 1.1605e4
tiny = 1e-20
dhuge = 1.0e30
kboltz = 1.3806504e-16 #Boltzmann's constant [cm2gs-2K-1] or [ergK-1] 

  
######################################################################
# 1. HI collisional ionization
# Calculation of k1 (HI + e --> HII + 2e)
def k1_rate( T,  units=1 ):
  T_ev = T / 11605.0
  logT_ev = np.log(T_ev)

  k1 = np.exp( -32.71396786375
                        + 13.53655609057*logT_ev
                        - 5.739328757388*(logT_ev**2)
                        + 1.563154982022*(logT_ev**3)
                        - 0.2877056004391*(logT_ev**4)
                        + 0.03482559773736999*(logT_ev**5)
                        - 0.00263197617559*(logT_ev**6)
                        + 0.0001119543953861*(logT_ev**7)
                        - 2.039149852002e-6*(logT_ev**8)) / units;
  if T_ev <= 0.8:
    k1 = max(tiny, k1) 
  
  return k1;

coll_i_HI = k1_rate
######################################################################
# 2. HII Recombination 
# Calculation of k2 (HII + e --> HI + photon)
def k2_rate( T,  units=1, use_case_B=False ):

  if use_case_B:
    if T < 1.0e9:
      return 4.881357e-6*(T**-1.5) * ((1.0 + 1.14813e2*(T**-0.407))**-2.242) / units;
    else:
      return tiny;
          
  else:
    if T > 5500:
      # Convert temperature to appropriate form.
      T_ev = T / tevk;
      logT_ev = np.log(T_ev);

      return np.exp( -28.61303380689232 \
          - 0.7241125657826851*logT_ev \
          - 0.02026044731984691*(logT_ev**2) \
          - 0.002380861877349834*(logT_ev**3) \
          - 0.0003212605213188796*(logT_ev**4) \
          - 0.00001421502914054107*(logT_ev**5) \
          + 4.989108920299513e-6*(logT_ev**6) \
          + 5.755614137575758e-7*(logT_ev**7) \
          - 1.856767039775261e-8*(logT_ev**8) \
          - 3.071135243196595e-9*(logT_ev**9)) / units;
    else:
      return k4_rate(T, units=units )

recomb_HII = k2_rate      
######################################################################


# 3. HeI collisional ionization
# Calculation of k3 (HeI + e --> HeII + 2e)
def k3_rate( T,  units=1 ):
  T_ev = T / 11605.0
  logT_ev = np.log(T_ev);

  if T_ev > 0.8:
    return np.exp( -44.09864886561001
            + 23.91596563469*logT_ev
            - 10.75323019821*(logT_ev**2)
            + 3.058038757198*(logT_ev**3)
            - 0.5685118909884001*(logT_ev**4)
            + 0.06795391233790001*(logT_ev**5)
            - 0.005009056101857001*(logT_ev**6)
            + 0.0002067236157507*(logT_ev**7)
            - 3.649161410833e-6*(logT_ev**8)) / units
  else:
    return tiny

coll_i_HeI = k3_rate   
######################################################################

# 4. Recombination of HeII
# Calculation of k4 (HeII + e --> HeI + photon)
def k4_rate( T,  units=1, use_case_B=False ):

  T_ev = T / 11605.0
  logT_ev = np.log(T_ev)

  # If case B recombination on.
  if use_case_B:
    return 1.26e-14 * (5.7067e5/T)**0.75 / units

  # If case B recombination off.
  if T_ev > 0.8:
    return (1.54e-9*(1.0 + 0.3 / np.exp(8.099328789667/T_ev))
         / (np.exp(40.49664394833662/T_ev)*(T_ev**1.5))
         + 3.92e-13/(T_ev**0.6353)) / units
  else:
    return 3.92e-13/(T_ev**0.6353) / units

recomb_HeII = k4_rate  
######################################################################

# HeII Collisional ionization
# Calculation of k5 (HeII + e --> HeIII + 2e)
def k5_rate( T,  units=1 ):
  T_ev = T / 11605.0
  logT_ev = np.log(T_ev)

  if T_ev > 0.8:
    k5 = np.exp(-68.71040990212001
            + 43.93347632635*logT_ev
            - 18.48066993568*(logT_ev**2)
            + 4.701626486759002*(logT_ev**3)
            - 0.7692466334492*(logT_ev**4)
            + 0.08113042097303*(logT_ev**5)
            - 0.005324020628287001*(logT_ev**6)
            + 0.0001975705312221*(logT_ev**7)
            - 3.165581065665e-6*(logT_ev**8)) / units;
  else:
    k5 = tiny;
  return k5;

coll_i_HeII = k5_rate
######################################################################

# HeIII Recombination
# Calculation of k6 (HeIII + e --> HeII + photon)
def k6_rate( T,  units=1, use_case_B=False ):
  
  # Has case B recombination setting.
  if use_case_B:
      if T < 1.0e9:
        k6 = 7.8155e-5*(T**-1.5) * ((1.0 + 2.0189e2*(T**-0.407))**-2.242) / units
      else:
        k6 = tiny
  else:
    k6 = 3.36e-10/np.sqrt(T)/((T/1.0e3)**0.2) / (1.0 + ((T/1.0e6)**0.7)) / units;
  return k6

recomb_HeIII = k6_rate
######################################################################

# HI HI Collisional ionization
# Calculation of k57 (HI + HI --> HII + HI + e)
def k57_rate( T,  units=1 ):
  # These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
  # k57 value based on experimental cross-sections from Gealy & van Zyl (1987).
  if T > 3.0e3:
    return 1.2e-17 * (T**1.2) * np.exp(-1.578e5 / T) / units
  else:
    return tiny

coll_i_HI_HI = k57_rate
######################################################################

# HI HeI Collisional ionization
# Calculation of k58 (HI + HeI --> HII + HeI + e)
def k58_rate( T,  units=1 ):
  # These rate coefficients are from Lenzuni, Chernoff & Salpeter (1991).
  # k58 value based on cross-sections from van Zyl, Le & Amme (1981).
  if T > 3.0e3:
    return 1.75e-17 * (T**1.3) * np.exp(-1.578e5 / T) / units;
  else: 
    return tiny;

coll_i_HI_HeI = k58_rate    
######################################################################

# Cooling HI collisional excitation
# Calculation of ceHI.
def cool_ceHI_rate( T,  units=1 ):
  return 7.5e-19*np.exp( -min(np.log(dhuge), 118348.0 / T) ) / ( 1.0 + np.sqrt(T / 1.0e5) ) / units

######################################################################

# Cooling HeI collisional excitation
# Calculation of ceHeI.
def cool_ceHeI_rate( T, units=1 ):
  return 9.1e-27*np.exp(-min(np.log(dhuge), 13179.0/T)) * (T**-0.1687) / ( 1.0 + np.sqrt(T/1.0e5) ) / units
  
######################################################################

# Cooling HeII collisional excitation
# Calculation of ceHeII.
def cool_ceHeII_rate( T,  units=1 ):
  return 5.54e-17*np.exp(-min(np.log(dhuge), 473638.0/T)) * (T**-0.3970) / ( 1.0 + np.sqrt(T/1.0e5) ) / units
  
######################################################################

# Colling collisional ionization HeIs
# Calculation of ciHeIS.
def cool_ciHeIS_rate( T, units=1 ):
  return 5.01e-27*(T**-0.1687) / ( 1.0 + np.sqrt(T/1.0e5) ) * np.exp(-min(np.log(dhuge), 55338.0/T)) / units
   
######################################################################

# Colling collisional ionization HI
# Calculation of ciHI.
def cool_ciHI_rate( T, units=1 ):
  return 2.18e-11 * k1_rate(T, units=units) / units;
   
######################################################################

# Colling collisional ionization HeI
# Calculation of ciHeI.
def cool_ciHeI_rate( T, units=1 ):
  return 3.94e-11 * k3_rate(T, units=units) / units;
  
######################################################################

# Colling collisional ionization HeII
# Calculation of ciHeII.
def cool_ciHeII_rate( T, units=1 ):
  return 8.72e-11 * k5_rate(T, units=units) / units;
  
  
######################################################################

# Cooling Recombination HII
# Calculation of reHII.
def cool_reHII_rate( T,  units=1, use_case_B=False ):
  lambdaHI = 2.0 * 157807.0 / T;
  if use_case_B:
    return 3.435e-30 * T * (lambdaHI**1.970) / ( 1.0 + (lambdaHI/2.25)**0.376)**3.720 / units;
  else:
    return 1.778e-29 * T * (lambdaHI**1.965) / ( 1.0 + (lambdaHI/0.541)**0.502)**2.697 / units; 

######################################################################

# Cooling Recombination HeII
# Calculation of reHeII.
def cool_reHeII1_rate( T,  units=1, use_case_B=False ):
  lambdaHeII  = 2.0 * 285335.0 / T;
  if use_case_B:
    return 1.26e-14 * kboltz * T * (lambdaHeII**0.75) / units;
  else:
    return 3e-14 * kboltz * T * (lambdaHeII**0.654) / units;

######################################################################

# Dielectronic HeII recombination (Cen, 1992).
# Calculation of reHII2.
def cool_reHeII2_rate( T, units=1 ):
  return 1.24e-13 * (T**-1.5) * np.exp( -min(np.log(dhuge), 470000.0 / T) ) * ( 1.0 + 0.3 * np.exp( -min(np.log(dhuge), 94000.0 / T) ) ) / units;

######################################################################

# Cooling recombination HeIII
# Calculation of reHIII.
def cool_reHeIII_rate( T, units=1, use_case_B=False ):
  lambdaHeIII = 2.0 * 631515.0 / T;
  if use_case_B:
    return 8.0 * 3.435e-30 * T * (lambdaHeIII**1.970) / (1.0 + (lambdaHeIII / 2.25)**0.376)**3.720 / units;
  else:
    return 8.0 * 1.778e-29 * T * (lambdaHeIII**1.965) / (1.0 + (lambdaHeIII / 0.541)**0.502)**2.697/ units;

######################################################################

# Bremsstrahlung
# Calculation of brem.
def cool_brem_rate( T, units=1 ):
  return 1.43e-27 * np.sqrt(T) * ( 1.1 + 0.34 * np.exp( -(5.5 - np.log10(T))**2 / 3.0 ) ) / units;


######################################################################
  

