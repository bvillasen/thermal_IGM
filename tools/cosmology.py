import numpy as np


class Cosmology:
  
  def __init__(self):
    # Initializa Planck 2018 parameters
    self.H0 = 67.66
    self.Omega_M = 0.3111
    self.Omega_L = 0.6889
    self.h = self.H0 / 100.
    
  def get_Hubble( current_a ):
      a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  )  
      H = a_dot / current_a
      return H
  
  def get_dt( current_a, delta_a ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  )
    dt = delta_a / a_dot
    return dt  
    
  def get_delta_a( current_a, dt ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  )
    delta_a = dt * a_dot
    return delta_a  
    
      