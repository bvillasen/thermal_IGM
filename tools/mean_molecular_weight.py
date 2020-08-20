import numpy as np


def get_mean_molecular_weight( dens, HI_dens, HII_dens, HeI_dens, HeII_dens, HeIII_dens   ):
  mu =  dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 )
  return mu
  
