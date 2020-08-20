import numpy as np

def RK4_step( f, t, x, dt, **kargs ):
  k1 = dt * f( t, x, kargs=kargs )
  k2 = dt * f( t + 0.5*dt, x+0.5*k1, kargs=kargs )
  k3 = dt * f( t + 0.5*dt, x+0.5*k2, kargs=kargs )
  k4 = dt * f( t + dt, x+k3, kargs=kargs )
  return x + 1./6*( k1 + 2*k2 + 2*k3 + k4)


def f( x, coords, kargs=None):
  n = kargs['n']
  theta, y = coords
  dydt = np.array([y, - (theta ** n) - 2. * y / x])
  return dydt
