import pymc as pm
import sim
import numpy as np 
import arviz as az
import scipy
import matplotlib.pyplot as plt


h0 = 45.0
v0 = 3.5
t0 = 0.0
tf = 20.0
x0 = 15.5
# rvs
time_error = scipy.stats.norm(loc=0, scale=0.001 )
alt_error = scipy.stats.norm( loc=0, scale=2)
# generate simulated measurements
t_true=  np.arange( start=t0 ,stop=tf , step=0.1)
t_measured = t_true #+ time_error.rvs( size= len(t_true) ) 
x_measured = x0 + v0 * t_measured
#s = sim.Map_1D( "1dim.csv")
a_true =  h0 - sim.get_h( x_measured )
a_measured = a_true + alt_error.rvs( size=len(a_true))

plt.figure(1)
plt.plot( t_measured , a_measured , marker="." )
plt.xlabel("Time (non-dim)")
plt.ylabel("Radar Altimeter")