import pymc as pm
import sim
import numpy as np 
import arviz as az
import scipy
import matplotlib.pyplot as plt

def doit():
    # Initial conditions 
    h0 = 45.0
    v0 = 2.5
    t0 = 0.0
    tf = 20.0
    x0 = 25.0
    # rvs
    time_error = scipy.stats.norm(loc=0, scale=0.001 )
    alt_error = scipy.stats.norm( loc=0, scale=0.001)
    # generate simulated measurements
    t_true=  np.arange( start=t0 ,stop=tf , step=1.0)
    t_measured = t_true #+ time_error.rvs( size= len(t_true) ) 
    x_measured = x0 + v0 * t_measured
    #s = sim.Map_1D( "1dim.csv")
    a_true =  h0 - sim.get_h( x_measured )
    a_measured = a_true #+ alt_error.rvs( size=len(a_true))
    # Now that we have the measurements setup the mcmc
    with pm.Model() as uPriors:
        # associate data with model (this makes prediction easier)
        H_data = pm.Data("Hm", a_measured, mutable=False)
        T_data = pm.Data("Tm", t_measured, mutable=True )
        # priors - uniform giving equal weight to all possible locations and velocities in range
        X0 = pm.Uniform("x0", lower=10, upper=40)
        H0 = pm.Uniform("h0", lower=40, upper=60)
        V0 = pm.Uniform("v0", lower=1, upper=3)
        # The t data is a 
        #T =  pm.Normal( "Time", mu=T_data, sigma=0.001  )
        # X position through the terrain 
        X = X0 + (V0 * T_data)
        # Altimeter measurement
        alt = H0-sim.get_h_wrap( X )
        #
        altDist = pm.Normal("altitude", mu=alt , sigma=0.001, observed=H_data)
        #
        # start sampling
        trace = pm.sample(5000)
    az.summary( trace )


if __name__=="__main__":
    print("wut?")
    doit()