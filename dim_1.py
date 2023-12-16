import pymc as pm
import terrain.pnw as terrain
import numpy as np 
import arviz as az
import scipy
import matplotlib.pyplot as plt
import pytensor.tensor as at
import logging
logger = logging.getLogger("pymc")
logger.propagate = False
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger("numba")
logger2.setLevel(logging.ERROR)
logger3 = logging.getLogger("arviz")
logger3.setLevel(logging.ERROR)
class Solver:
    


    def __init__( self , x0 , v0, h0  , xBounds, vBounds, hBounds , sigma,  n ):
        #
        self.ter = terrain.pnw_1d
        self.h0 = h0 
        # velocity is non dimensionalized in units/time
        self.v0 = v0
        # time will be non-dim from 0-1
        t0 = 0.0 
        tf = 1.0
        # initial position
        self.x0 = x0
        #
        # rvs
        self.sigmaKm = sigma
        self.sigmaM = self.sigmaKm * 1000.0
        self.varM2 = self.sigmaM **2 
        self.tau   = 1.0 / self.varM2
        alt_error = scipy.stats.norm( loc=0, scale=self.sigmaKm ) 
        # generate simulated measurements
        self.t_measured=  np.arange( start=t0 ,stop=tf , step=(1.0/n))
        self.x_measured = x0 + v0 * self.t_measured
        a_true =  h0 - terrain.pnw_1d.sample( self.x_measured )
        self.a_measured = a_true + alt_error.rvs( size=len(a_true))
        # model bounds 
        self.xBounds=xBounds
        self.vBounds = vBounds
        self.hBounds = hBounds
        # create the model

        self.model = pm.Model()
    def plot_altimeter( self , fignum  ):
        fig = plt.figure(fignum)
        plt.plot( self.t_measured , self.a_measured , marker="." )
        plt.xlabel("Time (non-dim)")
        plt.ylabel("Radar Altimeter (km)")
    def plot_map( self , fignum ):
        self.ter.show_map( fignum  )
    def run(self, samples ):
        XL0 = self.xBounds[0]
        XU0 = self.xBounds[1]
        HL0 = self.hBounds[0]
        HU0 = self.hBounds[1]
        VL0 = self.vBounds[0]
        VH0 = self.vBounds[1]
        with self.model:
            # associate data with model (this makes prediction easier)
            H_data = pm.Data("Hm", self.a_measured, mutable=False)
            T_data = pm.Data("Tm", self.t_measured, mutable=True )
            # priors - uniform giving equal weight to all possible locations and velocities in range
            X0 = pm.Uniform("X0", lower=XL0, upper=XU0)  #pm.Uniform("x0", lower=10, upper=40) # pm.Normal( "X0" , mu=35, sigma=3)
            H0 = pm.Uniform("H0", lower=HL0, upper=HU0) # pm.Normal( "H0", mu=40, sigma=5 )
            V0 = pm.Uniform("V0", lower=VL0, upper=VH0)
            Sigma = pm.Gamma( "Sigma", alpha=100, beta=1)
            # X position through the terrain
            X = pm.Deterministic( "X" , X0 + (V0 * T_data)  )
            M = terrain.terrain_alt_1d( X )

            A = H0 - M 
            # Likelihood of altitude distribution
            Adist = pm.Normal( "Alimeter", mu=A , sigma=Sigma , observed=H_data )
            #
            # start sampling
            self.trace = pm.sample(samples, progressbar=False)

    def posterior_initial( self  , fignum ):
        plt.figure( fignum )
        az.plot_posterior( self.trace , var_names=["X0","H0","V0", "Sigma"])
    def plot_x_credible( self , fignum ,credset=0.95):
        xd = self.trace["posterior"]["X"][0,:,:]
        mx =  np.mean( xd , axis=0) 
        az.style.use("arviz-doc")
        x = az.plot_hdi( self.t_measured,  xd - self.x_measured , hdi_prob=credset)
        x.plot( self.t_measured , mx - self.x_measured )
        plt.xlabel("Time (non-dim)")
        plt.ylabel("X Error 95% Credible Set")
        plt.title("Posterior X Error Compared to Truth")
    def summary( self , variable ):
        o = az.summary( self.trace , var_names=["X0","H0","V0", "Sigma"] ) 
        return o
    def flight_summary( self ):

        text = " | Variable | Value | Units | Description |\n"
        text += " |-------- | ------ | ---- | ----------  |\n"
        text += f"| $X_0$ | {self.x0} | degrees | The initial longitude of the flight|\n"
        text += f"| $V_0$ | {self.v0} | degrees  | The velocity of the flight. Since time is non dimensional from 0-1 this is equivalent to the distance traveled|\n"
        text += f"| $H_0$ | {self.h0} | km | Altitude of the flight | \n"
        text += f"| $\sigma$ | {self.sigmaKm} | km | Standard deviation of noise on the altimeter measurement |\n"
        return text
    def prior_summary( self ):
        text = "| Prior | Lower Bound | Upper Bound |\n"
        text += "|-------| ------------| ----------- |\n"
        text += f"| $X_0$ | {self.xBounds[0]} | {self.xBounds[1]} |\n "
        text += f"| $V_0$ | {self.vBounds[0]} | {self.vBounds[1]} |\n "
        text += f"| $H_0$ | {self.hBounds[0]} | {self.hBounds[1]} |\n "
        return text