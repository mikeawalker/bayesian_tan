import pymc as pm
import pymc.math as pmm
import terrain.pnw as terrain
import numpy as np 
import arviz as az
import scipy
import matplotlib.pyplot as plt
import pytensor.tensor as at
import logging
logger = logging.getLogger("pymc")
logger.propagate = False

class Solver:


    def __init__( self , theta, phi, gamma , v0, h0  , muTheta, muPhi, muV, thSigma, vSigma , sigmaH,  n ):
        #
        self.ter = terrain.pnw_2d
        self.h0 = h0 
        # time will be non-dim from 0-1
        t0 = 0.0 
        tf = 1.0
        # initial position
        self.theta0 = theta
        self.phi0 = phi 
        # Heading 
        self.gamma = gamma
        # velocity is non dimensionalized in units/time
        self.v0 = v0
        self.vPhi = self.v0 * np.sin( np.deg2rad( self.gamma) )
        self.vTheta = self.v0 * np.cos( np.deg2rad( self.gamma ) )
        # rvs
        self.sigmaKm = sigmaH
        alt_error = scipy.stats.norm( loc=0, scale=self.sigmaKm ) 
        # generate simulated measurements
        self.t_measured=  np.arange( start=t0 ,stop=tf , step=(1.0/n))
        
        self.theta_measured = self.theta0 + self.vTheta * self.t_measured
        self.phi_measured  = self.phi0 + self.vPhi * self.t_measured
        a_true =  h0 - terrain.pnw_2d.sample( long=self.theta_measured , lat=self.phi_measured  )
        self.a_measured = a_true + alt_error.rvs( size=len(a_true))
        # prior info
        self.muTheta = muTheta
        self.muPhi = muPhi
        self.muV = muV
        self.sigmaAng = thSigma
        self.sigmaV = vSigma 
        self.sigmaH = sigmaH
        
    def plot_altimeter( self , fignum  ):
        fig = plt.figure(fignum)
        plt.plot( self.t_measured , self.a_measured , marker="." )
        plt.xlabel("Time (non-dim)")
        plt.ylabel("Radar Altimeter (km)")
    def polygonify( self, x,y):
        ps = np.array( x ) + 1j*np.array( y )
        sidx = 0
        r = ps[sidx]
        ps = np.delete( ps , sidx )
        angs = np.angle( ps - r)
        dists = np.abs( ps - r)
        sorted = np.lexsort( (-dists, angs ))
        xPoly= np.array(x)[ sorted +1]
        yPoly= np.array(y)[ sorted+1]
        
        xPoly = np.insert( xPoly, 0 , x[0])
        yPoly = np.insert( yPoly, 0 , y[0])
        xPoly = np.append( xPoly, xPoly[0])
        yPoly = np.append( yPoly, yPoly[0])
        return xPoly,yPoly
    def plot_map( self , fignum ):
        self.ter.show_map( fignum  )
        lons0 = [ self.thetaBounds[0], self.thetaBounds[0] , self.thetaBounds[1] , self.thetaBounds[1]]
        lats0 = [ self.phiBounds[0], self.phiBounds[1], self.phiBounds[1], self.phiBounds[0]]
        latsf = []
        lonsf = []
        #
        for th,ph in zip( lons0, lats0):
            # lower
            latsf.append( ph + self.v0 * np.sin( np.pi * self.gammaBounds[0] / 180 )) 
            lonsf.append( th + self.v0 * np.cos( np.pi * self.gammaBounds[0] / 180 ))
            # upper
            latsf.append( ph + self.v0 * np.sin( np.pi * self.gammaBounds[1] / 180 )) 
            lonsf.append( th + self.v0 * np.cos( np.pi * self.gammaBounds[1] / 180 ))

        lons0,lats0 = self.polygonify( lons0, lats0 )
        lonsf,latsf = self.polygonify( lonsf, latsf )
        left = np.min( np.concatenate( (lons0, lonsf) ) ) 
        bot = np.min( np.concatenate( (lats0,latsf)))
        right = np.max( np.concatenate((lons0,lonsf) )) 
        top = np.max( np.concatenate( (lats0,latsf))) 
        plt.plot(lons0, lats0 , color="r")
        plt.plot( lonsf, latsf, color="g"  )
        plt.xlim( [ left-1 , right+1 ])
        plt.ylim( [bot-1 , top+1] )
        plt.show()

    def run(self, samples ):
        # create the model
        self.model = pm.Model()

        with self.model:
            # associate data with model (this makes prediction easier)
            H_data = pm.Data("Hm", self.a_measured, mutable=False)
            T_data = pm.Data("Tm", self.t_measured, mutable=True )
            # priors - uniform giving equal weight to all possible locations and velocities in range
            θ0 = pm.Normal("θ0", mu=self.muTheta, sigma=self.sigmaAng)  
            φ0 = pm.Normal("φ0", mu=self.muPhi, sigma=self.sigmaAng) 
            γ0 = self.gamma # pm.Normal("γ", mu=self.muGamma, sigma=self.sigmaGamma ) 
            σ =  pm.Gamma( "σ", alpha=100, beta=1) 
            V0 = pm.Normal( "V0", mu=self.muV,  sigma=self.sigmaV  )
            H0 = 8#pm.Uniform("H0", lower=7.8 , upper=8.2)
            # X position through the terrain
            Vθ = V0 * pmm.cos( np.pi * γ0 / 180.0)
            Vφ = V0 * pmm.sin( np.pi * γ0 / 180.0 )
            θ = pm.Deterministic( "θ" , θ0 + (Vθ * T_data)  )
            φ = pm.Deterministic( "φ" , φ0 + (Vφ * T_data)  )
            T = terrain.terrain_alt_2d(θ , φ )

            # Likelihood of altitude distribution
            AltLike = pm.Normal( "Alimeter", mu=H0-T , sigma=σ , observed=H_data )
            #
            # start sampling
            self.trace = pm.sample(samples, progressbar=True, chains=4 , cores=12)

    def posterior_initial( self  , fignum ):
        plt.figure( fignum )
        az.plot_posterior( self.trace , var_names=["θ0","φ0","V0", "σ"])
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
        self.sum = az.summary( self.trace , var_names=["θ0","φ0","V0", "σ"] ) 
        return self.sum
    def flight_summary( self ):

        text = " | Variable | Value | Units | Description |\n"
        text += " |-------- | ------ | ---- | ----------  |\n"
        text += fr"| $\theta_0$ | {self.theta0} | degrees | The initial longitude of the flight|"
        text += f"\n| $\phi_0$ | {self.phi0} | degrees | The initial latitude of the flight|\n"
        text += f"| $\gamma_0$ | {self.gamma} | degrees | The initial heading of the flight|\n"
        text += f"| $V_0$ | {self.v0} | --  | The velocity of the flight. Since time is non dimensional from 0-1 this is similar to the distance traveled. However it is not in degrees because you cant combine angle values like scalar positions. |\n"
        text += f"| $H_0$ | {self.h0} | km | Altitude of the flight | \n"
        text += f"| $\sigma$ | {self.sigmaKm} | km | Standard deviation of noise on the altimeter measurement |\n"
        return text
    def prior_summary( self ):
        text = "| Prior | Lower Bound | Upper Bound |\n"
        text += "|-------| ------------| ----------- |\n"
        text += fr"| $\theta_0$ | {self.thetaBounds[0]} | {self.thetaBounds[1]} | "
        text += f"\n| $\phi_0$ | {self.phiBounds[0]} | {self.phiBounds[1]} |\n "
        text += f"| $\gamma_0$ | {self.gammaBounds[0]} | {self.gammaBounds[1]} |\n "
        return text
    

if __name__ == "__main__":

    s = Solver( theta=-122.253748, phi=47.588818, gamma=45, v0=10, h0=8, muTheta=-122.2, muPhi=47.4, muV=9.9, vSigma=0.05, thSigma=0.05, sigmaH=0.01 , n =10000 ) 
    s.run( samples = 10000 )
    #s.summary(1)
    #az.plot_posterior( s.trace )
    s.plot_map(2)
    print( "done")