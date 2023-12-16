from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform 
import numpy as np
import terrain.pnw as T
columns = ["x", "dx", "h"]


N=10000


def dynamics( x0 ):
    o = np.zeros( shape=x0.shape)
    o[:,0] = x0[:,0] +  x0[:,1]*(1.0/N)
    o[:,1] = x0[:,1]
    o[:,2] = x0[:,2]
    return o
def altimeter( states ):
    x = states[:,0]
    h = states[:,2]
    map= h- T.pnw_1d.sample(x)
    return np.array([map,map]) 

def main( ):
    N = 10000

    v0 = 10 
    h0 = 45.0
    x0 = -130
    t_true=  np.linspace( start=0 ,stop=1, num=N )
    x_measured = x0 + v0 * t_true
    priors = independent_sample( [
        norm( loc=-130 , scale=.0001 ).rvs,
        norm( loc=v0, scale=0.001 ).rvs,
        norm( loc=8 , scale=.0001).rvs
        ])

    alt_error = norm( loc=0, scale=0.01 ).rvs( size=len(x_measured))
    alts = T.pnw_1d.sample(  x_measured )
    measures = h0 - alts + alt_error 

    pf = ParticleFilter( prior_fn=priors ,
                         observe_fn=altimeter,
                         n_particles=2,
                         dynamics_fn=dynamics,
                         weight_fn=lambda x,y:squared_error(x, y, sigma=2),
                         column_names=columns,
                         resample_proportion=0.1,

                         )
    history = np.zeros( (3, N ))
    k = 0
    for a in measures:
        pf.update( a )
        history[:,k] =  pf.map_state
        k+=1
    print("DoNE")

if __name__ == "__main__":
    main( )