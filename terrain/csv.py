# A simulation of steady level flight data over a known land mass
import numpy as np
import pandas as pd
import pymc as pm 
import pytensor.tensor as pt
import rasterio as rio
import matplotlib.pyplot as plt
import scipy.interpolate as si
from pytensor.compile.ops import as_op



class Csv_1D:
    def __init__( self , csvFile ):
        self.data = pd.read_csv(csvFile)
    def sample( self, x ):
        return  np.interp( x, self.data['x'], self.data['h'])

data = Csv_1D( "simple_1d.csv" )

@as_op(itypes=[pt.dvector],otypes=[pt.dvector])
def terrain_alt(  x ):
    terrainAlt = data.sample( x )
    return terrainAlt