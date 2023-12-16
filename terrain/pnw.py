import terrain.tiff as T
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import numpy as np
pnw_1d = T.Terrain_1D("gt30_pnw.tif",52 , "lat")
pnw_2d = T.Terrain_2D("gt30_pnw.tif")


@as_op(itypes=[pt.dvector],otypes=[pt.dvector])
def terrain_alt_1d(  x ):
    terrainAlt = pnw_1d.sample( x )

    return terrainAlt



@as_op(itypes=[pt.dvector, pt.dvector, ],otypes=[pt.dvector])
def terrain_alt_2d(  long, lat ):

    terrainAlt = pnw_2d.sample( lat=lat ,long=long )

    return terrainAlt
