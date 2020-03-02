from __future__ import print_function

from floater import generators as gen
import numpy as np
import os
import pytest
import xarray as xr

domain_geometries = [
    # xlim, ylim, dx, dy, model_grid
    ((-50,50), (-50,50), 10,10, None)
    ]

grid_dict=dict(
    lon=np.arange(-30,30,1),
    lat=np.arange(-30,30,1),
    land_mask=np.ones((60,60))
    ) 


@pytest.fixture(params=domain_geometries)
def hex_masked_grid(request):
    xlim, ylim, dx, dy, _ = request.param
    test = gen.FloatSet(xlim, ylim, dx, dy,model_grid=grid_dict) 
    rec_g = np.asarray(test.get_rectmesh())
    hex_g = np.asarray(test.get_hexmesh())
    grids_parm=dict(hex=hex_g,dx=dx,dy=dy)
    return grids_parm

def test_hex_grid(hex_masked_grid):
    """Make sure we can create masked hexagonal grids correctly."""
    grids_parm=hex_masked_grid
    # Make sure the separation between grid points is constant
    # (i.e identical to dx)
    grid_diff=np.diff(grids_parm['hex'][0,:].reshape(grids_parm['dx'],grids_parm['dy']))
    assert np.all(grid_diff==grids_parm['dx'])

if __name__ == "__main__":
    import pylab as plt 
    xlim, ylim, dx, dy, _ = domain_geometries[0]

    test = gen.FloatSet(xlim, ylim, dx, dy,model_grid=grid_dict) 
    rec_g = np.asarray(test.get_rectmesh())
    hex_g = np.asarray(test.get_hexmesh())
    grids_parm=dict(rec=rec_g,hex=hex_g,dx=dx)

    dhex=grids_parm['hex']
    drec=grids_parm['rec']

    plt.plot(dhex[0,:],dhex[1,:],'x',label='Hexagonal Grid')
    plt.plot(drec[0,:],drec[1,:],'or',label='Rectangular Grid')
    plt.legend()
    plt.show()