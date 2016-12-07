from floater import generators as gen
import numpy as np
import os
import pytest

# these are all combinations of (xlim, ylim, dx, dy) that will be fed to the
# different tests
@pytest.fixture(params=[
    # should work with integers and floats the same way
    ((0,10), (0,20), 1, 1,),
    ((10,20), (20,30), 1, 1,),
    ((0.,10.), (0.,20.), 1., 1.,),
    ((10.,20.), (20,30.), 1., 1.,),
    ((10.,20.), (20,30.), 0.1, 0.1,),
    # Anirban's example
    # https://github.com/rabernat/floater/issues/3
    ((-60.,-50.), (33.,40.5), 0.02, 0.015)
])
def fs(request):
    xlim, ylim, dx, dy = request.param
    return gen.FloatSet(xlim, ylim, dx, dy)

def test_float_set_shape(fs):
    """Make sure we can create FloatSet objects correctly."""
    # check shape
    nx, ny = (fs.xlim[1]-fs.xlim[0])/fs.dx, (fs.ylim[1]-fs.ylim[0])/fs.dy
    assert (fs.Nx, fs.Ny) == (nx, ny)

def test_rectmesh(fs):
    """Make sure the rectangular grid looks right."""

    x, y = fs.get_rectmesh()
    assert x.shape == (fs.Ny, fs.Nx)
    assert y.shape == (fs.Ny, fs.Nx)
    # make sure the first point is where we expect it
    assert np.allclose(x[0,0], fs.xlim[0] + fs.dx/2.)
    assert np.allclose(y[0,0], fs.ylim[0] + fs.dy/2.)
    # make sure the last point is where we expect it
    assert np.allclose(x[-1,-1], fs.xlim[1] - fs.dx/2.)
    assert np.allclose(y[-1,-1], fs.ylim[1] - fs.dy/2.)
    # make sure things are "rectangular"
    assert np.allclose(x[0], x[1])

def test_hexmesh(fs):
    """Make sure the rectangular grid looks right."""

    x, y = fs.get_hexmesh()
    assert x.shape == (fs.Ny, fs.Nx)
    assert y.shape == (fs.Ny, fs.Nx)
    # make sure the first point is where we expect it
    assert np.allclose(x[0,0], fs.xlim[0] + 3*fs.dx/4.)
    assert np.allclose(y[0,0], fs.ylim[0] + fs.dy/2.)
    # make sure things are "rectangular"
    assert np.allclose(x[0], x[1] + fs.dx/2)

def test_oceancoords(fs):
    """Verifies that the float grid excludes the masked regions properly."""

    #mask grid is half the spatial resolution of float set
    mask_lon = np.linspace(fs.xlim[0], fs.xlim[1], int(fs.Nx/2))
    mask_lat = np.linspace(fs.ylim[0], fs.ylim[1],int(fs.Ny/2))
    mask = np.ones((np.shape(mask_lon)[0], np.shape(mask_lat)[0]))
    

    #fill top half of the domain with land 
    coast_ind = int(len(mask_lat)/2) #index of coast 
    coast_lat = mask_lat[coast_ind]
    for j in range(np.shape(mask)[1]):
        if j >= coast_ind:
            mask[:,j] = 0
    
    test_model = {'lon': mask_lon, 'lat': mask_lat, 'land_mask': mask}

    #rect grid test 
    float_x, float_y = np.transpose(fs.get_oceancoords(test_model, mesh='rect'))
    
    # check that all floats are within domain 
    assert np.all((float_x >= fs.xlim[0] ) & ( float_x <= fs.xlim[1] ))
    assert np.all((float_y >= fs.ylim[0] ) & ( float_y <= fs.ylim[1] ))
    
    # check that the floats are beneath the coast (margin of error 2*dy)
    assert np.all(float_y <= coast_lat+2*fs.dy)
    
    #check that there are several floats
    assert len(float_x) > 1
    assert len(float_y) > 1

    #hex grid test 
    float_x, float_y = np.transpose(fs.get_oceancoords(test_model, mesh='hex'))
    
    # check that all floats are within domain 
    assert np.all((float_x >= fs.xlim[0]) & (float_x <= fs.xlim[1]))
    assert np.all((float_y >= fs.ylim[0]) & (float_y <= fs.ylim[1]))
    
    # check that the floats are beneath the coast (margin of error 2*dy)
    assert np.all(float_y <= coast_lat+2*fs.dy)

    #check that there are several floats
    assert len(float_x) > 1
    assert len(float_y) > 1
        
def test_area(fs):
    """Kind of just a placeholder. Doesn't check actual values."""

    a = fs.parcel_area()
    assert np.all(a > 0)

def test_to_mitgcm_format(fs, tmpdir):
    for mesh in ['rect', 'hex']:
        for iup in [-1, 0, 1]:
            filename = str(tmpdir.join('ini_flt_pos_%s_%g.bin' % (mesh, iup)))
            fs.to_mitgcm_format(filename, mesh=mesh, iup=iup)
	    assert os.path.exists(filename)
            array = np.fromfile(filename, dtype='>f4')
            assert len(array)==(fs.Nx*fs.Ny + 1 )* 9
	    # TODO : check each element of this array and 
            # check that the values are correct.   
