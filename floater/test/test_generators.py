from floater import generators as gen
import numpy as np
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
    print(request.param)
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

def test_area(fs):
    """Kind of just a placeholder. Doesn't check actual values."""

    a = fs.parcel_area()
    assert np.all(a > 0)
