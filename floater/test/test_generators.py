from __future__ import print_function

from floater import generators as gen
import numpy as np
import os
import pytest
import xarray as xr

# these are all combinations of (xlim, ylim, dx, dy) that will be fed to the
# different tests
domain_geometries = [
    # xlim, ylim, dx, dy, model_grid
    ((0,10), (0,20), 1, 1, None),
    ((10,20), (20,30), 1, 1, None),
    ((0.,10.), (0.,20.), 1., 1., None),
    ((10.,20.), (20,30.), 1., 1., None),
    ((10.,20.), (20,30.), 0.1, 0.1, None),
    # Anirban's example
    # https://github.com/rabernat/floater/issues/3
    ((-60.,-50.), (33.,40.5), 0.02, 0.015, None),
]

def _generate_model_grid(dg):
    xlim, ylim, dx, dy, _ = dg
    # use half the resoltion of the float set for the mask
    Nx = int((xlim[1] - xlim[0]) / dx)//2
    Ny = int((ylim[1] - ylim[0]) / dy)//2
    lon = np.linspace(xlim[0], xlim[1], Nx)
    lat = np.linspace(ylim[0], ylim[1], Ny)
    mask = np.ones((Ny, Nx), dtype='bool')
    # mask the upper half of the domain
    mask[Ny//2:] = False
    return {'lon': lon, 'lat': lat, 'land_mask': mask}

domain_geometries_with_land = [
    dg[:-1] + (_generate_model_grid(dg),) for dg in domain_geometries
]


@pytest.fixture(params=domain_geometries)
def fs(request):
    xlim, ylim, dx, dy, _ = request.param
    return gen.FloatSet(xlim, ylim, dx, dy)


@pytest.fixture(params=domain_geometries_with_land)
def fs_with_land(request):
    xlim, ylim, dx, dy, model_grid = request.param
    return gen.FloatSet(xlim, ylim, dx, dy, model_grid=model_grid)


@pytest.fixture(params=(domain_geometries+domain_geometries_with_land))
def fs_all(request):
    xlim, ylim, dx, dy, model_grid = request.param
    return gen.FloatSet(xlim, ylim, dx, dy, model_grid=model_grid)

@pytest.fixture(scope='module')
def fs_big():
    # Nathaniel's funky big domain
    #setup produces N = 16792975, exceeds float32 exact precision
    xlim = (0.0, 102.5)
    ylim = (-80.0,80.0)
    dx = 1/32.
    dy = 1/32.

    #mask grid resolution is five times as coarse as the float set
    Nx = int((xlim[1] - xlim[0]) / dx)/5
    Ny = int((ylim[1] - ylim[0]) / dy)/5
    lon = np.linspace(xlim[0], xlim[1], Nx)
    lat = np.linspace(ylim[0], ylim[1], Ny)

    #set entire domain to ocean
    mask = np.ones((Ny, Nx), dtype='bool')

    #make a small land island
    mask[5:10,5:10] = False

    model_grid = {'lon': lon, 'lat': lat, 'land_mask': mask}
    fs = gen.FloatSet(xlim, ylim, dx, dy, model_grid=model_grid)
    return fs


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

def test_land_mask(fs_with_land):
    """Verifies that the float grid excludes the masked regions properly."""

    fs = fs_with_land
    test_model = fs.model_grid
    # the top half of the domain should be masked
    grid_lat = test_model['lat']
    coast_lat = grid_lat[len(grid_lat)//2]

    #rect grid test
    float_x, float_y = fs.get_rectmesh()

    # check that all floats are within domain
    assert np.all((float_x >= fs.xlim[0] ) & ( float_x <= fs.xlim[1] ))
    assert np.all((float_y >= fs.ylim[0] ) & ( float_y <= fs.ylim[1] ))

    # check that the floats are beneath the coast (margin of error 2*dy)
    assert np.all(float_y <= coast_lat+2*fs.dy)

    num_floats = len(float_x)
    assert num_floats == len(float_y)
    #check that there are several floats
    assert num_floats > 1
    # check that something has been masked
    assert num_floats < (fs.Nx*fs.Ny)

    #hex grid test
    float_x, float_y = fs.get_hexmesh()

    # check that all floats are within domain
    assert np.all((float_x >= fs.xlim[0]) & (float_x <= fs.xlim[1]))
    assert np.all((float_y >= fs.ylim[0]) & (float_y <= fs.ylim[1]))

    # check that the floats are beneath the coast (margin of error 2*dy)
    assert np.all(float_y <= coast_lat+2*fs.dy)

    #check that there are several floats
    assert len(float_x) > 1
    assert len(float_y) > 1

def test_area(fs_all):
    """Kind of just a placeholder. Doesn't check actual values."""

    fs = fs_all
    a = fs.parcel_area()
    assert np.all(a > 0)


def test_to_mitgcm_format(fs_all, tmpdir):
    _actually_do_mitgcm_check(fs_all, tmpdir)

def test_pickling(fs, tmpdir):
    filename = str(tmpdir.join('pickled_floatset.pkl'))
    fs.to_pickle(filename)
    fs_from_pickle = gen.FloatSet(load_path=filename)

    for key in fs.__dict__.keys():
        assert np.all(fs.__dict__[key] == fs_from_pickle.__dict__[key])

def test_pickling_with_land(fs_with_land, tmpdir):
    fs = fs_with_land
    filename = str(tmpdir.join('pickled_floatset.pkl'))
    fs.to_pickle(filename)
    fs_from_pickle = gen.FloatSet(load_path=filename)

    for key in fs.__dict__.keys():
        if key is not 'model_grid':
            assert np.all(fs.__dict__[key] == fs_from_pickle.__dict__[key])
        else:
            for sub_key in fs.__dict__[key].keys():
                assert np.all(fs.__dict__[key][sub_key] == fs_from_pickle.__dict__[key][sub_key])


def test_npart_to_2D_array():
    # floatsets
    lon = np.linspace(0, 8, 9, dtype=np.float32)
    lat = np.linspace(-4, 4, 9, dtype=np.float32)
    land_mask = np.zeros(81, dtype=bool)==False
    land_mask.shape = (len(lat), len(lon))
    land_mask[:,0:2] = False
    model_grid = {'lon': lon, 'lat': lat, 'land_mask': land_mask}
    fs_none = gen.FloatSet(xlim=(0, 9), ylim=(-4, 5), dx=1.0, dy=1.0)
    fs_mask = gen.FloatSet(xlim=(0, 9), ylim=(-4, 5), dx=1.0, dy=1.0, model_grid=model_grid)
    # dataarray/dataset
    var_list = ['test_01', 'test_02', 'test_03']
    values_list_none = []
    values_list_mask = []
    data_vars_none = {}
    data_vars_mask = {}
    for var in var_list:
        values_none = np.random.random(81)
        values_none.shape = (1, 1, 81)
        values_mask = np.random.random(69)
        values_mask.shape = (1, 1, 69)
        values_list_none.append(values_none)
        values_list_mask.append(values_mask)
        data_vars_none.update({var: (['date', 'loc', 'npart'], values_none)})
        data_vars_mask.update({var: (['date', 'loc', 'npart'], values_mask)})
    npart_none = np.linspace(1, 81, 81, dtype=np.int32)
    npart_mask = np.linspace(1, 69, 69, dtype=np.int32)
    coords_none = {'date': (['date'], np.array([np.datetime64('2000-01-01')])),
                   'loc': (['loc'], np.array(['New York'])),
                   'npart': (['npart'], npart_none)}
    coords_mask = {'date': (['date'], np.array([np.datetime64('2000-01-01')])),
                   'loc': (['loc'], np.array(['New York'])),
                   'npart': (['npart'], npart_mask)}
    ds1d_none = xr.Dataset(data_vars=data_vars_none, coords=coords_none)
    ds1d_mask = xr.Dataset(data_vars=data_vars_mask, coords=coords_mask)
    da1d_none = ds1d_none['test_01']
    da1d_mask = ds1d_mask['test_01']
    # starts testing
    test_none = (fs_none, da1d_none, ds1d_none, values_list_none)
    test_mask = (fs_mask, da1d_mask, ds1d_mask, values_list_mask)
    test_list = [test_none, test_mask]
    for fs, da1d, ds1d, values_list in test_list:
        fs.get_rectmesh()
        # method test
        da2d = fs.npart_to_2D_array(da1d)
        ds2d = fs.npart_to_2D_array(ds1d)
        # shape test
        assert da2d.to_array().values.shape == (1, 1, 1, fs.Ny, fs.Nx)
        assert ds2d.to_array().values.shape == (3, 1, 1, fs.Ny, fs.Nx)
        # dimension test
        assert da2d.dims == {'date': 1, 'loc': 1, 'lat': 9, 'lon': 9}
        assert ds2d.dims == {'date': 1, 'loc': 1, 'lat': 9, 'lon': 9}
        # coordinates test
        np.testing.assert_allclose(da2d.lon.values, fs.x)
        np.testing.assert_allclose(da2d.lat.values, fs.y)
        np.testing.assert_allclose(ds2d.lon.values, fs.x)
        np.testing.assert_allclose(ds2d.lat.values, fs.y)
        # values test
        da1d_values = values_list[0][0][0]
        da2d_values_full = da2d.to_array().values[0].ravel()
        da2d_values = da2d_values_full[~np.isnan(da2d_values_full)]
        np.testing.assert_allclose(da2d_values, da1d_values)
        for i in range(3):
            ds1d_values = values_list[i][0][0]
            ds2d_values_full = ds2d.to_array().values[i].ravel()
            ds2d_values = ds2d_values_full[~np.isnan(ds2d_values_full)]
            np.testing.assert_allclose(ds2d_values, ds1d_values)
        # mask test
        if fs.model_grid is not None:
            mask1d = fs.ocean_bools
            mask2d_da = (np.isnan(da2d.to_array().values[0])==False).ravel()
            np.testing.assert_allclose(mask2d_da, mask1d)
            for i in range(3):
                mask2d_ds = (np.isnan(ds2d.to_array().values[i])==False).ravel()
                np.testing.assert_allclose(mask2d_ds, mask1d)


# Nathaniel's example
# https://github.com/rabernat/floater/issues/20
# http://stackoverflow.com/questions/15094611/behavior-of-float-that-is-used-as-an-integer
@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI because it's too slow.")
@pytest.mark.xfail(reason="number of floats too big to represent with 32-bit float")
def test_big_domain_32bit(fs_big, tmpdir):
    _actually_do_mitgcm_check(fs_big, tmpdir, prec=32)


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI because it's too slow.")
def test_big_domain_64bit(fs_big, tmpdir):
    _actually_do_mitgcm_check(fs_big, tmpdir, prec=64)

def _actually_do_mitgcm_check(single_fs, tmpdir, prec=32):
    fs = single_fs
    for mesh in ['rect', 'hex']:
        if mesh=='rect':
            xx, yy = fs.get_rectmesh()
        else:
            xx, yy = fs.get_hexmesh()
        num_floats = len(xx.ravel())
        for iup in [-1, 0, 1]:
            filename = str(tmpdir.join('ini_flt_pos_%s_%g.bin' % (mesh, iup)))
            fs.to_mitgcm_format(filename, mesh=mesh, iup=iup,
                                read_binary_prec=prec)
            # check the file exists
            assert os.path.exists(filename)
            # check the file has the correct length
            array = np.fromfile(filename, dtype='>f%g' % (prec/8))
            assert len(array)==(num_floats + 1 )* 9
            # check that the number of floats written to the file matches
            # the actual number of floats
            array = array.reshape(-1,9)
            assert int(array[0,0]) == num_floats
