from __future__ import print_function

import pytest
import os
import tarfile
import numpy as np

from floater import utils

_TESTDATA_FILENAME = 'sample_mitgcm_float_trajectories.tar.gz'
_TMPDIR_SUBDIR = 'sample_mitgcm_data'
_NAMES = ['npart', 'time', 'x', 'y', 'z', 'i', 'j', 'k', 'p',
          'u', 'v', 't', 's', 'vort']
_TESTVALS_FIRST = (1127925.0, 6134400.0, 247.69285583496094, -63.59305191040039,
                -0.5, 2477.428466796875, 164.56948852539062, 0.5049999952316284,
                 -0.08639287948608398, 0.12957383692264557, -0.12062723934650421,
                 0.0, 0.0, 2.6598372642183676e-06)

_TESTDATA_FILENAME_CSV = 'sample_mitgcm_float_trajectories_csv.tar.gz'
_TMPDIR_SUBDIR_CSV = 'sample_mitgcm_data_csv'


#@pytest.fixture()
#def empty_output_dir(tmpdir):
#    return tmpdir.mkdir('test')

@pytest.fixture(scope='module')
def mitgcm_float_datadir(tmpdir_factory, request):
    # find the tar archive in the test directory
    # http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    # tmpdir_factory returns LocalPath objects
    # for stuff to work, has to be converted to string
    target_dir = str(tmpdir_factory.mktemp(_TMPDIR_SUBDIR))
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    return target_dir

@pytest.fixture(scope='module')
def mitgcm_float_datadir_csv(tmpdir_factory, request):
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME_CSV)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    target_dir = str(tmpdir_factory.mktemp(_TMPDIR_SUBDIR_CSV))
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    return target_dir

def test_floats_to_bcolz(tmpdir, mitgcm_float_datadir):
    """Test that we can convert MITgcm float data into bcolz format.
    """
    # fixutre give a py.path.LocalPath object
    # need to convert to string
    import bcolz

    output_dir = str(tmpdir)
    input_dir = str(mitgcm_float_datadir)

    print('input_dir:', input_dir)
    print('output_dir:', output_dir)

    output_dtype = np.dtype('f4')

    data = utils.floats_to_bcolz(input_dir, output_dir)
    # test with progress bar
    data =  utils.floats_to_bcolz(input_dir, output_dir, progress=True)

    # now read the bcolz data
    bcolz_dir = output_dir + '.bcolz'
    bc = bcolz.open(rootdir=bcolz_dir)

    ctables_attrs_check = {'nbytes': 1606584, 'ndim': 1, 'names': _NAMES,
                           'shape': (28689,), 'size': 28689}
    for attr in ctables_attrs_check:
        assert getattr(bc, attr) == ctables_attrs_check[attr]

    # datatype is compound
    for name, dt in bc.dtype.descr:
        assert output_dtype == np.dtype(dt)

    # now make sure the values are in the correct range
    for name, val in zip(_NAMES, _TESTVALS_FIRST):
        np.testing.assert_almost_equal(bc[0][name], val)

def test_floats_to_netcdf(tmpdir, mitgcm_float_datadir_csv):
    """Test that we can convert MITgcm float data into NetCDF format.
    """
    import xarray as xr

    input_dir = str(mitgcm_float_datadir_csv) + '/'
    output_dir = str(tmpdir) + '/'
    os.chdir(input_dir)

    # least options
    utils.floats_to_netcdf(input_dir=input_dir, output_fname='test')
    # most options
    utils.floats_to_netcdf(input_dir=input_dir, output_fname='test',
                           float_file_prefix='float_trajectories',
                           ref_time='1993-01-01',
                           output_dir=output_dir, output_prefix='prefix_test')

    # filename prefix test
    os.chdir(input_dir)
    mfdl = xr.open_mfdataset('test_netcdf/float_trajectories.*.nc')
    os.chdir(output_dir)
    mfdm = xr.open_mfdataset('test_netcdf/prefix_test.*.nc')

    # dimensions test
    dims = {'npart': 40, 'time': 2}
    assert mfdl.dims == dims
    assert mfdm.dims == dims

    # variables and values test
    vars_values = [('x',  0.3237109375000000e+03), ('y',   -0.7798437500000000e+02),
                   ('z', -0.4999999999999893e+00), ('u',   -0.5346306607990328e-02),
                   ('v', -0.2787361934305595e-02), ('vort', 0.9160626946271506e-10)]
    for var, value in vars_values:
        np.testing.assert_almost_equal(mfdl[var].values[0][0], value, 8)
        np.testing.assert_almost_equal(mfdm[var].values[0][0], value, 8)

    # times test
    times = [(0, 0, np.datetime64('1993-01-01', 'ns')), (1, 86400, np.datetime64('1993-01-02', 'ns'))]
    for i, sec, time in times:
        assert mfdl['time'][i].values == sec
        assert mfdm['time'][i].values == time
