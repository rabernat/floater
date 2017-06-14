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

_TESTDATA_FILENAME_CSV_OLD = 'sample_mitgcm_float_trajectories_csv_old.tar.gz'
_TMPDIR_SUBDIR_CSV_OLD = 'sample_mitgcm_data_csv_old'

_TESTDATA_FILENAME_CSV_NEW = 'sample_mitgcm_float_trajectories_csv_new.tar.gz'
_TMPDIR_SUBDIR_CSV_NEW = 'sample_mitgcm_data_csv_new'

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
def mitgcm_float_datadir_csv_old(tmpdir_factory, request):
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME_CSV_OLD)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    target_dir = str(tmpdir_factory.mktemp(_TMPDIR_SUBDIR_CSV_OLD))
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    return target_dir

@pytest.fixture(scope='module')
def mitgcm_float_datadir_csv_new(tmpdir_factory, request):
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME_CSV_NEW)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    target_dir = str(tmpdir_factory.mktemp(_TMPDIR_SUBDIR_CSV_NEW))
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

def test_floats_to_netcdf(tmpdir,
                          mitgcm_float_datadir_csv_old,
                          mitgcm_float_datadir_csv_new):
    """Test that we can convert MITgcm float data into NetCDF format.
    """
    import xarray as xr
    from floater.generators import FloatSet

    input_dir_old = str(mitgcm_float_datadir_csv_old)
    input_dir_new = str(mitgcm_float_datadir_csv_new)
    output_dir = str(tmpdir)
    fs = FloatSet(xlim=(-5, 5), ylim=(-2, 2))

    os.chdir(input_dir_old)
    fs.to_pickle('./fs.pkl')
    # least options
    utils.floats_to_netcdf(input_dir='./', output_fname='test_old')
    # most options
    utils.floats_to_netcdf(input_dir='./', output_fname='test_old',
                           float_file_prefix='float_trajectories',
                           ref_time='1993-01-01', pkl_path='./fs.pkl',
                           output_dir=output_dir, output_prefix='prefix_test')

    os.chdir(input_dir_new)
    fs.to_pickle('./fs.pkl')
    # least options
    utils.floats_to_netcdf(input_dir='./', output_fname='test_new')
    # most options
    utils.floats_to_netcdf(input_dir='./', output_fname='test_new',
                           float_file_prefix='float_trajectories',
                           ref_time='1993-01-01', pkl_path='./fs.pkl',
                           output_dir=output_dir, output_prefix='prefix_test')

    # filename prefix test
    os.chdir(input_dir_old)
    mfdol = xr.open_mfdataset('test_old_netcdf/float_trajectories.*.nc')
    os.chdir(input_dir_new)
    mfdnl = xr.open_mfdataset('test_new_netcdf/float_trajectories.*.nc')
    os.chdir(output_dir)
    mfdom = xr.open_mfdataset('test_old_netcdf/prefix_test.*.nc')
    mfdnm = xr.open_mfdataset('test_new_netcdf/prefix_test.*.nc')

    # dimensions test
    dims = [{'time': 2, 'npart': 40}, {'time': 2, 'y0': 4, 'x0': 10}]
    assert mfdol.dims == dims[0]
    assert mfdom.dims == dims[1]
    assert mfdnl.dims == dims[0]
    assert mfdnm.dims == dims[1]

    # variables and values test
    vars_values = [('x',  0.1961093750000000E+03), ('y',   -0.7848437500000000E+02),
                   ('z', -0.4999999999999893E+00), ('u',    0.3567512409555351E-04),
                   ('v',  0.1028276712547044E-03), ('vort', 0.0000000000000000E+00)]
    for var, value in vars_values:
        np.testing.assert_almost_equal(mfdol[var].values[0][0], value, 8)
        np.testing.assert_almost_equal(mfdom[var].values[0][0][0], value, 8)
    vars_values.append(('lavd', 0.0000000000000000E+00))
    for var, value in vars_values:
        np.testing.assert_almost_equal(mfdnl[var].values[0][0], value, 8)
        np.testing.assert_almost_equal(mfdnm[var].values[0][0][0], value, 8)

    # times test
    times = [(0, 0, np.datetime64('1993-01-01', 'ns')), (1, 2592000, np.datetime64('1993-01-31', 'ns'))]
    for i, sec, time in times:
        assert mfdol['time'][i].values == sec
        assert mfdom['time'][i].values == time
        assert mfdnl['time'][i].values == sec
        assert mfdnm['time'][i].values == time
