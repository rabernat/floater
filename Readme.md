# Floater #

[![Build Status](https://travis-ci.org/rabernat/floater.svg?branch=master)](https://travis-ci.org/rabernat/floater)
[![codecov.io](https://codecov.io/github/rabernat/floater/coverage.svg?branch=master)](https://codecov.io/github/rabernat/floater?branch=master)


Transcode [MITgcm float output](http://mitgcm.org/) into:
* [PyTables](https://pytables.github.io/) HDF5 format
* [pandas](http://pandas.pydata.org/) HDF5 format
* [bcolz](http://bcolz.blosc.org/)
* [NetCDF](https://www.unidata.ucar.edu/software/netcdf/)

Transcoding is done via the `floater_convert` script, which is installed with the package.

```bash
$ floater_convert
usage: floater_convert [-h] [--float_file_prefix PREFIX] [--float_buf_dim N]
                       [--progress] [--input_dir DIR] [--output_format FMT]
                       [--keep_fields FIELDS] [--ref_time RT] [--step_time ST]
                       [--output_dir OD] [--output_prefix OP]
                       output_file
```

Also generators and analysis tools for Lagrangian trajectories.
