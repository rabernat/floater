"""Input formats for different types of float data.
"""
from __future__ import print_function

import numpy as np
import os
import glob
import sys

class MITgcmFloatData(object):
    """Everything we need to interact with MITgcm float output data."""

    def __init__(self, data_path, file_prefix='float_trajectories',
                 buf_dim=14, file_dtype=np.dtype('>f4'),
                 cast_to_dtype=np.dtype('f4'), keep_fields=None):
        """Initialize object with information about the location and datatype of
        the floats.

        Parameters
        ----------
        data_path : path
            The location of the output data on the filesystem
        file_prefix : str
        buf_dim : int
            Number of records in the float file
        file_dtype : numpy.dtype
            How the data is encoded in the files
        cast_to_dtype:
            Recast the output as this type
        keep_fields : list
            The names of the fields to retain (discard those not in the list)
        """

        self._data_path = data_path
        self._file_prefix = file_prefix
        self._buf_dim = buf_dim
        self._dtype = file_dtype

        if file_prefix == 'float_trajectories':
             # map buffer lengths to knowledge about MITgcm float output format
             assert self._buf_dim in [8, 13, 14]
             self.fields = ['npart', 'time', 'x', 'y', 'z', 'i', 'j', 'k']
             if buf_dim >= 9:
                 self.fields += ['p', 'u', 'v', 't', 's']
             if buf_dim >= 14:
                 self.fields += ['vort']
        elif file_prefix == 'float_profiles':
            self.fields = ['npart', 'time', 'x', 'y', 'z', 'i', 'j', 'k','etaN']
            dummylist=list(map(str, np.arange(1,(buf_dim-8)/4).astype(int)))
            self.fields += ['u' + s for s in dummylist]
            self.fields += ['v' + s for s in dummylist]
            self.fields += ['t' + s for s in dummylist]
            self.fields += ['s' + s for s in dummylist]
        else:
            raise RuntimeError('Unrecognized file_prefix: must be float_profiles or float_trajectories')
        self._bytes_per_float = self._buf_dim * self._dtype.itemsize
        # examine input files
        self.files = glob.glob(os.path.join(
                        self._data_path, self._file_prefix + '*.data'))
        if len(self.files)==0:
            raise RuntimeError('No float files found in %s' % float_dir)
        # calculate total number of float records in dataset
        self.nrecs = sum([self._nrecs_in_file(fname) for fname in self.files])

        # set up the datatype for each record
        # MITgcm writes the index as a float, even though it is an int
        # need to decide whether it makes sense to recast
        self.rec_dtype = np.dtype([ (k, self._dtype) for k in self.fields ])
        if (keep_fields is None) or (len(keep_fields)==0):
            self.out_dtype = self.rec_dtype
        else:
            self.out_dtype = np.dtype([ (k, self._dtype) for k in keep_fields ])
        # do typecasting here
        if cast_to_dtype is not None:
            self.out_dtype = np.dtype([
                    (k, cast_to_dtype) for k in self.out_dtype.names ])

    def generator(self, read_blocksize_mb=64, return_full_block=False,
                    progress=True
            ):
        """Returns a generator which will loop through every record in the
        dataset.

        Parameters
        ----------
        read_blocksize_mb : int
            The number of mb to read in at a time
        return_full_block : bool
            If ``True``, returns the whole block on each yield. Otherwise one
            record at a time
        progress : bool
            Report on progress to stdout
        """

        blocksize_read = int(read_blocksize_mb * 1e6 / self._bytes_per_float)
        for nproc, fname in enumerate(self.files):
            nrecs_file = self._nrecs_in_file(fname)
            with open(fname, 'rb') as f:
                # read the header and do nothing with it
                header = np.fromfile(f, dtype=self.rec_dtype, count=1)
                n = 0
                while True:
                    traj = np.fromfile(f, dtype=self.rec_dtype,
                                          count=blocksize_read
                                          ).astype(self.out_dtype)
                    nrecs = len(traj)
                    n += 1
                    if nrecs==0:
                        break
                    if progress:
                        status = ('Processing file %s (% 3d/% 3d) '
                                 'block % 5d/% 5.2f') % (
                                fname, nproc+1, len(self.files),
                                n, nrecs_file/float(blocksize_read))
                        sys.stdout.write("\r" + status)
                        sys.stdout.flush()
                    if return_full_block:
                        yield traj
                    else:
                        for rec in traj:
                            yield rec


    def _nrecs_in_file(self, fname):
        """Examine file size and figure out how many records are in it."""
        # subtract one because of the header
        return os.path.getsize(fname) / self._bytes_per_float - 1
