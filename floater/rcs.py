from __future__ import print_function

import os
import numpy as np
import pandas as pd
import bcolz
from floater.generators import FloatSet


class LavFloatData(object):

    def __init__(self, fs, bcolz_fname, reftime=None, timeunits='seconds',
            t_start=0, t_end=90):
    """Initialize new Lagrangian Averaged Vorticity Float Data object.

    Parameters
    ----------
    fs : FloatSet
        The object used to generate the original data. Has all the important
        grid information.
    bcolz_fname : path
        Path to the bcolz file where the data is stored
    reftime : datetime
        Reference datetime relative to which time is measures
    timeunits : str
        Units for time in bcolz data
    t_start : float
        Initial time for trajectory data
    t_end : float
        Final time for trajectory data
    """
        self.fs = fs
        self.bcolz_fname = bcolz_fname
        self.reftime = reftime
        self.timeunits = timeunits
        self.t_start = t_start
        self.t_end = t_end

        self._load_bcolz_data()
        self._calc_initial_final_position()
        self._calc_lav()

    def _load_bcolz_data(self):
        bc = bcolz.open(rootdir=self.bcolz_fname, mode='r')
        self.df = bc.todataframe()

    def _calc_initial_final_position(self):
        df_start = self.df.where(df.time==self.t_start)
        df_end = self.df.where(df.time==self.t_end)
        self.x0 = df_start.groupby('npart')['x'].first()
        self.y0 = df_start.groupby('npart')['y'].first()
        self.x1 = df_end.groupby('npart')['x'].first()
        self.y1 = df_end.groupby('npart')['y'].first()

    def _calc_lav(self):
        lav_series = self.df.groupby('npart')['vort'].mean()
        

