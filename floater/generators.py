"""For generating grids of floats."""

import numpy as np
from scipy.spatial import cKDTree


def geo_to_xyz(geo_cord):
    lon_deg, lat_deg = geo_cord.transpose()
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)
    xyz_cord = np.transpose(np.array((x,y,z)))
    return xyz_cord

def xyz_to_geo(xyz_coord):
    x,y,z = xyz_coord.transpose()
    lat = np.arcsin(z)
    lon = np.arctan2(y,x)
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)
    geo_cord = np.transpose(np.array((lon_deg, lat_deg)))
    return geo_cord


class FloatSet(object):
    """Represents a set of initial float positions on a regular grid."""

    def __init__(self, xlim, ylim, dx=1., dy=1.):
        """Initialize FloatSet according to specified rectangular grid geometry.

        PARAMETERS
        ----------
        xlim : tuple of two floats
            minimum and maximum x coordinate of cell vertices
        ylim : tuple of two floats
            minimum and maximum y coordinate of cell vertices
        dx : float
            grid spacing in x direction
        dy : float
            grid spacing in y direction
        """

        if not len(xlim)==2 and len(ylim)==2:
            raise ValueError('xlim and ylim should both by length 2.')
        self.xlim = [float(x) for x in xlim]
        self.ylim = [float(y) for y in ylim]
        self.Lx = xlim[1] - xlim[0]
        self.Ly = ylim[1] - ylim[0]
        if ((self.Lx*10.0**4.0) % (dx*10.0**4.0))/10.0**4.0 != 0.0:
            raise ValueError("Lx is not divisible evenly by dx")
        if ((self.Ly*10.0**4.0) % (dy*10.0**4.0))/10.0**4.0 != 0.0:
            raise ValueError("Ly is not divisible evenly by dy")
        self.dx = float(dx)
        self.dy = float(dy)
        self.Nx = int(self.Lx / self.dx)
        self.Ny = int(self.Ly / self.dy)
        self.x = self.xlim[0] + self.dx * np.arange(self.Nx) + self.dx/2
        self.y = self.ylim[0] + self.dy * np.arange(self.Ny) + self.dy/2

    def get_rectmesh(self):
        """Get the coordinates of the float positions in a rectangualr mesh.

        RETURNS
        -------
        x : np.ndarray
            2D array of float x coordinates
        y : np.ndarray
            2D array of float y coordinates
        """

        return np.meshgrid(self.x, self.y)

    def get_oceancoords(self, model_grid, mesh='rect'):
        """Get the coordinates of float positions taking into account the land mask.

        PARAMETERS
        ----------
        model_grid : dictionary 
            the following key value pairs are expected  
                'land_mask': np.ndarray of bools
                    2d array of dimensions len(lon) by len(lat).
                    An element is True iff the corresponding tracer cell grid point is unmasked (ocean)
                'lon': 1d array of the mask grid longitudes      
                'lat': 1d array of the mask grid latitudes
                     
         mesh : string
            options are 
                'rect' - a rectangular mesh, the default 
                'hex' - a hexagonal mesh 


        RETURNS
        -------
        floats_ocean: np.ndarray
            1D array of float coordinate subarrays:
            e.g. floats_ocean[i] = [some_float_lon, some_float_lat] 
        """

        xx, yy = self.get_rectmesh()

        if mesh == 'hex':
            xx[::2] += self.dx/4
            xx[1::2] -= self.dx/4

        mask_lon = model_grid['lon'] 
        mask_lat = model_grid['lat'] 
        land_mask = model_grid['land_mask']
        mask_geo = np.dstack(np.meshgrid(mask_lon, mask_lat)).reshape(-1, 2) # fast cartesian product
        mask_bool_flat = land_mask.ravel('F')
        mask_xyz = geo_to_xyz(mask_geo)
        mask_tree = cKDTree(mask_xyz) # a KDTree of the mask data in xyz form 
        floats_geo = np.transpose([xx.ravel(), yy.ravel()]) # uniform hexagonal tiling  
        queries_xyz = geo_to_xyz(floats_geo)
        # search for nearest neighbors
        dist, neighbor_indices = mask_tree.query(queries_xyz, n_jobs=-1) 
        ocean_bools = np.take(mask_bool_flat, neighbor_indices.ravel()) # True -> neighbor is tracer ocean
        floats_ocean = floats_geo[np.nonzero(ocean_bools.astype('int'))]

        return floats_ocean 

    def get_hexmesh(self):
        """Get the coordinates of the float positions in a hexagonal mesh.

        RETURNS
        -------
        x : np.ndarray
            2D array of float x coordinates
        y : np.ndarray
            2D array of float y coordinates
        """

        xx, yy = self.get_rectmesh()
        # modify to be even-R horizontal offset
        xx[::2] += self.dx/4
        xx[1::2] -= self.dx/4

        return xx, yy

    def npart_index_to_ndarray(self, data, npart):
        """Grid the data according to its npart (particle id)

        PARAMETERS
        ----------
        data : 1D array
            The data to be gridded
        npart : 1D array
            The particle ids
        """
        pass

    def parcel_area(self, latlon=True):
        """Get the area of each parcel."""

        if latlon==False:
            # cartesian grid
            return self.dx * self.dy
        else:
            R = 6.371e6
            lon, lat = self.get_rectmesh()
            dy = R * np.radians(self.dy)                            
            dx = R * np.radians(self.dx) * np.cos(np.radians(lat))  
            # old code, wrong!
            #dy = self.dy * R / 360.
            #dx = dy * np.cos(np.radians(lat)) * self.dx * R / 360.
            return dx * dy

    def to_mitgcm_format(self, filename, tstart=0, iup=0, mesh='rect', model_grid=None):
        """Output floatset in MITgcm format
        PARAMETERS
        ----------
        filename : The filename to save the floatset data in 
               (e.g.float.ini.pos.hex.bin)
        tstart : time for float initialisation (default = 0)
        iup : flag if the float
             - should profile ( > 0 = return cycle (in s) to surface)
             - remain at depth ( = 0 )
             - is a 3D float ( = -1 )
             - should be advected WITHOUT additional noise (= -2 ); 
        this implies that the float is non-profiling
             - is a mooring ( = -3 ); i.e. the float is not advected
        mesh : choice of mesh
         - 'rect' : rectangular cartesian
         - 'hex' : hexagonal
        model_grid : dictionary 
            - expected key value pairs   
                'land_mask': np.ndarray of bools
                        2d array of dimensions len(lon) by len(lat).
                        An element is True iff the corresponding tracer cell center point is unmasked (ocean)
                'lon': 1d array of the model grid tracer center longitudes      
                'lat': 1d array of the model grid tracer center latitudes
        """ 
        if model_grid is None: 
            if mesh == 'hex':
                xx, yy = self.get_hexmesh()
            else:
                xx, yy = self.get_rectmesh()
            myx = xx

            ini_times = 1

            # initial positions
            lon = myx.ravel()
            lat = yy.ravel()
        else:
            #place floats just in ocean
            lon, lat = np.transpose(self.get_oceancoords(model_grid, mesh))
            
        # other float properties

        # kpart: depth of float release in meters, depth is negative, i.e. -1500
        # for 1500 m
        #kpart = -0.5 # for 3d float
        kpart = -0.5

        # kfloat: target level of float (??)
        kfloat = -0.5

        # itop: time of float at the surface (in s)

        itop = 0
        # end time of integration of float (in s); note if tend = 1 floats are
        # integrated till the end of the integration;
        tend = -1;

        # initialization
        #tstart = 259200;

        # number of floats
        # previous way: N = self.Nx * self.Ny
        # which was was wrong for masked cases
        N = len(lon)

        output_dtype = np.dtype('>f4')
        # for all the float data
        flt_matrix = np.zeros((N+1,9), dtype=output_dtype)

        flt_matrix[1:,0] = np.arange(N)+1
        flt_matrix[1:,1] = tstart
        flt_matrix[1:,2] = lon
        flt_matrix[1:,3] = lat
        flt_matrix[1:,4] = kpart
        flt_matrix[1:,5] = kfloat
        flt_matrix[1:,6] = iup
        flt_matrix[1:,7] = itop
        flt_matrix[1:,8] = tend

        # first line in initialization file contains a record with
        # - the number of floats on that tile in the first record
        # - the total number of floats in the sixth record

        # original
        # flt_matrix[0,0] = N;
        # flt_matrix[1,0] = -1;
        # flt_matrix[5,0] = N;
        # flt_matrix[8,0] = -1;

        # copied from Andreas
        flt_matrix[0,0] = N;
        flt_matrix[0,1] = -1
        flt_matrix[0,4] = -1
        flt_matrix[0,5] = N
        flt_matrix[0,8] = -1

        #fname = os.path.join(output_dir, output_fname)
        return flt_matrix.tofile(filename)