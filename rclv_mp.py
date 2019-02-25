from __future__ import print_function

import numpy as np
import xarray as xr
from skimage.measure import find_contours, points_in_poly, grid_points_in_poly
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image, watershed
from scipy.spatial import qhull
from time import time

import logging
logger = logging.getLogger(__name__)

R_earth = 6.371e6

def polygon_area(verts):
    """Compute the area of a polygon.

    Parameters
    ----------
    verts : array_like
        2D shape (N,2) array of vertices. Uses scikit image convetions
        (j,i indexing)

    Returns
    area : float
        Area of polygon enclolsed by verts. Sign is determined by vertex
        order (cc vs ccw)
    """
    verts_roll = np.roll(verts, 1, axis=0)
    # use scikit image convetions (j,i indexing)
    area_elements = ((verts_roll[:,1] + verts[:,1]) *
                     (verts_roll[:,0] - verts[:,0]))
    # absolute value makes results independent of orientation
    return abs(area_elements.sum())/2.0


def get_local_region(data, ji, border_j, border_i, periodic=(False, False)):
    logger.debug("get_local_region "
                 + repr(ji) + repr(border_i) + repr(border_j))
    j, i = ji
    nj, ni = data.shape
    jmin = j - border_j[0]
    jmax = j + border_j[1] + 1
    imin = i - border_i[0]
    imax = i + border_i[1] + 1

    # we could easily implement wrapping with take
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html
    # unfortunately, take is ~1000000 times slower than raw indexing and copies
    # the data. So we need to slice and concatenate

    concat_down = (jmin < 0)
    concat_up = (jmax > nj)
    concat_left = (imin < 0)
    concat_right = (imax > ni)

    # check for valid region limits
    if (concat_left or concat_right) and not periodic[1]:
        raise ValueError("Region i-axis limits " + repr((imin, imax)) +
                         " outside of array shape " + repr((nj,ni)) +
                         " and i-axis is not periodic")
    if (concat_up or concat_down) and not periodic[0]:
        raise ValueError("Region j-axis limits " + repr((jmin, jmax)) +
                         " outside of array shape " + repr((nj,ni)) +
                         " and j-axis is not periodic")
    if (concat_left and concat_right) or (concat_up and concat_down):
        raise ValueError("Can't concatenate on more than one side on the same "
                         "axis. Limits are " +
                         repr(((jmin, jmax), (imin, imax))))

    # limits for central region
    imin_reg = max(imin, 0)
    imax_reg = min(imax, ni)
    jmin_reg = max(jmin, 0)
    jmax_reg = min(jmax, nj)
    data_center = data[jmin_reg:jmax_reg, imin_reg:imax_reg]

    if concat_down:
        data_down = data[jmin:, imin_reg:imax_reg]
    if concat_up:
        data_up = data[:(jmax - nj), imin_reg:imax_reg]
    if concat_left:
        data_left= data[jmin_reg:jmax_reg, imin:]
    if concat_right:
        data_right = data[jmin_reg:jmax_reg, :(imax - ni)]

    # corner cases
    if concat_down and concat_left:
        data_down_left = data[jmin:, imin:]
    if concat_down and concat_right:
        data_down_right = data[jmin:, :(imax - ni)]
    if concat_up and concat_left:
        data_up_left = data[:(jmax - nj), imin:]
    if concat_up and concat_right:
        data_up_right = data[:(jmax - nj), :(imax - ni)]

    # now put things together, starting with the corner cases
    # it feels like there must be a more elegant way to do this
    if concat_down and concat_left:
        data_reg = np.concatenate(
                        (np.concatenate((data_down_left, data_down), axis=1),
                         np.concatenate((data_left, data_center), axis=1)),
                        axis=0)
    elif concat_down and concat_right:
        data_reg = np.concatenate(
                        (np.concatenate((data_down, data_down_right), axis=1),
                         np.concatenate((data_center, data_right), axis=1)),
                        axis=0)
    elif concat_up and concat_left:
        data_reg = np.concatenate(
                        (np.concatenate((data_left, data_center), axis=1),
                         np.concatenate((data_up_left, data_up), axis=1)),
                        axis=0)
    elif concat_up and concat_right:
        data_reg = np.concatenate(
                        (np.concatenate((data_center, data_right), axis=1),
                         np.concatenate((data_up, data_up_right), axis=1)),
                        axis=0)
    elif concat_down:
        data_reg = np.concatenate((data_down, data_center), axis=0)
    elif concat_up:
        data_reg = np.concatenate((data_center, data_up), axis=0)
    elif concat_left:
        data_reg = np.concatenate((data_left, data_center), axis=1)
    elif concat_right:
        data_reg = np.concatenate((data_center, data_right), axis=1)
    else:
        data_reg = data_center

    return (j - jmin, i - imin), data[j,i] - data_reg


def is_contour_closed(con):
    return np.all(con[0] == con[-1])


def point_in_contour(con, ji):
    j, i = ji
    return points_in_poly(np.array([i, j])[None], con[:,::-1])[0]


def contour_area(con):
    """Calculate the area, convex hull area, and convexity deficiency
    of a polygon contour.

    Parameters
    ----------
    con : arraylike
        A 2D array of vertices with shape (N,2) that follows the scikit
        image conventions (con[:,0] are j indices)

    Returns
    -------
    region_area : float
    hull_area : float
    convexity_deficiency : float
    """
    # reshape the data to x, y order
    con_points = con[:,::-1]

    # calculate area of polygon
    region_area = polygon_area(con_points)

    # find convex hull
    hull = qhull.ConvexHull(con_points)
    #hull_points = np.array([con_points[pt] for pt in hull.vertices])
    hull_area = hull.volume

    cd = (hull_area - region_area ) / region_area

    return region_area, hull_area, cd


def project_vertices(verts, lon0, lat0, dlon, dlat):
    """Project the logical coordinates of vertices into physical map
    coordiantes.

    Parameters
    ----------
    verts : arraylike
        A 2D array of vertices with shape (N,2) that follows the scikit
        image conventions (con[:,0] are j indices)
    lon0, lat0 : float
        center lon and lat for the projection
    dlon, dlat : float
        spacing of points in longitude
    dlat : float
        spacing of points in latitude

    Returns
    -------
    verts_proj : arraylike
        A 2D array of projected vertices with shape (N,2) that follows the
        scikit image conventions (con[:,0] are j indices)
    """

    i, j = verts[:, 1], verts[:, 0]

    # use the simplest local tangent plane projection
    dy = (np.pi * R_earth / 180.)
    dx = dy * np.cos(np.radians(lat0))
    x = dx * dlon *i
    y = dy * dlat * j

    return np.vstack([y, x]).T


def find_contour_around_maximum(data, ji, level, border_j=(5,5),
        border_i=(5,5), max_footprint=None, proj_kwargs={},
        periodic=(False, False)):
    j,i = ji
    max_val = data[j,i]

    # increments for increasing bounds of region
    delta_b = 5

    target_con = None
    grow_down, grow_up, grow_left, grow_right = 4*(False,)

    while target_con is None:

        footprint_area = sum(border_j) * sum(border_i)
        if max_footprint and footprint_area > max_footprint:
            raise ValueError('Footprint exceeded max_footprint')

        # maybe expand the border
        if grow_down:
            border_j = (border_j[0] + delta_b, border_j[1])
        if grow_up:
            border_j = (border_j[0], border_j[1] + delta_b)
        if grow_left:
            border_i = (border_i[0] + delta_b, border_i[1])
        if grow_right:
            border_i = (border_i[0], border_i[1] + delta_b)

        # find the local region
        (j_rel, i_rel), region_data = get_local_region(data, (j, i),
                                                       border_j, border_i,
                                                       periodic=periodic)
        nj, ni = region_data.shape

        # extract the contours
        contours = find_contours(region_data, level)

        if len(contours)==0:
            # no contours found, grow in all directions
            grow_down, grow_up, grow_left, grow_right = 4*(True,)

        # check each contour
        for con in contours:
            is_closed = is_contour_closed(con)
            is_inside = point_in_contour(con, (j_rel, i_rel))

            if is_inside and is_closed:
                # we found the right contour
                target_con = con
                break

            # check for is_inside doesn't work for non-closed contours
            grow_down |= (con[0][0] == 0) or (con[-1][0] == 0)
            grow_up |= (con[0][0] == nj-1) or (con[-1][0] == nj-1)
            grow_left |= (con[0][1] == 0) or (con[-1][1] == 0)
            grow_right |= (con[0][1] == ni-1) or (con[-1][1] == ni-1)

        # if we got here without growing the region in any direction,
        # we are probably in a weird situation where there is a closed
        # contour that does not enclose the maximum
        if target_con is None and not (
              grow_down or grow_up or grow_left or grow_right):
                raise ValueError("Couldn't find a contour")

    return target_con, region_data, border_j, border_i


def convex_contour_around_maximum(data, ji, init_contour_step_frac=0.1,
                                  border=5,
                                  convex_def=0.01, convex_def_tol=0.001,
                                  max_footprint=None, proj_kwargs=None,
                                  periodic=(False, False),
                                  max_iters=1000, min_limit_diff=1e-10):
    """Find the largest convex contour around a maximum.

    Parameters
    ----------
    data : array_like
        The 2D data to contour
    ji : tuple
        The index of the maximum in (j, i) order
    init_contour_step_frac : float
        the value with which to increment the initial contour level
        (multiplied by the local maximum value)
    border: int
        the initial window around the maximum
    convex_def : float, optional
        The target convexity deficiency allowed for the contour.
    convex_def_tol : float, optional
        The tolerance for which the convexity deficiency will be sought
    verbose : bool, optional
        Whether to print out diagnostic information
    proj_kwargs : dict, optional
        Information for projecting the contour into spatial coordinates. Should
        contain entries `lon0`, `lat0`, `dlon`, and `dlat`.
    periodic : tuple
        Tuple of bools which specificies the periodicity of each axis (j, i) of
        the data

    Returns
    -------
    contour : array_like
        2D array of contour vertices with shape (N,2) that follows
        the scikit image conventions (contour[:,0] are j indices)
    area : float
        The area enclosed by the contour
    cd : float
        The actual convexity deficiency of the identified contour
    """

    # the maximum
    j, i = ji

    # the initial search region
    border_j = (border, border)
    border_i = (border, border)
    max_value = data[tuple(ji)]

    logger.info("convex_contour_around_maximum " + repr(tuple(ji))
                + " max_value %g" % max_value)

    init_contour_step_size = max_value * init_contour_step_frac
    logger.debug("init_contour_step_frac %g" % init_contour_step_frac)
    logger.debug("init_contour_step_size %g" % init_contour_step_size)

    lower_lim = 0
    upper_lim = 2*init_contour_step_size
    # special logic needed to find the first contour greater than convex_def
    exceeded = False
    cd = np.inf
    contour = None
    region_area = None

    for n_iter in range(max_iters):
        logger.debug('iter %g, cd %g' % (n_iter, cd))
        if abs(cd - convex_def) <= convex_def_tol:
            logger.debug('cd %g is close to target %g within tolerance %g' %
                         (cd, convex_def, convex_def_tol))
            break

        # the new contour level to try
        logger.debug('current lims: (%10.20f, %10.20f)' % (lower_lim, upper_lim))
        level = 0.5*(upper_lim + lower_lim)

        if (upper_lim - lower_lim) < min_limit_diff:
            logger.debug('limit diff below threshold; ending search')
            break

        logger.debug(('contouring level: %10.20f border: ' % level)
                     + repr(border_j) + repr(border_i))
        try:
            # try to get a contour
            contour, region_data, border_j, border_i = find_contour_around_maximum(
                data, (j, i), level, border_j, border_i,
                max_footprint=max_footprint, periodic=periodic)
        except ValueError as ve:
            # we will probably be here if the contour search ended up covering
            # a too large area. What to do about this depends on whether we
            # have already found a contour
            logger.debug(repr(ve))
            #if contour is None:
            upper_lim = level
            exceeded = True
            continue
            #else:
            #    break

        # get the convexity deficiency
        if proj_kwargs is None:
            contour_proj = contour
        else:
            contour_proj = project_vertices(contour, **proj_kwargs)

        region_area, hull_area, cd = contour_area(contour_proj)
        logger.debug('region_area: % 6.1f, hull_area: % 6.1f, convex_def: % 6.5e'
                     % (region_area, hull_area, cd))

        # special logic needed to find the first contour greater than convex_def
        if not exceeded:
            if cd < convex_def:
                # need to keep upper_lim until we exceed the convex def
                lower_lim = level
                upper_lim = level + 2*init_contour_step_size
                logger.debug('still searching for upper_lim, new lims '
                             '(%g, %g)' % (lower_lim, upper_lim))
            else:
                exceeded = True
        else:
            # from here we can assume that the target contour lies between
            # lower_lim and_upper_lim
            if cd < convex_def:
                lower_lim = level
            else:
                upper_lim = level

    # now we should have a contour to return
    if contour is not None:
        contour[:, 0] += (j-border_j[0])
        contour[:, 1] += (i-border_i[0])
    return contour, region_area, cd


def find_convex_contours(data, min_distance=5, min_area=100.,
                             use_multi_process=False, lon=None, lat=None,
                             progress=False, **contour_kwargs):
    """Find the outermost convex contours around the maxima of
    data with specified convexity deficiency.

    Parameters
    ----------
    data : array_like
        The 2D data to contour
    min_distance : int, optional
        The minimum distance around maxima (pixel units)
    min_area : float, optional
        The minimum area of the regions (pixels or projected if `lon` and `lat`
        are specified)
    lon, lat : arraylike
        Longitude and latitude of data points. Should be 1D arrays such that
        ``len(lon) == data.shape[1]`` and ``len(lat) == data.shape[0]``
    init_contour_step_frac : float
        the value with which to increment the initial contour level
        (multiplied by the local maximum value)
    border: int
        the initial window around the maximum
    convex_def : float, optional
        The target convexity deficiency allowed for the contour.
    convex_def_tol : float, optional
        The tolerance for which the convexity deficiency will be sought
    verbose : bool, optional
        Whether to print out diagnostic information
    proj_kwargs : dict, optional
        Information for projecting the contour into spatial coordinates. Should
        contain entries `lon0`, `lat0`, `dlon`, and `dlat`.
    periodic : tuple
        Tuple of bools which specificies the periodicity of each axis (j, i) of
        the data

    Yields
    ------
    contour : array_like
        2D array of contour vertices with shape (N,2) that follows
        the scikit image conventions (contour[:,0] are j indices)
    area : float
        The area enclosed by the contour (in pixels or projected if
        `lon` and `lat` are specified)
    cd : float
        The actual convexity deficiency of the identified contour
    """

    # do some checks on the coordinates if they are specified
    if (lon is not None) or (lat is not None):
        if not ((len(lat) == data.shape[0]) and (len(lon) == data.shape[1])):
            raise ValueError('`lon` or `lat` have the incorrect length')
        dlon = lon[1] - lon[0]
        dlat = lat[1] - lat[0]
        # make sure that the lon and lat are evenly spaced
        if not (np.allclose(np.diff(lon), dlon) and
                np.allclose(np.diff(lat), dlat)):
            raise ValueError('`lon` and `lat` need to be evenly spaced')
        proj = True
    else:
        proj = False

    if use_multi_process:
        from pathos.multiprocessing import ProcessingPool as Pool
        #map_function = Pool(processes = 4).imap_unordered
        map_function = Pool().uimap
    else:
        try:
            from itertools import imap
            map_function = imap
        except ImportError:
            # must be python 3
            map_function = map

    plm = peak_local_max(data, min_distance=min_distance)

    def test_func(ji):
        contour, area, cd = convex_contour_around_maximum(data, ji,
                                **contour_kwargs)
        return ji, contour, area, cd
 
   # function to map
    def maybe_contour_maximum(ji):
        #tic = time()
        result = None
        '''
        if proj:
            contour_kwargs['proj_kwargs'] = {'lon0': lon[ji[1]],
                                             'lat0': lat[ji[0]],
                                             'dlon': dlon, 'dlat': dlat}
        else:
            if 'proj_kwargs' in contour_kwargs:
                del contour_kwargs['proj_kwargs']
        '''
        contour, area, cd = convex_contour_around_maximum(data, ji,
                                **contour_kwargs)
        if area and (area >= min_area):
            result = ji, contour, area, cd
        #toc = time()
        #logger.debug("point " + repr(tuple(ji)) + " took %g s" % (toc-tic))
        return result

    if progress:
        from tqdm import tqdm
    else:
        tqdm = _DummyTqdm
    with tqdm(total=len(plm)) as pbar:
        for item in map_function(maybe_contour_maximum, plm):
        #for item in map_function(test_func, plm):
            pbar.update(1)
            if item is not None:
                yield item


class _DummyTqdm:

    def __init__(*args, **kwargs):
        pass

    def __enter__(self):
        class dummy_pbar:
            def update(self, *args, **kwargs):
                pass
        return dummy_pbar()

    def __exit__(self, type, value, traceback):
        pass

def label_points_in_contours(shape, contours):
    """Label the points inside each contour.

    Parameters
    ----------
    shape : tuple
        Shape of the original domain from which the contours were detected
        (e.g. LAVD field)
    contours : list of vertices
        The contours to label (e.g. result of RCLV detection)

    Returns
    -------
    labels : array_like
        Array with contour labels assigned. Zero means not inside a contour
    """

    assert len(shape)==2
    ny, nx = shape

    # modify data in place with this function
    def fill_in_contour(contour, label_data, value=1):
        ymin, xmin = (np.floor(contour.min(axis=0)) - 1).astype('int')
        ymax, xmax = (np.ceil(contour.max(axis=0)) + 1).astype('int')
        # possibly roll the data to deal with periodicity
        roll_x, roll_y = 0, 0
        if ymin < 0:
            roll_y = -ymin
        if ymax > ny:
            roll_y = ny - ymax
        if xmin < 0:
            roll_x = -xmin
        if xmax > nx:
            roll_x = nx - xmax

        contour_rel = contour - np.array([ymin, xmin])

        ymax += roll_y
        ymin += roll_y
        xmax += roll_x
        xmin += roll_x

	    # only roll if necessary
        if roll_x or roll_y:
            data = np.roll(np.roll(label_data, roll_x, axis=1), roll_y, axis=0)
        else:
            data = label_data
        region_slice = (slice(ymin,ymax), slice(xmin,xmax))
        region_data = data[region_slice]
        data[region_slice] = value*grid_points_in_poly(region_data.shape,
                                                       contour_rel)

        if roll_x or roll_y:
            res = np.roll(np.roll(data, -roll_x, axis=1), -roll_y, axis=0)
        else:
            res = data
        return res

    labels = np.zeros(shape, dtype='i4')
    for n, con in enumerate(contours):
        labels = fill_in_contour(con, labels, n+1)

    return labels

def contour_ji_to_geo(contour_ji, lon, lat):
    """ converts a contour in ij pixel coordinates to lat/lon

    Parameters
    ---------------------
    contour_ij: the list of vertices in (the LAVD image's) ij pixel coordinates
    lon: 1-D array of grid longitudes
    lat: 1-D array of gird latitudes

    Returns
    --------------------
    contour_geo: the list of vertices in lat/lon coordinates
    """

    dlon = abs(abs(lon[1]) - abs(lon[0]))
    dlat = abs(abs(lat[1]) - abs(lat[0]))

    j,i  = contour_ji.T

    x = lon[0] + dlon*i
    y = lat[0] + dlat*j

    contour_geo = np.array([x, y]).transpose()
    return contour_geo
