import numpy as np
import xarray as xr
from skimage.measure import find_contours, points_in_poly, grid_points_in_poly
from skimage.feature import peak_local_max
from skimage.morphology import convex_hull_image, watershed
from scipy.spatial import qhull


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
    return area_elements.sum()/2.0


def get_local_region(data, ji, border_j, border_i):
    j, i = ji
    nj, ni = data.shape
    jmin = j - border_j[0]
    jmax = j + border_j[1] + 1
    imin = i - border_i[0]
    imax = i + border_i[1] + 1

    if (jmin < 0) or (imin < 0) or (jmax >= nj) or (imax >= ni):
        raise ValueError("Limits " + repr(((jmin, jmax), (imin, imax))) +
                         " outside of array shape " + repr((nj,ni)))

    return (j - jmin, i - imin), data[j,i] - data[jmin:jmax, imin:imax]


def is_contour_closed(con):
    return np.all(con[0] == con[-1])


def point_in_contour(con, ji):
    j, i = ji
    return points_in_poly(np.array([i, j])[None], con[:,::-1])[0]


def find_contour_around_maximum(data, ji, level, border_j=(5,5), border_i=(5,5)):
    j,i = ji
    max_val = data[j,i]

    # increments for increasing bounds of region
    delta_b = 5

    target_con = None
    grow_down, grow_up, grow_left, grow_right = 4*(False,)

    while target_con is None:
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
        (j_rel, i_rel), region_data = get_local_region(data, (j,i), border_j, border_i)
        nj, ni = region_data.shape

        # extract the contours
        contours = find_contours(region_data, level)

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

    return target_con, region_data, border_j, border_i


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
    convexity_deficiency : fload
    """
    # reshape the data to x, y order
    con_points = con[:,::-1]

    # calculate area of polygon
    region_area = abs(polygon_area(con_points))

    # find convex hull
    hull = qhull.ConvexHull(con_points)
    #hull_points = np.array([con_points[pt] for pt in hull.vertices])
    hull_area = hull.volume

    cd = (hull_area - region_area ) / region_area

    return region_area, hull_area, cd


def convex_contour_around_maximum(data, ji, step, border=5,
                                  convex_def=0.01, verbose=False):
    """Find the largest convex contour around a maximum.

    Parameters
    ----------
    data : array_like
        The 2D data to contour
    ji : tuple
        The index of the maximum in (j, i) order
    step : float
        the value with which to increment the contour level
    border: int
        the initial window around the maximum
    convex_def : float, optional
        The maximum convexity deficiency allowed for the contour
        before the seach stops.
    verbose: bool, optional
        Whether to print out diagnostic information

    Returns
    -------
    contour : array_like
        2D array of contour vertices with shape (N,2) that follows
        the scikit image conventions (contour[:,0] are j indices)
    area : float
        The area enclosed by the contour
    """

    # the maximum
    j, i = ji

    # the initial search region
    border_j = (5,5)
    border_i = (5,5)

    # the test contours
    # the finer stepsize, the more careful the search
    # the local region is normalized such that the max is 0
    contour_levels = np.arange(step, data.max(), step)

    contour_prev = None
    region_area_prev = None

    for level in contour_levels:
        if verbose:
            print (repr(ji) + ': level: %g border: ' % level) + repr(border_j) + repr(border_i)

        try:
            # try to get a contour
            contour, region_data, border_j, border_i = find_contour_around_maximum(
                data, (j,i), level, border_j, border_i)
        except ValueError as ve:
            if verbose:
                print ve
            break

        # get the convexity deficiency
        region_area, hull_area, cd = contour_area(contour)
        if verbose:
            print('  region_area: % 6.1f, hull_area: % 6.1f, convex_def: % 6.5e '
                  % (region_area, hull_area, cd))

        if cd > convex_def:
            break
        else:
            # keep going
            # re-center the previous contour to be referenced to the
            # absolute position
            contour[:, 0] += (j-border_j[0])
            contour[:, 1] += (i-border_i[0])
            contour_prev, region_area_prev = contour, region_area

    return contour_prev, region_area_prev


def find_convex_contours(data, min_distance=5, min_area=100.,
                             step=1e-7, convex_def=0.001, verbose=False):
    """Find the outermost convex contours around the maxima of
    data with specified convexity deficiency.

    Parameters
    ----------
    data : array_like
        The 2D data to contour
    min_distance : int, optional
        The minimum distance around maxima
    min_area : float, optional
        The minimum area of the regions
    step : float, optional
        the step size with which to increment the contour level
    convex_def : float, optional
        The maximum convexity deficiency allowed for the contour
        before the seach stops.
    verbose: bool, optional
        Whether to print out diagnostic information

    Yields
    -------
    contour : array_like
        2D array of contour vertices with shape (N,2) that follows
        the scikit image conventions (contour[:,0] are j indices)
    area : float
        The area enclosed by the contour
    """

    plm = peak_local_max(data, min_distance=min_distance)

    for ji in plm:
        contour, area = convex_contour_around_maximum(data, ji, 1e-7, border=5,
                                                      convex_def=convex_def, verbose=verbose)
        if area >= min_area:
            yield ji, contour, area
