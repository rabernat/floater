#cython: profile=True
# #cython: boundscheck=False
# #cython: wraparound=False
# #cython: nonecheck=False
#from __future__ import division
import numpy as np
import cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.unordered_set cimport unordered_set
#from libcpp.set cimport set
from libcpp.vector cimport vector
from cython.parallel cimport prange, threadid
from scipy.spatial import qhull
#import matplotlib.path as mplPath

from cython.operator cimport dereference as deref, preincrement as inc


# integer indices
DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t
# array values
DTYPE_flt = np.float64
ctypedef np.float64_t DTYPE_flt_t

# what type should things be?
# http://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view
# answer: malloc

# special value for border points
cdef DTYPE_int_t INT_NOT_FOUND = -9999

cdef class HexArray:
    # array shape
    cdef readonly DTYPE_int_t Nx, Ny, N
    cdef readonly bint index_right, has_data
    # raveled array data
    cdef DTYPE_flt_t [:] ar

    def __init__(self, np.ndarray[DTYPE_flt_t, ndim=2] data=None,
                        shape=None, idx_right=1):
        """Initialize new hex array of a given shape."""

        if (data is None and shape is None) or (data is None and shape is None):
            raise ValueError('Either data or shape must be specified')
        elif data is not None:
            if data.ndim != 2:
                raise ValueError('Only 2D data is allowed')
            self.Nx = data.shape[1]
            self.Ny = data.shape[0]
            self.ar = data.ravel()
            self.has_data = True
        else:
            if len(shape) != 2:
                raise ValueError('Shape muse be 2D')
            self.Nx = shape[1]
            self.Ny = shape[0]
            self.has_data = False
        self.N = self.Nx*self.Ny
        self.index_right = idx_right

    # this function sucks because it uses a python type (tuple)
    # and thus can't release the gil
    # using a c data structure (e.g. struct) would overcome gil limitation
    cpdef tuple ji_from_n(self, DTYPE_int_t n):
        cdef DTYPE_int_t j, i
        cdef tuple coord
        j = n / self.Nx
        i = n % self.Nx
        coord = (j, i)
        return coord

    cdef DTYPE_int_t n_from_ji(self, DTYPE_int_t j, DTYPE_int_t i) nogil:
        cdef DTYPE_int_t n
        n = i + j*self.Nx
        return n

    def N_from_ji(self, j, i):
        return self.n_from_ji(j, i)

    cdef bint is_border_n(self, DTYPE_int_t n) nogil:
        cdef DTYPE_int_t j, i
        j = n / self.Nx
        i = n % self.Nx
        return self.is_border_ji(j, i)

    cdef bint is_border_ji(self, DTYPE_int_t j, DTYPE_int_t i) nogil:
        return (i==0) or (i==self.Nx-1) or (j==0) or (j==self.Ny-1)

    cdef DTYPE_flt_t _xpos(self, int n) nogil:
        return <DTYPE_flt_t> (n % self.Nx) + 0.25 - 0.5*((n / self.Nx)%2)

    cdef DTYPE_flt_t _ypos(self, int n) nogil:
        return <DTYPE_flt_t> (n / self.Nx)

    def pos(self, int n):
        return (self._xpos(n), self._ypos(n))

    cdef int* _neighbors(self, DTYPE_int_t n) nogil:
        """Given index n, return neighbor indices.

        PARAMETERS
        ----------
        n : int
            1D index of point

        RETURNS
        -------
        nbr : int*
            pointer to array of 6 neighbor indices. Must be freed manually?
        """

        cdef DTYPE_int_t j, i
        cdef bint evenrow
        cdef int* nbr
        nbr = <int*> malloc(sizeof(int) * 6)
        nbr[0] = INT_NOT_FOUND

        j = n / self.Nx
        i = n % self.Nx
        evenrow = j % 2

        # don't even bother with border points
        if self.is_border_ji(j, i):
            return nbr

        if not evenrow:
            nbr[0] = self.n_from_ji(j-1, i)
            nbr[1] = self.n_from_ji(j-1, i+1)
            nbr[2] = self.n_from_ji(j, i+1)
            nbr[3] = self.n_from_ji(j+1, i+1)
            nbr[4] = self.n_from_ji(j+1, i)
            nbr[5] = self.n_from_ji(j, i-1)
        else:
            nbr[0] = self.n_from_ji(j-1, i-1)
            nbr[1] = self.n_from_ji(j-1, i)
            nbr[2] = self.n_from_ji(j, i+1)
            nbr[3] = self.n_from_ji(j+1, i)
            nbr[4] = self.n_from_ji(j+1, i-1)
            nbr[5] = self.n_from_ji(j, i-1)
        return nbr

    cdef DTYPE_int_t _classify_point(self, int n) nogil:
        cdef int k, kprev
        # neighbor pointer
        cdef int* nbr
        # raw difference
        cdef DTYPE_flt_t diff, diff_p
        # sign of differences
        cdef DTYPE_int_t sign_diff, sign_diff_p
        cdef DTYPE_int_t sum_sign_diff = 0

        if self.is_border_n(n):
            return 0
        else:
            nbr = self._neighbors(n)
            # fill in the differnces
            for k in range(6):
                kprev = (k-1) % 6
                diff = self.ar[n] - self.ar[nbr[k]]
                diff_p = self.ar[n] - self.ar[nbr[kprev]]
                # check for zero gradient
                if (diff==0.0) or (diff_p==0.0):
                    sum_sign_diff = -2
                    break
                if diff>0.0:
                    sign_diff = 1
                else:
                    sign_diff = -1
                if diff_p>0.0:
                    sign_diff_p = 1
                else:
                    sign_diff_p = -1
                sum_sign_diff += <DTYPE_int_t> (sign_diff != sign_diff_p)
            free(nbr)

            # regular point
            if sum_sign_diff==2:
                return 0
            # extremum
            elif sum_sign_diff==0:
                return sign_diff
            # saddle
            elif sum_sign_diff==4:
                return 2
            # zero gradient point
            elif sum_sign_diff==-2:
                return -2
            # something weird happened
            else:
                return -3


    def neighbors(self, n):
        """Given index n, return neighbor indices.

        PARAMETERS
        ----------
        n : int
            1D index of point

        RETURNS
        -------
        nbr : arraylike
            Numpy ndarray of neighbor points
        """
        cdef int * nptr = self._neighbors(n)
        cdef int [:] nbr
        if nptr[0] == INT_NOT_FOUND:
            free(nptr)
            return np.array([], DTYPE_int)
        else:
            nbr = <int[:6]> nptr
            numpy_array = np.asarray(nbr.copy())
            free(nptr)
            return numpy_array

    def classify_critical_points(self):
        """Identify and classify the critical points of data array.
        0: regular point
        +1: maximum
        -1: minimum
        +2: saddle point
        -2: zero gradient detected
        -3: monkey point

        RETURNS
        -------
        c : arraylike
            array with the same shape as a, with critical points marked
        """

        if not self.has_data:
            raise ValueError('HexArray was not initialized with data.')

        cdef DTYPE_int_t [:] c
        c = np.zeros(self.N, DTYPE_int)

        # loop index
        cdef int n

        for n in range(self.N):
            c[n] = self._classify_point(n)
        res = np.asarray(c).reshape(self.Ny, self.Nx)
        return res

    cpdef np.ndarray[int, ndim=1] maxima(self):
        cpoints = self.classify_critical_points()
        return np.nonzero(cpoints.ravel()==1)[0].astype(DTYPE_int)

cdef class HexArrayRegion:

    # the parent region
    cdef HexArray ha
    # the points in the region
    cdef unordered_set[int] members
    cdef vector[int] members_ordered
    cdef int first_point

    def __cinit__(self, HexArray ha, int first_pt = INT_NOT_FOUND):
        self.ha = ha
        if first_pt != INT_NOT_FOUND:
            self._add_point(first_pt)
        self.first_point = first_pt

    property members:
        def __get__(self):
            return self.members

    property first_point:
        def __get__(self):
            if self.first_point == INT_NOT_FOUND:
                return None
            else:
                return self.first_point

    def __contains__(self, int pt):
        return self.members.count(pt) > 0

    def add_point(self, int pt):
        self._add_point(pt)

    cdef void _add_point(self, int pt) nogil:
        self.members.insert(pt)

    def remove_point(self, int pt):
        self._remove_point(pt)

    cdef void _remove_point(self, int pt) nogil:
        self.members.erase(pt)

    def exterior_boundary(self):
        return self._exterior_boundary()

    cdef unordered_set[int] _exterior_boundary(self) nogil:
        cdef unordered_set[int] boundary
        cdef int n, k
        cdef int* nbr
        cdef int npt
        cdef size_t cnt
        for n in self.members:
            nbr = self.ha._neighbors(n)
            if not nbr[0] == INT_NOT_FOUND:
                for k in range(6):
                    cnt = self.members.count(nbr[k])
                    if cnt==0:
                        boundary.insert(nbr[k])
            free(nbr)
        return boundary

    def interior_boundary(self):
        return self._interior_boundary()

    cdef unordered_set[int] _interior_boundary(self) nogil:
        cdef unordered_set[int] boundary
        cdef int n
        for n in self.members:
            if self._is_boundary(n):
                boundary.insert(n)
        return boundary

    cdef bint _is_boundary(self, int n) nogil:
        cdef int* nbr
        cdef size_t cnt, k
        cdef bint result = 0
        nbr = self.ha._neighbors(n)
        cnt = 0
        if not nbr[0] == INT_NOT_FOUND:
            for k in range(6):
                cnt += self.members.count(nbr[k])
            if (cnt<6) and (cnt>0):
                result = 1
        free(nbr)
        return result

    def interior_boundary_ordered(self):
        return self._interior_boundary_ordered()

    cdef vector[int] _interior_boundary_ordered(self) nogil:
        cdef vector[int] boundary
        cdef vector[int].reverse_iterator it
        cdef int* nbr
        cdef int n, initpt, startpt, prevpt, testpt, nextpt, nextpt_line
        cdef int orth_vert_pt0, orth_vert_pt1, nbr_count
        cdef bint looking = 1
        cdef bint already_in_boundary
        cdef size_t cnt = 0
        cdef size_t cntmax = 100
        cdef size_t cnt_back
        # first just find any point on the boundary
        for n in self.members:
            if self._is_boundary(n):
                boundary.push_back(n)
                initpt = n
                break
        # now iterate through neighbors, looking for other boundary points

        # index of the starting point...when we get back here we are done
        startpt = initpt
        # index of the previous point...needed for walking line-like segments
        prevpt = initpt
        cnt = 0
        while True:
            # get the neighbors of the most recently added point
            looking = 0
            nbr = self.ha._neighbors(startpt)
            if nbr[0] == INT_NOT_FOUND:
                # we are on a boundary, stop looking
                break
            # loop through each neighbor and check its proprties
            nextpt = -1
            nextpt_line = -1
            # counter for how many neighbors are in the region
            nbr_count = 0
            for n in range(6):
                # only consider points that are still in the region
                testpt = nbr[n]
                if self.members.count(testpt)==1:
                    nbr_count += 1
                    # if we are on a line, the orthogonal vertex approach will
                    # fail, and we still need to figure out which direction to
                    # go. On a line, there will be two neighbors in the
                    # region: the previous point and the next point. We don't
                    # want to add the previous point UNLESS we are at the end
                    # of a line segment. In that case, we want to loop back and
                    # traverse the line back to the initial point (initpt).
                    if testpt != prevpt:
                        nextpt_line = testpt

                    # the vertex under consideration is (startpt, testpt)
                    # get the orthogonal vertex to this vertex
                    # (orth_vert_pt0, orth_vert_pt1)
                    if n==0:
                        orth_vert_pt0 = nbr[5]
                    else:
                        orth_vert_pt0 = nbr[n-1]
                    if n==5:
                        orth_vert_pt1 = nbr[0]
                    else:
                        orth_vert_pt1 = nbr[n+1]
                    # for positive orientation of the boundary, we want
                    # orth_vert_pt1 OUTSIDE the region and orth_vert_pt2 INSIDE,
                    # a vector pointing normal to the boundary following the
                    # right-hand rule
                    if ((self.members.count(orth_vert_pt0)==0) and
                        (self.members.count(orth_vert_pt1)==1)):
                        # we found the next point
                        nextpt = testpt
                        break

            free(nbr)


            # if we got here without finding a nextpt, it probably means we are
            # on a line segnment...need special logic for that
            if nextpt == -1:
                if nbr_count == 1:
                    # there is only one region neighbor, which means we are at
                    # the end of a line segment. we turn around
                    nextpt = prevpt
                elif (nbr_count == 2) and (nextpt_line != -1):
                    # we are traversing a line segment
                    nextpt = nextpt_line
                else:
                    # we should never get here
                    # something went wrong
                    with gil:
                        print("FAILED TO FIND NEXT POINT "
                              "startpt=%g, nbr_count=%g, nextpt=%g, nextpt_line=%g"
                              % (startpt, nbr_count, nextpt, nextpt_line))
                    return boundary

            #with gil:
            #    print('from %g adding %g' % (startpt, nextpt))

            #cnt += 1
            #if cnt>cntmax:
            #    break

            # add the point to the boundary vector, only if it is NOT the
            # the original point
            if nextpt != initpt:
                prevpt = startpt
                startpt = nextpt
                boundary.push_back(nextpt)
            # otherwise we are done with the iteration
            else:
                break
        return boundary

    def area(self):
        return self._area()

    # http://www.mathopenref.com/coordpolygonarea2.html
    cdef DTYPE_flt_t _area(self) nogil:
        cdef vector[int] ibo = self._interior_boundary_ordered()
        with gil:
            print "got interior boundary ordered"
        cdef DTYPE_flt_t x0, y0, x1, y1
        cdef size_t nverts = ibo.size()
        # vertex loop counter
        cdef size_t i = 0
        # other vertex loop counter
        # The last vertex is the 'previous' one to the first
        cdef size_t j = nverts - 1
        # accumulates area in the loop
        cdef DTYPE_flt_t area = 0.0;
        for i in range(nverts):
            x0 = self.ha._xpos(i)
            y0 = self.ha._ypos(i)
            x1 = self.ha._xpos(j)
            y1 = self.ha._ypos(j)
            area += (x1 + x0) * (y1 - y0)
            j = i

        # minus needed because the interior boundary points are counterclockwise,
        # not clockwise
        return -area/2.0

    def convex_hull_area(self):
        return self._convex_hull_area()

    cdef DTYPE_flt_t _convex_hull_area(self) nogil:
        # interior boundary
        cdef unordered_set[int] ib = self._interior_boundary()
        cdef DTYPE_flt_t [:,:] ib_points
        cdef size_t nib = ib.size()

        # vertices of hull
        cdef DTYPE_flt_t [:,:] hull_vertices
        cdef DTYPE_flt_t hull_area
        # worth making a view? how to release gil?

        cdef size_t npt, nhull, Nverts
        cdef size_t n = 0

        # straight python from here on
        # how to speed this up?
        with gil:
            ib_points = np.empty((nib, 2), dtype=DTYPE_flt)
        for npt in ib:
            ib_points[n,0] = self.ha._xpos(npt)
            ib_points[n,1] = self.ha._ypos(npt)
            n += 1

        # can't do try without gil
        #try:
        #    hull_vertices = _get_qhull_verts(ib_points)
        #except:
        #    return False
        with gil:
            hull_area = _get_qhull_area(ib_points)

        return hull_area

    def is_convex(self):
        return self._is_convex()

    @cython.linetrace(True)
    cdef bint _is_convex(self) nogil:
        # interior boundary
        cdef unordered_set[int] ib = self._interior_boundary()
        cdef unordered_set[int] eb = self._exterior_boundary()
        cdef DTYPE_flt_t [:,:] ib_points
        cdef size_t nib = ib.size()
        # the coordinates of the test point
        cdef DTYPE_flt_t xpt, ypt
        # the coordinates of the boundary points
        # need to pass a numpy array to qhull anyway
        #cdef np.ndarray[DTYPE_flt_t, ndim=2] ib_points

        # vertices of hull
        cdef DTYPE_flt_t [:,:] hull_vertices
        # worth making a view? how to release gil?

        cdef size_t npt, nhull, Nverts
        cdef int n = 0


        # straight python from here on
        # how to speed this up?
        with gil:
            ib_points = np.empty((nib, 2), dtype=DTYPE_flt)
        for npt in ib:
            ib_points[n,0] = self.ha._xpos(npt)
            ib_points[n,1] = self.ha._ypos(npt)
            n += 1

        # can't do try without gil
        #try:
        #    hull_vertices = _get_qhull_verts(ib_points)
        #except:
        #    return False
        with gil:
            hull_vertices = _get_qhull_verts(ib_points)
        if hull_vertices.shape[0]==0:
            return False

        # check to see if any of the exterior boundary points lie
        # inside the convex hull
        #with nogil:
        for npt in eb:
            xpt = self.ha._xpos(npt)
            ypt = self.ha._ypos(npt)
            if _point_in_poly(hull_vertices, xpt, ypt):
            #if _mpl_point_in_poly(hull_vertices, xpt, ypt):
                return False
        return True

    def still_convex(self, int pt):
        return self._still_convex(pt)

    cdef bint _still_convex(self, int pt):
        cdef bint sc
        self._add_point(pt)
        sc = self._is_convex()
        self._remove_point(pt)
        return sc


def find_convex_regions(np.ndarray[DTYPE_flt_t, ndim=2] a, int minsize=0,
                        return_labeled_array=False,
                        target_convexity_deficiency=1e-3
                        ):
    """Find convex regions around the extrema of ``a``.

    PARAMETERS
    ----------
    a : arraylike
        The 2D field in which to search for convex regions. Must be dtype=f64.
    minsize : int
        The minimum size of regions to return (number of points)

    Haller: "Convexity Deficiency" is ratio of the area between the curve and
    its convex hull to the area enclosed by the curve

    RETURNS
    -------
    regions : list of HexArrayRegion elements
    """
    cdef HexArray ha = HexArray(a)
    cdef DTYPE_int_t [:] maxima = ha.maxima()
    cdef DTYPE_int_t nmax
    # these are local to the loop
    cdef HexArrayRegion hr
    cdef unordered_set[int] bndry
    cdef size_t cnt, pt, next_pt
    cdef DTYPE_flt_t diff, diff_min, convex_def, hull_area, region_area
    cdef bint first_pt



    regions = []
    for nmax in maxima:
        hr = HexArrayRegion(ha, nmax)
        cnt = 0
        diff_min = 0.0
        # set initial convexity deficiency to zero
        convex_def = 0.0
        while convex_def <= target_convexity_deficiency:
            bndry = hr._exterior_boundary()
            first_pt = True
            # examine the boundary neighbors, looking for the next point to add
            for pt in bndry:
                diff = ha.ar[nmax] - ha.ar[pt]
                if first_pt:
                    diff_min = diff
                    first_pt = False
                if diff <= diff_min:
                    next_pt = pt
                    diff_min = diff
            hr._add_point(next_pt)
            cnt += 1

            # calculate convexity
            if cnt > 3:
                print('region npoints %g' % len(hr.members))
                region_area = hr._area()
                print('region_area %e' % region_area)
                # only bother moving on if area is nonzero
                if abs(region_area) > 1e-12:
                    print('calculating convex hull')
                    hull_area = hr._convex_hull_area()
                    print('hull_area: %f' % hull_area)
                    convex_def = (hull_area - region_area)/region_area
                    print('convex_def: %f' % convex_def)

        # if we got here, we exceeded the convexity deficiency, so we need to
        # remove the last point
        hr._remove_point(next_pt)

        if hr.members.size() > minsize:
            regions.append(hr)

    if return_labeled_array:
        return label_regions(regions, ha)
    else:
        return regions

def label_regions(regions, ha):
    r = np.full(ha.N, -1)
    for reg in regions:
        r[list(reg.members)] = reg.first_point
    r.shape = ha.Ny, ha.Nx
    return r

cdef bint _test_convex(HexArrayRegion hr, int pt):
    cdef unordered_set[int] ib = hr.interior_boundary()
    return 1

def point_in_poly(np.ndarray[DTYPE_flt_t, ndim=2] npverts,
                    DTYPE_flt_t testx, DTYPE_flt_t testy):
    cdef DTYPE_flt_t [:,:] verts
    verts = npverts
    return _point_in_poly(verts, testx, testy)

def mpl_point_in_poly(np.ndarray[DTYPE_flt_t, ndim=2] npverts,
                    DTYPE_flt_t testx, DTYPE_flt_t testy):
    cdef DTYPE_flt_t [:,:] verts
    verts = npverts
    return _mpl_point_in_poly(verts, testx, testy)

# I don't fully understand why this works, but it does
# http://stackoverflow.com/a/2922778/3266235
cdef bint _point_in_poly(DTYPE_flt_t [:,:] verts,
                         DTYPE_flt_t testx, DTYPE_flt_t testy) nogil:
    cdef size_t nvert = verts.shape[0]
    cdef size_t i = 0
    cdef size_t j = nvert -1
    cdef bint c = 0
    while (i < nvert):
        # apparently all I had to do was change the > to >= to make this
        # agree with matplotlib
        # if ( ((verts[i,1]>testy) != (verts[j,1]>testy)) and
        #      (testx < (verts[j,0]-verts[i,0]) * (testy-verts[i,1])
        #               / (verts[j,1]-verts[i,1]) + verts[i,0]) ):
        if ( ((verts[i,1]>=testy) != (verts[j,1]>=testy)) and
             (testx <= (verts[j,0]-verts[i,0]) * (testy-verts[i,1])
                      / (verts[j,1]-verts[i,1]) + verts[i,0]) ):

            c = not c
        j = i
        i += 1
    return c


# don't want to have to rely on matplotlib
#@cython.wraparound(True)
@cython.linetrace(True)
cdef bint _mpl_point_in_poly(DTYPE_flt_t [:,:] verts,
                         DTYPE_flt_t testx, DTYPE_flt_t testy):
    return 0
    # make sure polygon is closed
    # do outside function
    ##vertices = np.vstack([vertices, vertices[0]])
    # cdef np.ndarray[DTYPE_int_t, ndim=1] codes
    # cdef size_t Nverts = len(verts)
    # codes = np.full(Nverts, mplPath.Path.LINETO, dtype=DTYPE_int)
    # codes[0] = mplPath.Path.MOVETO
    # codes[Nverts-1] = mplPath.Path.CLOSEPOLY
    # bbPath = mplPath.Path(verts, codes)
    # return bbPath.contains_point((testx, testy), radius=0.0)

def get_qhull_verts(np.ndarray[DTYPE_flt_t, ndim=2] points):
    return np.asarray(_get_qhull_verts(points))

cdef DTYPE_flt_t [:,:] _get_qhull_verts(DTYPE_flt_t [:,:] points) nogil:
    cdef DTYPE_flt_t [:,:] hull_vertices, vert_pts
    cdef DTYPE_int_t [:] vert_idx
    cdef size_t Nverts, n
    with gil:
        try:
            hull = qhull.ConvexHull(points)
            Nverts = len(hull.vertices)
            hull_vertices = np.empty((Nverts+1,2), hull.points.dtype)
            vert_idx = hull.vertices
            vert_pts = hull.points
            hull_vertices[Nverts,0] = vert_pts[vert_idx[0],0]
            hull_vertices[Nverts,1] = vert_pts[vert_idx[0],1]
        except:
            return np.empty((0,2), DTYPE_flt)
        #Nverts = 0

    #hull_vertices = hull.points[hull.vertices]
    for n in range(Nverts):
        hull_vertices[n,0] = vert_pts[vert_idx[n],0]
        hull_vertices[n,1] = vert_pts[vert_idx[n],1]
    return hull_vertices

def get_qhull_area(np.ndarray[DTYPE_flt_t, ndim=2] points):
    return _get_qhull_area(points)

cdef DTYPE_flt_t _get_qhull_area(DTYPE_flt_t [:,:] points):
    try:
        hull = qhull.ConvexHull(points)
        return hull.volume
    except:
        return 0.0


def polygon_area(np.ndarray[DTYPE_flt_t, ndim=2] points):
    return _polygon_area(points)

# http://www.mathopenref.com/coordpolygonarea2.html
cdef DTYPE_flt_t _polygon_area(DTYPE_flt_t [:,:] verts) nogil:
    cdef size_t nverts = verts.shape[0]
    # vertex loop counter
    cdef size_t i = 0
    # other vertex loop counter
    # The last vertex is the 'previous' one to the first
    cdef size_t j = nverts - 1

    # accumulates area in the loop
    cdef DTYPE_flt_t area = 0.0;
    for i in range(nverts):
        area += (verts[j,0]+verts[i,0]) * (verts[j,1]-verts[i,1])
        j = i

    # minus needed because the interior boundary points are counterclockwise,
    # not clockwise
    return -area/2.0




# cdef int pnpoly(int nvert, float vertx*, float verty*,
#                 float testx*, float testy*):
#     int i, j, c = 0
#     for (i = 0, j = nvert-1; i < nvert; j = i++) {
#       if ( ((verty[i]>testy) != (verty[j]>testy)) &&
#        (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
#          c = !c;
#     }
#     return c;
#   }
