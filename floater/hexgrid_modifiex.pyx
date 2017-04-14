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

#from cython.operator cimport dereference as deref, preincrement as inc


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
# maximum iterations in search algorithms
cdef size_t MAX_ITERS = 1000000
# flag for failed convexity deficiency test
cdef DTYPE_flt_t CONVEX_DEF_UNDEFINED = -9999999999.

@cython.final
cdef class HexArray:
    # array shape
    cdef readonly DTYPE_int_t Nx, Ny, N
    cdef readonly bint index_right, has_data
    # raveled array data
    cdef public DTYPE_flt_t [:] ar

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

@cython.final
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

    cpdef void _add_point(self, int pt) nogil:
        self.members.insert(pt)

    def remove_point(self, int pt):
        self._remove_point(pt)

    cdef void _remove_point(self, int pt) nogil:
        self.members.erase(pt)

    def exterior_boundary(self):
        return self._exterior_boundary()

    cpdef unordered_set[int] _exterior_boundary(self) nogil:
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

    cpdef int _vertex_orientation(self, int startpt, size_t n) nogil:
        """Check the boundary orientation of a vertex originating at startpt
        in the n-th direction."""
        cdef int* nbr
        cdef int orth_vert_left, orth_vert_right
        cdef int is_inside_left, is_inside_right

        nbr = self.ha._neighbors(startpt)
        if nbr[0] == INT_NOT_FOUND:
            free(nbr)
            # don't even try to orient if we are on a boundary
            return INT_NOT_FOUND
        #orth_vert_left = nbr[(n+1) % 6]
        #orth_vert_right = nbr[(n-1) % 6]
        if n==5:
            orth_vert_left = nbr[0]
        else:
            orth_vert_left = nbr[n+1]
        if n==0:
            orth_vert_right = nbr[5]
        else:
            orth_vert_right = nbr[n-1]
        free(nbr)
        # for positive orientation of the boundary, we want
        # orth_vert_right OUTSIDE the region and orth_vert_left INSIDE,
        # a vector pointing normal to the boundary following the
        # right-hand rule
        is_inside_left = self.members.count(orth_vert_left)==1
        is_inside_right = self.members.count(orth_vert_right)==1
        #with gil:
        #    print 'vertex_orientation:', n, is_inside_left, is_inside_right
        return (is_inside_left - is_inside_right)


    def interior_boundary_ordered(self):
        return self._interior_boundary_ordered()

    cpdef vector[int] _interior_boundary_ordered(self) nogil:
        """Find the ordered interior boundary of a simply-connected
        hexarray region.

        Returns an *empty vector* if any line segments are found in the region.
        Some reasons why:
            - Line segments are not locally orientable (although they may be
              globally orientable)
            - Line segments attached to regions cause the iteration to go
              into an infinite look because it never gets back onto the line.
              (This could probably be fixed with some more complex logic), but...
            - We will never want to keep such regions
        """
        cdef vector[int] boundary
        cdef vector[int] failure
        cdef int* nbr
        cdef int n, initpt, startpt, prevpt, testpt, nextpt, nextpt_line, n_start
        cdef size_t nbr_count, line_neighbor_count, origin_visit_count
        cdef int orth_vert_left, orth_vert_right, nvertex
        cdef int inc, n_idx, is_inside_left, is_inside_right
        cdef bint already_in_boundary
        cdef size_t cnt = 0
        cdef size_t cntmax = 100
        cdef size_t cnt_back
        # +1: positive orientation (right hand rule: counterclockwise path around center)
        # -1: negative orientation (clockwise path around center)
        # 0: not determined yet
        cdef int orientation = 0
        cdef int test_orientation
        # first just find any point on the boundary
        for n in self.members:
            if self._is_boundary(n):
                boundary.push_back(n)
                initpt = n
                startpt = n
                break

        # now get another point to form a vertex
        nextpt = -1
        nvertex = -1
        line_neighbor_count = 0
        nbr_count = 0
        nbr = self.ha._neighbors(startpt)
        if nbr[0] != INT_NOT_FOUND:
            for n in range(6):
                testpt = nbr[n]
                # only consider points that are still in the region...
                if self.members.count(testpt)==1:
                    # ...and on the boundary
                    nbr_count += 1
                    if self._is_boundary(testpt):
                        nextpt = testpt
                        # we can stop searching if we got an orientation
                        orientation = self._vertex_orientation(startpt, n)
                        if orientation != 0:
                            nvertex = n
                        else:
                            # a neigbhor point with no orientation is a line
                            line_neighbor_count += 1

        # override nextpt if we got an orientation
        if nvertex != -1:
            nextpt = nbr[nvertex]
            n = nvertex
        free(nbr)

        if nextpt == -1:
            return failure

        #with gil:
        #    print "line_neigbor_count: %g" % line_neighbor_count

        if (nbr_count > 1) and (line_neighbor_count > 0):
            # we need to revisit the origin twice!
            origin_visit_count = 2
        else:
            origin_visit_count = 1

        # index mapping from one point to the next

        #
        # FORWARD NEIGHBOR INDICES
        #
        #     4     3
        #
        #  5     p     2
        #
        #     0     1
        #
        #
        # BACKWARD NEIGHBOR INDICES
        #
        #     1     0
        #
        #  2     p     5
        #
        #     3     4
        #
        # Examples of weird connectivity
        #
        #     x     x             x     o             x     o
        #
        #  o     p     o      o      p     x      o      p      o
        #
        #     x     x             x     o             x    x
        #
        # What do these have in common? at least two gaps between
        #
        # here is one idea: if the starting point has any line-like neighbors,
        # we know we need to visit it *twice* before we are done


        cnt = 0
        #while nextpt != initpt:
        while origin_visit_count > 0:
            # shift everything forward
            boundary.push_back(nextpt)
            prevpt = startpt
            startpt = nextpt
            nextpt = -1

            # always check orientation
            # ************************
            # the previous vertex is now (prevtpt, startpt)
            # startpt is the nth neighbot of prevpt
            test_orientation = self._vertex_orientation(prevpt, n)
            # check for boundary error
            if test_orientation == INT_NOT_FOUND:
                return failure
            #with gil:
            #    print('Got test orientation %g' % test_orientation)


            if orientation == 0:
                if test_orientation != 0:
                    # we found the orientation
                    orientation = test_orientation
            else:
                # make sure this orientation is consistent
                # if we encounter a different test_orientation after orientation
                # is already set, it means this is a multiply connected or
                # otherwise messed up region
                if test_orientation != 0:
                    if orientation != test_orientation:
                        return failure


            # now we can walk to the next point

            nbr = self.ha._neighbors(startpt)
            if nbr[0] == INT_NOT_FOUND:
                # on a boundary
                # nextpt = initpt
                return failure

            # figure out where the previous link is coming from
            n_start = (n + 3) % 6

            # the search direction (angular)
            if orientation < 0:
                inc = -1
            else:
                # if orientation is 0, doesn't matter what direction we search
                inc = 1

            # check the neighbor points in the proper order
            for n_idx in range(1, 7):
                # index of the neighbor point to check
                n = (n_start + inc*n_idx) % 6
                testpt = nbr[n]
                if self.members.count(testpt)==1:
                    # found our next point
                    nextpt = testpt
                    break

            cnt += 1
            if cnt > MAX_ITERS:
                break

            free(nbr)

            # we are about to visit the origin:
            if nextpt == initpt:
                # decrement origin visit counter
                origin_visit_count -= 1

        # make sure we actually got all the points
        # for certain pathalogical regions, we won't actually detect all the
        # boundary points
        # potentiall expensive call to _interior_boundary
        # check that each point in the interior boundary is also in the
        # ordered interior boundary
        #if boundary.size() < self._interior_boundary().size():
        #    return boundary
        #else:
        #    return failure
        return boundary

    def area(self):
        cdef vector[int] ibo = self._interior_boundary_ordered()
        return self._area_from_interior_boundary(ibo)

    # http://www.mathopenref.com/coordpolygonarea2.html
    cpdef DTYPE_flt_t _area_from_interior_boundary(self, vector[int] ibo) nogil:
        #cdef vector[int] ibo = self._interior_boundary_ordered()
        cdef DTYPE_flt_t x0, y0, x1, y1
        cdef size_t nverts = ibo.size()
        # vertex loop counter
        cdef size_t i = 0
        # other vertex loop counter
        # The last vertex is the 'previous' one to the first
        cdef size_t j = nverts - 1
        # accumulates area in the loop
        cdef DTYPE_flt_t area = 0.0
        for i in range(nverts):
            x0 = self.ha._xpos(ibo[i])
            y0 = self.ha._ypos(ibo[i])
            x1 = self.ha._xpos(ibo[j])
            y1 = self.ha._ypos(ibo[j])
            area += (x1 + x0) * (y1 - y0)
            #with gil:
            #    print('x0, y0, x1, y1, area: %g, %g, %g, %g, %g' % (x0, y0, x1, y1, area))
            j = i

        # we don't know the area, so we need to normalize
        if area < 0:
            return -area/2.0
        else:
            return area/2.0

    def convex_hull_area(self):
        cdef vector[int] ibo = self._interior_boundary_ordered()
        return self._convex_hull_area_from_interior_boundary(ibo)

    cpdef DTYPE_flt_t _convex_hull_area_from_interior_boundary(self, vector[int] ibo) nogil:
        # interior boundary
        #cdef unordered_set[int] ib = self._interior_boundary()
        cdef DTYPE_flt_t [:,:] ib_points
        cdef size_t nib = ibo.size()

        # vertices of hull
        cdef DTYPE_flt_t [:,:] hull_vertices
        cdef DTYPE_flt_t hull_area
        # worth making a view? how to release gil?

        cdef size_t npt, nhull, Nverts
        cdef size_t n = 0

        with gil:
            ib_points = np.empty((nib, 2), dtype=DTYPE_flt)
        for npt in ibo:
            ib_points[n,0] = self.ha._xpos(npt)
            ib_points[n,1] = self.ha._ypos(npt)
            n += 1

        with gil:
            hull_area = _get_qhull_area(ib_points)

        return hull_area

    cpdef DTYPE_flt_t convexity_deficiency(self) nogil:
        cdef DTYPE_flt_t region_area, hull_area
        cdef vector[int] ibo = self._interior_boundary_ordered()
        region_area = self._area_from_interior_boundary(ibo)
        #with gil:
        #    print('region_area %e' % region_area)
        # only bother moving on if area is nonzero
        if region_area > 0.0:
            #print('calculating convex hull')
            hull_area = self._convex_hull_area_from_interior_boundary(ibo)
            #with gil:
            #    print('hull_area: %f' % hull_area)
            return (hull_area - region_area)/region_area
        else:
            return CONVEX_DEF_UNDEFINED

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


def find_convex_regions(np.ndarray[DTYPE_flt_t, ndim=2] a,
                        int minsize=0, int maxsize=0,
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
    cdef bint first_pt, success



    regions = []
    # need to refactor this to put into its own function
    for nmax in maxima:
        hr = HexArrayRegion(ha, nmax)
        diff_min = 0.0
        # set initial convexity deficiency to zero
        convex_def = 0.0
        region_area = 0.0
        cnt = 0
        success = 0
        while True:
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

            # calculate convexity deficiency
            if cnt >= minsize:
                convex_def = hr.convexity_deficiency()
                if convex_def == CONVEX_DEF_UNDEFINED:
                    # stop searching if we got a weird region
                    break
                elif convex_def > target_convexity_deficiency:
                    # stop searching if we exceed convexity deficiency target
                    break
                else:
                    success = 1

            #print('cnt=%g, convex_def: %f' % (cnt, convex_def))

            if (maxsize > 0) and (cnt>maxsize):
                #print('exceeded count')
                break

        # if we got here, we exceeded the convexity deficiency, so we need to
        # remove the last point
        hr._remove_point(next_pt)

        if success:
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

cpdef DTYPE_flt_t [:,:] _get_qhull_verts(DTYPE_flt_t [:,:] points) nogil:
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
