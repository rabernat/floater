from __future__ import print_function

import unittest
import numpy as np
from floater import hexgrid

def _get_test_array(shape=(3,3), nonzero=[(1,1)], dtype=np.float64):
    a = np.zeros(shape, dtype)
    for (j,i) in nonzero:
        a[j,i] = 1
    return a

class HexArrayTester(unittest.TestCase):

    def setUp(self):
        pass

    def test_neighbors(self):
        for ha in [hexgrid.HexArray(shape=(3,3)),
                   hexgrid.HexArray(np.empty((3,3), dtype=np.float64))]:
            for n in range(9):
                if n == 4:
                    self.assertSequenceEqual(
                        list(ha.neighbors(n)),
                        [0, 1, 5, 7, 6, 3]
                     )
                else:
                    self.assertEqual(len(ha.neighbors(n)), 0)

    def test_critical_points(self):
        a = _get_test_array()
        ha = hexgrid.HexArray(a)
        c = ha.classify_critical_points()

        self.assertSequenceEqual(a.shape, c.shape)
        # c should actually come out the same as a
        np.testing.assert_array_equal(a.astype('i4'), c)

    def test_maxima(self):
        a = _get_test_array()
        ha = hexgrid.HexArray(a)
        maxima = ha.maxima()
        self.assertEqual(len(maxima), 1)
        self.assertEqual(maxima[0], 4)

    def test_pos(self):
        a = _get_test_array()
        ha = hexgrid.HexArray(a)
        self.assertSequenceEqual(ha.pos(0), (0.25, 0.0))
        self.assertSequenceEqual(ha.pos(1), (1.25, 0.0))
        self.assertSequenceEqual(ha.pos(3), (-0.25, 1.0))
        self.assertSequenceEqual(ha.pos(4), (0.75, 1.0))

class HexArrayRegionTester(unittest.TestCase):

    def setUp(self):
        self.ha = hexgrid.HexArray(_get_test_array())

    def test_add_remove_point(self):
        hr = hexgrid.HexArrayRegion(self.ha)
        self.assertNotIn(4, hr)
        hr.add_point(4)
        self.assertIn(4, hr)
        hr.add_point(5)
        self.assertIn(5, hr)
        hr.remove_point(5)
        self.assertNotIn(5, hr)

    def test_first_point(self):
        hr = hexgrid.HexArrayRegion(self.ha, 1)
        self.assertIn(1, hr)
        self.assertEqual(hr.first_point, 1)

    def test_exterior_boundary(self):
        # points on the boundary should give no boundary
        hr = hexgrid.HexArrayRegion(self.ha)
        hr.add_point(0)
        self.assertSetEqual(hr.exterior_boundary(), set())
        # points in the middle should
        hr = hexgrid.HexArrayRegion(self.ha)
        hr.add_point(4)
        self.assertSetEqual(hr.exterior_boundary(), {0,1,5,7,6,3})

    def test_interior_exterior_boundary(self):
        a = np.array([[0,0,0,0,0],
                      [0,0,2,2,0],
                      [0,2,1,2,0],
                      [0,0,2,2,0],
                      [0,0,0,0,0]])
        ha = hexgrid.HexArray(shape=a.shape)
        hr = hexgrid.HexArrayRegion(ha)

        a = a.ravel()
        cpt = np.nonzero(a==1)[0][0]
        hr.add_point(cpt)
        bd = set(np.nonzero(a==2)[0])
        eb = hr.exterior_boundary()
        self.assertSetEqual(eb, bd)
        for pt in bd:
            hr.add_point(pt)
        ib = hr.interior_boundary()
        print(hr.members)
        self.assertSetEqual(ib, bd)

    def test_convex(self):
        ha = hexgrid.HexArray(shape=(10,10))
        pos = np.array([ha.pos(n) for n in range(ha.N)])
        x, y = pos.T
        r = np.sqrt((x - x.mean())**2 + (y-y.mean())**2)
        mask = r<=3
        hr = hexgrid.HexArrayRegion(ha)
        for n in np.nonzero(mask.ravel())[0]:
            hr.add_point(n)
        self.assertTrue(hr.is_convex())
        mask += (abs(x)<=1)
        for n in np.nonzero(mask.ravel())[0]:
            hr.add_point(n)
        self.assertFalse(hr.is_convex())

class HexgridStandaloneFunctionTester(unittest.TestCase):
    def test_points_in_poly(self):
        verts = np.array([[1,-1],[1,1],[-1,1],[-1,-1]], dtype='f8')
        self.assertTrue(hexgrid.point_in_poly(verts, 0., 0.))
        self.assertTrue(hexgrid.point_in_poly(verts, 0.99, 0.99))
        self.assertTrue(hexgrid.point_in_poly(verts, -0.99, -0.99))
        self.assertFalse(hexgrid.point_in_poly(verts, 1.1, 1.1))
        self.assertFalse(hexgrid.point_in_poly(verts, -1.1, -1.1))

    def test_find_convex_regions(self):
        # set up kelvin's cat's eyes flow on hexgrid
        nx = 100
        ha = hexgrid.HexArray(shape=(nx,nx))
        pos = np.array([ha.pos(n) for n in range(ha.N)])
        x, y = pos.T
        x = (x-5)*2*np.pi/80
        y = (y - nx/2)*2*np.pi/100
        # Kelvin's cats eyes flow
        a = 0.8
        psi = np.log(np.cosh(y) + a*np.cos(x)) - np.log(1 + a)
        cr = hexgrid.find_convex_regions(-psi.reshape(nx,nx))
        self.assertEqual(len(cr), 1)
        hr = cr[0]
        self.assertTrue(np.all(psi[list(hr.members)]<0))
        # how do we know that the region was correctly identified?
        psimask = (psi<=0) & (x>0.05) & (x<=(2*np.pi + 0.05))
        psiset = set(np.nonzero(psimask)[0])
        # now remove a point we know is problematic
        psiset.remove(5085)
        self.assertSetEqual(psiset, hr.members)
        # test minsize kwarg
        cr = hexgrid.find_convex_regions(-psi.reshape(nx,nx), minsize=1e6)
        self.assertEqual(len(cr), 0)

boundary_examples = """
Exterior Boundary
  o   o   .
o   x   o
  o   o   .

Interior Boundary
  .   .   .   .   .
.   .   o   o   .
  .   o   x   o   .
.   .   o   o   .
  .   .   .   .   .
"""
