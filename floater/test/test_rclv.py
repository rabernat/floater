import numpy as np
import pytest
from floater import rclv

@pytest.fixture()
def sample_data_and_maximum():
    ny, nx = 100, 100
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x = (x-5)*2*np.pi/80
    y = (y - nx/2)*2*np.pi/100
    # Kelvin's cats eyes flow
    a = 0.8
    psi = np.log(np.cosh(y) + a*np.cos(x)) - np.log(1 + a)
    # want the extremum to be positive, need to reverse sign
    psi = -psi
    # max located at psi[50, 45] = 2.1972245773362196
    ji = (50,45)
    return psi, ji, psi[ji]


@pytest.fixture()
def sample_data_lon_lat():
    ny, nx = 100, 100
    dlon, dlat = 0.01, 0.01
    lon = np.arange(nx)*dlon + 100
    lat = np.arange(ny)*dlat - 45
    return lon, lat

@pytest.fixture()
def square_verts():
    return np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                     [-0.5, 0.5], [-0.5, -0.5]])


def circular_verts(r=1., x0=0., y0=0.):
    n = 100
    theta = np.linspace(0, 2*np.pi, n)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    return np.vstack([y, x]).T


def test_polygon_area(square_verts):
    assert rclv.polygon_area(square_verts) == 1.0
    # try without the last vertex
    assert rclv.polygon_area(square_verts[:-1]) == 1.0

    cverts = circular_verts()
    np.testing.assert_allclose(rclv.polygon_area(cverts), np.pi, rtol=0.01)


def test_projection():
    verts = circular_verts()
    lat0, lon0 = (0, 0)
    dlon, dlat = (0.1, 0.1)
    verts_proj_equator = rclv.project_vertices(verts, lon0, lat0, 0.1, 0.1)

    # area of a circle of radius 0.1 degree at the equator
    area_equator_expected = 388.4e6
    np.testing.assert_allclose(rclv.polygon_area(verts_proj_equator),
                               area_equator_expected, rtol=0.01)

    lat0 = 60
    verts_proj_60N = rclv.project_vertices(verts, lon0, lat0, 0.1, 0.1)
    np.testing.assert_allclose(rclv.polygon_area(verts_proj_60N),
                               area_equator_expected/2, rtol=0.01)

    lat0 = -60
    verts_proj_60S = rclv.project_vertices(verts, lon0, lat0, 0.1, 0.1)
    np.testing.assert_allclose(rclv.polygon_area(verts_proj_60S),
                               area_equator_expected/2, rtol=0.01)

def test_get_local_region():
    # create some data
    n = 10
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(2,2))
    assert x_reg.shape == (5,5)
    assert x_reg[j,i] == 0
    assert x_reg[j,0] == 2
    assert x_reg[j,-1] == -2

    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(3,2), border_i=(2,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,7), border_i=(2,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(3,2))
    with pytest.raises(ValueError) as ve:
        (j,i), x_reg = rclv.get_local_region(x, (2,2), border_j=(2,2), border_i=(2,7))


def test_is_contour_closed(square_verts):
    assert rclv.is_contour_closed(square_verts)
    assert not rclv.is_contour_closed(square_verts[:-1])


def test_point_in_contour(square_verts):
    assert rclv.point_in_contour(square_verts, (0., 0.))
    assert not rclv.point_in_contour(square_verts, (1., 0.))


def test_contour_area(square_verts):
    region_area, hull_area, convex_def = rclv.contour_area(square_verts)
    assert region_area == 1.0
    assert hull_area == 1.0
    assert convex_def == 0.0


def test_contour_area_projected(square_verts):
    lon0, lat0 = 0, 0
    dlon, dlat = 1, 1
    verts_proj = rclv.project_vertices(square_verts, lon0, lat0, dlon, dlat)
    region_area, hull_area, convex_def = rclv.contour_area(verts_proj)

    square_area_equator = 123.64311711e8

    np.testing.assert_allclose(region_area, square_area_equator)
    np.testing.assert_allclose(hull_area, square_area_equator)
    np.testing.assert_allclose(convex_def, 0.0)


def test_contour_around_maximum(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum

    # we should get an error if the contour intersects the domain boundary
    with pytest.raises(ValueError):
        _ = rclv.find_contour_around_maximum(psi, ji, psi_max + 0.1)

    con, region_data, border_i, border_j = rclv.find_contour_around_maximum(
                                                            psi, ji, psi_max/2)

    # region data should be normalized to have the center point 0
    assert region_data[border_i[1], border_j[0]] == 0.0
    assert region_data.shape == (sum(border_j)+1, sum(border_j)+1)

    # the contour should be closed
    assert rclv.is_contour_closed(con)

    # check size against reference solution
    region_area, hull_area, convex_def = rclv.contour_area(con)
    np.testing.assert_allclose(region_area, 575.02954788959767)
    np.testing.assert_allclose(hull_area, 575.0296629815823)
    assert convex_def == (hull_area - region_area) / region_area


def test_convex_contour_around_maximum(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum

    # step determines how precise the contour identification is
    step = 0.001
    con, area = rclv.convex_contour_around_maximum(psi, ji, step)

    # check against reference solution
    np.testing.assert_allclose(area, 2693.8731123245125)
    assert len(con) == 261

    # for this specific psi, contour should be symmetric around maximum
    assert tuple(con[:-1].mean(axis=0).astype('int')) == ji



def test_convex_contour_around_maximum_projected(sample_data_and_maximum,
                                                 sample_data_lon_lat):
    psi, ji, psi_max = sample_data_and_maximum
    lon, lat = sample_data_lon_lat

    proj_kwargs = dict(
        dlon=lon[1] - lon[0],
        dlat=lat[1] - lat[0],
        lon0=lon[ji[1]],
        lat0=lat[ji[0]],
    )

    # step determines how precise the contour identification is
    step = 0.001
    con, area = rclv.convex_contour_around_maximum(psi, ji, step,
                                proj_kwargs=proj_kwargs)

    # check against reference solution
    np.testing.assert_allclose(area, 2375686527)
    assert len(con) == 261



def test_find_convex_contours(sample_data_and_maximum):
    psi, ji, psi_max = sample_data_and_maximum
    res =list(rclv.find_convex_contours(psi, step=0.001))

    assert len(res) == 1

    ji_found, con, area = res[0]
    assert tuple(ji_found) == ji
    assert len(con) == 261
    np.testing.assert_allclose(area, 2693.8731123245125)

    # also test the "filling in" function
    labels = rclv.label_points_in_contours(psi.shape, [con])
    assert labels.max() == 1
    assert labels.min() == 0
    assert labels.sum() == 2693


def test_find_convex_contours_projected(sample_data_and_maximum,
                                        sample_data_lon_lat):
    psi, ji, psi_max = sample_data_and_maximum
    lon, lat = sample_data_lon_lat

    with pytest.raises(ValueError):
        _ = list(rclv.find_convex_contours(psi, step=0.001, lon=lon[1:], lat=lat))

    res = list(rclv.find_convex_contours(psi, step=0.001, lon=lon, lat=lat))

    assert len(res) == 1

    ji_found, con, area = res[0]
    assert tuple(ji_found) == ji
    assert len(con) == 261
    np.testing.assert_allclose(area, 2375686527)
