import datetime
import numpy as np
import click
import geopandas as gpd
import pygmt
import pyproj
import shapely
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric, ICRF
from skyfield.timelib import Time
from skyfield.units import Distance, Velocity

# TLE_RELOAD = True
TLE_RELOAD = False

ts = load.timescale()

lonlat_to_south_polar = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
itrf_to_south_polar = pyproj.Transformer.from_crs("EPSG:7789", "EPSG:3031", always_xy=True)

def latlon_to_xy(coords_ll):
    """
    coords_ll: (N, 2) array [[lat, lon], ...]
    returns: (N, 2) array [[x, y], ...] in projected units
    """
    lon = coords_ll[:, 1]
    lat = coords_ll[:, 0]
    x, y = lonlat_to_south_polar.transform(lon, lat)
    return np.column_stack([x, y])

def xy_to_latlon(coords_xy):
    """
    coords_xy: (N, 2) array [[x, y], ...] in projected units
    returns: (N, 2) array [[lat, lon], ...]
    """
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    lon, lat = lonlat_to_south_polar.transform(x, y, direction='INVERSE')
    return np.column_stack([lat, lon])

def calc_local_to_itrf_transform(g: Geocentric):
    """
    Returns a matrix that transforms coordinates from a local frame of reference (along-track, cross-track, up) to the ITRF frame (Earth-centered/fixed) used by skyfield.
    """

    # Satellite position/velocity in ITRF frame
    (r_itrf, v_itrf) = g.frame_xyz_and_velocity(itrs)
    r = np.array(r_itrf.km)
    v = np.array(v_itrf.km_per_s)

    # Ground projection of satellite
    sub = wgs84.subpoint(g)

    # Up vector at ground projection
    r_sub = sub.itrs_xyz.km
    U = r_sub/np.linalg.norm(r_sub)

    # Along-track vector
    v_tan = v - np.dot(v, U)*U # subtract component of velocity that is along local up vector
    A = v_tan/np.linalg.norm(v_tan)

    # Cross-track vector = U X A
    C = np.cross(U, A)/np.linalg.norm(np.cross(U, A))

    # R maps ITRF frame to local frame
    R = np.vstack([A, C, U])

    return R.T # Return R.T --- transforms from local frame to ITRF frame

def find_swatch_line(
    sat: EarthSatellite,
    t: Time,
    min_incidence_angle_deg: float,
    max_incidence_angle_deg: float
):
    g: Geocentric = sat.at(t)
    RT = calc_local_to_itrf_transform(g)
    r_sat = np.linalg.norm(g.itrf_xyz().km)
    r_earth = 6371
    l = lambda theta: r_sat*np.cos(theta) - np.sqrt(r_sat**2*np.cos(theta)**2 - r_sat**2 + r_earth**2)
    P = lambda theta: np.array([0, l(theta)*np.sin(theta), r_sat - l(theta)*np.cos(theta)])
    line = []
    for theta_deg in [min_incidence_angle_deg, max_incidence_angle_deg]:
        p_itrf = RT@P(np.deg2rad(theta_deg))
        p = ICRF.from_time_and_frame_vectors(t, itrs, Distance(km=p_itrf), Velocity(km_per_s=np.zeros(3)))
        p.center = 399 # geocentric
        # look at WGS84 elevation to see the error from my spherical assumptions?
        lat,lon = wgs84.latlon_of(p)
        # calc los and azimuth normals -- TODO plot these with tracks and check that they are correct in south polar projection
        p_itrf_m = p_itrf*1e3
        los = np.array(itrf_to_south_polar.transform(*p_itrf_m)) - np.array(itrf_to_south_polar.transform(*g.itrf_xyz().m))
        los /= np.linalg.norm(los)
        azimuth = np.array(itrf_to_south_polar.transform(*(p_itrf_m + RT@np.array([1,0,0])))) - np.array(itrf_to_south_polar.transform(*p_itrf_m))
        azimuth /= np.linalg.norm(azimuth)
        # With this method I'm getting azimuth z values from around 5e-3 to 1e-4.
        # Not sure if this will be a problem, but I think that they should be strictly 0.
        # This is probably a result of computing R^T (and its along-track vector) in the ITRF frame and then transforming to south polar.
        line.append(dict(latlon=(lat.degrees, lon.degrees), normals=dict(los=los, azimuth=azimuth)))
    return line

def load_satellite():
    [sat,] = load.tle_file(
        "https://celestrak.org/NORAD/elements/gp.php?CATNR=65053&FORMAT=TLE",
        filename="NISAR.tle",
        reload=TLE_RELOAD,
        ts=ts,
    )
    click.echo(f"Satellite: {sat}")
    click.echo(f"Elasped time since epoch: {datetime.datetime.now(tz=datetime.timezone.utc) - sat.epoch.utc_datetime()}")
    return sat

def parse_utc_datetime(s: str) -> datetime.datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt

def compute_intersections(
    sat: EarthSatellite,
    study_area_poly,
    t0_dt: datetime.datetime,
    n_hours: int,
    dt_s: int,
    min_incidence_angle_deg: float,
    max_incidence_angle_deg: float
):
    """
    Find the tracks intersecting the study area for n_hours after t0.

    Returns a list [{"los": ..., "azimuth": ..., "time": ..., "coords": ..., "poly": ...}, ...],
    where "los" and "azimuth" are the respective normals at the corners of the track rectangle defined by "coords",
    "time" is the time the satellite passes the corners defined by "coords",
    and "poly" is a shapely polygon of the intersection with the study area.
    """
    t0 = ts.from_datetime(t0_dt)
    t1 = t0 + datetime.timedelta(hours=n_hours)
    dt = datetime.timedelta(seconds=dt_s)

    t = t0
    prev_line = find_swatch_line(sat, t, min_incidence_angle_deg, max_incidence_angle_deg)

    intersections = []

    while t < t1:
        t += dt
        line = find_swatch_line(sat, t, min_incidence_angle_deg, max_incidence_angle_deg)
        poly = shapely.geometry.Polygon(
            latlon_to_xy(np.array([prev_line[0]["latlon"], prev_line[1]["latlon"], line[1]["latlon"], line[0]["latlon"], prev_line[0]["latlon"]]))
        )
        if shapely.intersects(poly, study_area_poly):
            los = np.array([x["normals"]["los"] for x in [*prev_line, *line]])
            azimuth = np.array([x["normals"]["azimuth"] for x in [*prev_line, *line]])
            t_prev = np.datetime64((t - dt).utc_datetime().replace(tzinfo=None))
            t_now = np.datetime64(t.utc_datetime().replace(tzinfo=None))
            time = np.array((t_prev, t_prev, t_now, t_now), dtype=np.datetime64)
            coords = latlon_to_xy(np.array([x["latlon"] for x in [*prev_line, *line]]))
            intersections.append(dict(
                los=los,
                azimuth=azimuth,
                time=time,
                coords=coords,
                poly=shapely.intersection(poly, study_area_poly)),
            )
            
        prev_line = line

    click.echo(f"Found {len(intersections)} flight lines that intersect the study area")
    return intersections

def save_intersections(intersections, crs, output_shp: str):
    intersections_gdf = gpd.GeoDataFrame(geometry=[i["poly"] for i in intersections], crs=crs)
    intersections_gdf.to_file(output_shp)
    click.echo(f"Exported {len(intersections_gdf)} intersection polygons to {output_shp}")

def plot_intersections(intersections, study_area_poly):
    fig = pygmt.Figure()

    projection = "S0/-90/30/20c"

    # Create the basemap with land, water, and coastlines
    fig.coast(
        projection=projection,
        region="g",
        frame="afg", # frame, gridlines, annotations
        land="lightgray",
        water="lightblue",
        shorelines="0.5p,black"
    )

    for i in intersections:
        poly = i["poly"]
        if poly.geom_type == 'Polygon':
            geoms = [poly]
        elif poly.geom_type == 'MultiPolygon':
            geoms = poly.geoms
        else:
            assert 0 # unsupported geometry type

        for geom in geoms:
            latlons = xy_to_latlon(np.array(geom.exterior.coords))
            fig.plot(x=latlons[:,1], y=latlons[:,0], fill="red@90", pen="0.1p,black")

    # Plot the study area outline
    latlons = xy_to_latlon(np.array(study_area_poly.exterior.coords))
    fig.plot(x=latlons[:,1], y=latlons[:,0], pen="1p,blue")

    fig.show(width=1000)

@click.command()
@click.argument("study_area_shp", type=click.Path(exists=True, dir_okay=False))
@click.argument("t0", type=str)
@click.argument("output_shp", type=str)
@click.option("--n-hours", default=12*24, show_default=True, type=int, help="Number of hours to simulate from t0.")
@click.option("--dt-s", default=5*60, show_default=True, type=int, help="Timestep in seconds.")
@click.option("--min-incidence-angle-deg", default=34.0, show_default=True, type=float, help="Minimum incidence angle in degrees.")
@click.option("--max-incidence-angle-deg", default=48.0, show_default=True, type=float, help="Maximum incidence angle in degrees.")
def main(
    study_area_shp: str,
    t0: str,
    output_shp: str,
    n_hours: int,
    dt_s: int,
    min_incidence_angle_deg: float,
    max_incidence_angle_deg: float
):
    # Load study area
    study_area_gdf = gpd.read_file(study_area_shp)
    if study_area_gdf.crs is None:
        raise ValueError("Study area CRS is undefined.")
    if study_area_gdf.crs.to_epsg() != 3031:
        click.echo("Converting study area CRS to EPSG:3031")
        study_area_gdf = study_area_gdf.to_crs("EPSG:3031")
    study_area_poly = study_area_gdf.geometry.iloc[0]

    # Load satellite
    sat = load_satellite()

    # Parse t0 and run
    t0_dt = parse_utc_datetime(t0)

    intersections = compute_intersections(sat, study_area_poly, t0_dt, n_hours, dt_s, min_incidence_angle_deg, max_incidence_angle_deg)
    save_intersections(intersections, study_area_gdf.crs, output_shp)
    plot_intersections(intersections, study_area_poly)

if __name__ == "__main__":
    main()