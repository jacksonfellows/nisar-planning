import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from functools import cache

import main
from scipy.interpolate import LinearNDInterpolator
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import itertools

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# def load_tracks(tracks_shp):
#     tracks_gdf = gpd.read_file(tracks_shp)
#     if tracks_gdf.crs is None:
#         raise ValueError("Tracks CRS is undefined.")
#     if tracks_gdf.crs.to_epsg() != 3031:
#         print("Converting tracks CRS to EPSG:3031")
#         tracks_gdf = tracks_gdf.to_crs("EPSG:3031")
#     return tracks_gdf

def make_grid(bounds, dx, dy):
    """Make a grid from bounds tuple (minx, miny, maxx, maxy)"""
    minx, miny, maxx, maxy = bounds
    xx, yy = np.meshgrid(np.arange(minx, maxx, dx), np.arange(miny, maxy, dy))
    return {"xx": xx, "yy": yy} # dict as crude grid object

# def count_tracks_in_grid(grid, tracks):
#     grid_points = gpd.points_from_xy(grid["xx"].flatten(), grid["yy"].flatten())
#     tracks_count = np.zeros(np.prod(grid["xx"].shape))
#     for poly in tracks.geometry:
#         tracks_count += grid_points.within(poly)
#     tracks_count = tracks_count.reshape(grid["xx"].shape)
#     plt.imshow(tracks_count)
#     plt.colorbar()
#     plt.show()

# @cache
# def load_intersecting_tracks(
#         study_area_shp="./boundaries/ross.shp",
#         t0="2025-11-24T00:00:00.000Z",
#         n_hours=24,
#         dt_s=5*60,
#         min_incidence_angle_deg=34,
#         max_incidence_angle_deg=48,
#         ):
#     # Load study area
#     study_area_gdf = gpd.read_file(study_area_shp)
#     if study_area_gdf.crs is None:
#         raise ValueError("Study area CRS is undefined.")
#     if study_area_gdf.crs.to_epsg() != 3031:
#         print("Converting study area CRS to EPSG:3031")
#         study_area_gdf = study_area_gdf.to_crs("EPSG:3031")
#     study_area_poly = study_area_gdf.geometry.iloc[0]

#     # Load satellite
#     sat = main.load_satellite()

#     # Parse t0 and run
#     t0_dt = main.parse_utc_datetime(t0)

#     return main.compute_intersections(sat, study_area_poly, t0_dt, n_hours, dt_s, min_incidence_angle_deg, max_incidence_angle_deg)

# Test function to plot track with interpolated normals.
# def plot_track_with_interpolated_los(track):
#     # plot a track's los and azimuth vectors in 3d
#     # matplotlib's 3d plot is not great but this is a quick way to check if they are pointing the right direction
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_box_aspect([1,1,1])
#     bounds = [track["coords"][:,0].min(), track["coords"][:,0].max(),
#               track["coords"][:,1].min(), track["coords"][:,1].max(),
#               0, 0]
#     max_range = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
#     mid_x = (bounds[0] + bounds[1]) / 2
#     mid_y = (bounds[2] + bounds[3]) / 2
#     ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
#     ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
#     ax.set_zlim(-max_range/2, max_range/2)

#     grid = make_grid((bounds[0], bounds[2], bounds[1], bounds[3]), dx=10**4, dy=10**4)
#     grid_points = gpd.points_from_xy(grid["xx"].flatten(), grid["yy"].flatten())
#     in_track = np.zeros(np.prod(grid["xx"].shape))
#     in_track += grid_points.within(track["poly"])
#     in_track = in_track.reshape(grid["xx"].shape)
#     ax.plot_surface(grid["xx"], grid["yy"], np.zeros_like(grid["xx"]), facecolors=plt.cm.gray(in_track), alpha=0.3, shade=False)

#     # Original LOS and azimuth vectors at track coords
#     L = 10**5
#     ax.quiver(track["coords"][:,0],track["coords"][:,1],np.zeros_like(track["coords"][:,0]),track["los"][:,0], track["los"][:,1], track["los"][:,2],length=L,color="blue",normalize=True)
#     ax.quiver(track["coords"][:,0],track["coords"][:,1],np.zeros_like(track["coords"][:,0]),track["azimuth"][:,0], track["azimuth"][:,1], track["azimuth"][:,2],length=L,color="red",normalize=True)
    
#     # Create interpolator for LOS vector components
#     los_interp = LinearNDInterpolator(track["coords"], track["los"])
    
#     # Downsample grid by 10 in each dimension for plotting
#     downsample_factor = 10
#     xx_down = grid["xx"][::downsample_factor, ::downsample_factor]
#     yy_down = grid["yy"][::downsample_factor, ::downsample_factor]
    
#     # Interpolate LOS on downsampled grid
#     los_interpolated = los_interp(xx_down, yy_down)
    
#     # Plot interpolated LOS vectors in green
#     ax.quiver(xx_down, yy_down, np.zeros_like(xx_down),
#               los_interpolated[:, :, 0], los_interpolated[:, :, 1], los_interpolated[:, :, 2],
#               length=L, normalize=True, color='green', alpha=0.7)
    
#     # Create interpolator for azimuth vector components
#     azimuth_interp = LinearNDInterpolator(track["coords"], track["azimuth"])
    
#     # Interpolate azimuth on downsampled grid
#     azimuth_interpolated = azimuth_interp(xx_down, yy_down)
    
#     # Plot interpolated azimuth vectors in purple
#     ax.quiver(xx_down, yy_down, np.zeros_like(xx_down),
#               azimuth_interpolated[:, :, 0], azimuth_interpolated[:, :, 1], azimuth_interpolated[:, :, 2],
#               length=L, normalize=True, color='purple', alpha=0.7)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Track with Original and Interpolated Normals')
    
#     plt.show()

def calc_G_harmonic(los_vectors, azimuth_vectors, times):
    T_M2 = 12.4206012 # hr
    omega_M2 = 2*np.pi/T_M2 # rad/hr
    vectors = [*zip(los_vectors, itertools.repeat(True)), *zip(azimuth_vectors, itertools.repeat(False))]
    assert len(los_vectors)==len(times)
    if len(azimuth_vectors) > 0: assert len(azimuth_vectors)==len(times)
    G = np.zeros((len(vectors), 9))
    # G = np.zeros((len(vectors), 10))
    for i,((vec,is_los),ta) in enumerate(zip(vectors, itertools.cycle(times))):
        # TODO better way to get hours?
        ta = ta.astype('datetime64[us]').astype(np.double)/(10**6)/(60*60)
        tb = ta + 12*24 # 12 days later
        dt = tb - ta
        dcos = np.cos(omega_M2*tb) - np.cos(omega_M2*ta)
        dsin = np.sin(omega_M2*tb) - np.sin(omega_M2*ta)
        G[i,0:3] = dt*vec
        G[i,3:6] = dcos*vec
        G[i,6:9] = dsin*vec
        # if is_los:
        #     G[i,9] = 1e-3 # psi_ab = B_perp_ab/(r_0*sin(theta_0))
    return G

def calc_sensitivity(
    study_area_shp,
    t0,
    output_path,
    n_hours=24*12,
    dt_s=5*60,
    min_incidence_angle_deg=34,
    max_incidence_angle_deg=48,
    grid_spacing_m=10**3
):
    # Load study area
    study_area_gdf = gpd.read_file(study_area_shp)
    if study_area_gdf.crs is None:
        raise ValueError("Study area CRS is undefined.")
    if study_area_gdf.crs.to_epsg() != 3031:
        print("Converting study area CRS to EPSG:3031")
        study_area_gdf = study_area_gdf.to_crs("EPSG:3031")
    study_area_poly = study_area_gdf.geometry.iloc[0]

    # Get intersecting orbit tracks
    sat = main.load_satellite()
    t0_dt = main.parse_utc_datetime(t0)
    tracks = main.compute_intersections(sat, study_area_poly, t0_dt, n_hours, dt_s, min_incidence_angle_deg, max_incidence_angle_deg)

    # Make a grid for the study area
    grid = make_grid(study_area_poly.bounds, dx=grid_spacing_m, dy=grid_spacing_m)
    grid_points = gpd.points_from_xy(grid["xx"].flatten(), grid["yy"].flatten())

    # Find intersecting grid cells and make normal interpolators for each track
    track_info = []
    for track in tracks:
        in_track = grid_points.within(track["poly"]).reshape(grid["xx"].shape)
        los_interp = LinearNDInterpolator(track["coords"], track["los"])
        azimuth_interp = LinearNDInterpolator(track["coords"], track["azimuth"])
        times_double = track["time"].astype('datetime64[us]').astype(np.double)
        time_interp = LinearNDInterpolator(track["coords"], times_double)
        track_info.append(dict(in_track=in_track, los_interp=los_interp, azimuth_interp=azimuth_interp, time_interp=time_interp))

    # Calculate sensitivities for each grid point (and number of tracks/cell)
    Lambda_los = np.full_like(grid["xx"], np.nan)
    Lambda_both = np.full_like(grid["xx"], np.nan)
    Lambda_m2_los = np.full_like(grid["xx"], np.nan)
    Lambda_m2_both = np.full_like(grid["xx"], np.nan)
    num_tracks = np.zeros_like(grid["xx"])
    for idx in np.ndindex(grid["xx"].shape):
        x = grid["xx"][idx]
        y = grid["yy"][idx]
        los_vectors = []
        azimuth_vectors = []
        times = []
        for info in track_info:
            if info["in_track"][idx]:
                los_vectors.append(info["los_interp"](x, y))
                azimuth_vectors.append(info["azimuth_interp"](x, y))
                times.append(info["time_interp"](x, y).astype('datetime64[us]'))
                num_tracks[idx] += 1
        if len(los_vectors) >= 3:
            try:
                G_los = np.vstack(los_vectors)
                G_both = np.vstack((*los_vectors, *azimuth_vectors))
                Lambda_los[idx] = np.linalg.trace(np.linalg.inv(G_los.T@G_los))
                Lambda_both[idx] = np.linalg.trace(np.linalg.inv(G_both.T@G_both))
                G_m2_los = calc_G_harmonic(los_vectors, [], times)
                Lambda_m2_los[idx] = np.linalg.trace(np.linalg.inv(G_m2_los.T@G_m2_los))
                G_m2_both = calc_G_harmonic(los_vectors, azimuth_vectors, times)
                Lambda_m2_both[idx] = np.linalg.trace(np.linalg.inv(G_m2_both.T@G_m2_both))
            except np.linalg.LinAlgError:
                pass

    # # Plot results
    # fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    # ax0, ax1, ax2, ax3, ax4, ax5 = axs.flatten()
    
    # im0 = ax0.pcolormesh(grid["xx"], grid["yy"], num_tracks)
    # ax0.set_title('Number of Tracks')
    # ax0.set_xlabel('X')
    # ax0.set_ylabel('Y')
    # plt.colorbar(im0, ax=ax0)
    
    # im1 = ax1.pcolormesh(grid["xx"], grid["yy"], Lambda_los)
    # ax1.set_title('$\\Lambda_g$ — LOS')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # plt.colorbar(im1, ax=ax1)
    
    # im2 = ax2.pcolormesh(grid["xx"], grid["yy"], Lambda_both)
    # ax2.set_title('$\\Lambda_g$ — LOS + azimuth')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # plt.colorbar(im2, ax=ax2)

    # im3 = ax3.pcolormesh(grid["xx"], grid["yy"], Lambda_m2_los)
    # ax3.set_title('$\\Lambda_g$ — M2, LOS, no $\\psi_{ab}$')
    # ax3.set_xlabel('X')
    # ax3.set_ylabel('Y')
    # plt.colorbar(im3, ax=ax3)

    # im4 = ax4.pcolormesh(grid["xx"], grid["yy"], Lambda_m2_both)
    # ax4.set_title('$\\Lambda_g$ — M2, LOS + azimuth, no $\\psi_{ab}$')
    # ax4.set_xlabel('X')
    # ax4.set_ylabel('Y')
    # plt.colorbar(im4, ax=ax4)
    
    # plt.tight_layout()
    # plt.show()

    if output_path is not None:
        np.savez(
            f'{output_path}.npz', 
            xx=grid["xx"], 
            yy=grid["yy"], 
            num_tracks=num_tracks,
            Lambda_los=Lambda_los, 
            Lambda_both=Lambda_both,
            Lambda_m2_los=Lambda_m2_los,
            Lambda_m2_both=Lambda_m2_both,
        )

def plot_sensitivity(npz_path, title, scale_loc):
    d = np.load(npz_path)

    fig, axs_ = plt.subplots(figsize=(15, 10), nrows=2, ncols=3, subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
    axs_[1,0].set_visible(False)
    axs = [axs_[0,0], axs_[0,1], axs_[0,2], axs_[1,1], axs_[1,2]]

    fig.suptitle(title, fontsize=24, y=0.98)

    # Number of tracks
    axs[0].add_feature(cfeature.COASTLINE, edgecolor='black')
    num_tracks_masked = np.ma.masked_where(d["num_tracks"] == 0, d["num_tracks"])
    im0 = axs[0].pcolormesh(d["xx"], d["yy"], num_tracks_masked, transform=ccrs.epsg(3031))
    axs[0].set_title('Number of Tracks')
    plt.colorbar(im0, ax=axs[0])

    # Lambda LOS
    axs[1].add_feature(cfeature.COASTLINE, edgecolor='black')
    im1 = axs[1].pcolormesh(d["xx"], d["yy"], d["Lambda_los"], transform=ccrs.epsg(3031))
    axs[1].set_title('$\\Lambda_g$ — LOS')
    plt.colorbar(im1, ax=axs[1])

    # Lambda both
    axs[2].add_feature(cfeature.COASTLINE, edgecolor='black')
    im2 = axs[2].pcolormesh(d["xx"], d["yy"], d["Lambda_both"], transform=ccrs.epsg(3031))
    axs[2].set_title('$\\Lambda_g$ — LOS + azimuth')
    plt.colorbar(im2, ax=axs[2])

    # Lambda M2 LOS
    axs[3].add_feature(cfeature.COASTLINE, edgecolor='black')
    im3 = axs[3].pcolormesh(d["xx"], d["yy"], d["Lambda_m2_los"], transform=ccrs.epsg(3031))
    axs[3].set_title('$\\Lambda_g$ — M2, LOS, no $\\psi_{ab}$')
    plt.colorbar(im3, ax=axs[3])

    # Lambda M2 both
    axs[4].add_feature(cfeature.COASTLINE, edgecolor='black')
    im4 = axs[4].pcolormesh(d["xx"], d["yy"], d["Lambda_m2_both"], transform=ccrs.epsg(3031))
    axs[4].set_title('$\\Lambda_g$ — M2, LOS + azimuth, no $\\psi_{ab}$')
    plt.colorbar(im4, ax=axs[4])

    for ax in axs:
        x = d["xx"].min() + (d["xx"].max() - d["xx"].min())*scale_loc[0]
        y = d["yy"].min() + (d["yy"].max() - d["yy"].min())*scale_loc[1]
        scale_length = 100e3
        rect = Rectangle((x - scale_length/2, y - 5e3), scale_length, 10e3, facecolor='black', transform=ccrs.epsg(3031))
        ax.add_patch(rect)
        ax.text(x, y - 20e3, '100 km', ha='center', va='top', fontsize=10, transform=ccrs.epsg(3031))

    plt.tight_layout()
    plt.show()

def plot_all():
    plot_sensitivity("ross_sensitivities.npz", "Ross Ice Shelf", (0.8,0.8))
    plot_sensitivity("fris_sensitivities.npz", "Filchner-Ronne Ice Shelf", (0.8,0.15))
    plot_sensitivity("amery_sensitivities.npz", "Amery Ice Shelf", (0.3,0.15))


# if __name__ == "__main__":
#     t0 = "2025-11-24T00:00:00.000Z"
#     for area in ["ross", "fris", "amery"]:
#         study_area_shp = f"boundaries/{area}.shp"
#         calc_sensitivity(study_area_shp, t0, f"{area}_sensitivities")