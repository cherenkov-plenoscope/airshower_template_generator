import numpy as np
import corsika_primary_wrapper as cpw
import tempfile
import os
import scipy
from scipy import spatial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


config = {
    "corsika_primary_path": os.path.join(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd",
    ),
    "sites": {
        "namibia": {
            "observation_level_asl_m": 2300,
            "earth_magnetic_field_x_muT": 12.5,
            "earth_magnetic_field_z_muT": -25.9,
            "atmosphere_id": 10,
            "geomagnetic_cutoff_rigidity_GV": 12.5,
        },
    },
    "particles": {
        "gamma": {
            "particle_id": 1,
        },
    },
    "energy_supports_GeV": np.geomspace(1e0, 1e3, 16),
    "azimuth_supports_deg": [0.0],
    "radius_supports_m": np.linspace(0, 1.25e3, 25),
    "aperture_radius_m": 10.0,
    "maximum_bin_edges_m": np.linspace(5e3, 25e3, 10),
    "max_energy_in_run": 1e3,
    "c_parallel_bin_edges_deg": np.linspace(-0.5, 2.5, 64+32 + 1),
    "c_perpendicular_bin_edges_deg": np.linspace(-0.5, 0.5, 32 + 1),
}


def make_corsika_steering_card(
    site, particle, energy, num_airshower
):
    steering = {
        "run": {
            "run_id": 1,
            "event_id_of_first_event": 1,
            "observation_level_asl_m": site["observation_level_asl_m"],
            "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
            "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
            "atmosphere_id": site["atmosphere_id"],
        },
        "primaries": []
    }

    for i in range(num_airshower):
        primary = {
            "particle_id": particle["particle_id"],
            "energy_GeV": energy,
            "zenith_rad": 0.0,
            "azimuth_rad": 0.0,
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw.simple_seed(i + 1),
        }
        steering["primaries"].append(primary)
    return steering


def _find_bin_in_edges(bin_edges, value):
    upper_bin_edge = int(np.digitize([value], bin_edges)[0])
    if upper_bin_edge == 0:
        return True, 0, False
    if upper_bin_edge == bin_edges.shape[0]:
        return False, upper_bin_edge - 1, True
    return False, upper_bin_edge - 1, False


def _project_to_image(
    cxs,
    cys,
    c_parallel_bin_edges,
    c_perpendicular_bin_edges,
    x,
    y,
):
    cys_ = cxs
    cxs_ = cys
    azimuth = np.arctan2(y, x)

    cPara = np.cos(-azimuth)*cys_ - np.sin(-azimuth)*cxs_
    cPerp = np.sin(-azimuth)*cys_ + np.cos(-azimuth)*cxs_

    hist = np.histogram2d(
        x=cPara,
        y=cPerp,
        bins=(c_parallel_bin_edges, c_perpendicular_bin_edges))[0]
    return hist


energy = 1.0


steering_dict = make_corsika_steering_card(
    site=config["sites"]["namibia"],
    particle=config["particles"]["gamma"],
    energy=energy,
    num_airshower=int(np.ceil(config["max_energy_in_run"]/energy))
)


corsika = cpw.CorsikaPrimary(
    corsika_path=config["corsika_primary_path"],
    steering_dict=steering_dict,
    stdout_path="corsika.o",
    stderr_path="corsika.e",
)


xy_supports = []
for az_deg in config["azimuth_supports_deg"]:
    for r_m in config["radius_supports_m"]:
        sup_x = np.cos(np.deg2rad(az_deg))*r_m
        sup_y = np.sin(np.deg2rad(az_deg))*r_m
        xy_supports.append([sup_x, sup_y])
xy_supports = np.array(xy_supports)

max_energy_in_altitude_bin = 1e2
num_altitude_bins = 15
altitude_bin_edges = np.linspace(5e3, 35e3, num_altitude_bins + 1)
energies_in_altitude_bins = np.zeros(num_altitude_bins)
num_thrown_in_altitude_bins = np.zeros(num_altitude_bins, dtype=np.int)


num_azimuth_bins = len(config["azimuth_supports_deg"])
num_radial_bins = len(config["radius_supports_m"])

views = {}
for az in range(num_azimuth_bins):
    views[az] = {}
    for r in range(num_radial_bins):
        views[az][r] = {}
        for alt in range(num_altitude_bins):
            views[az][r][alt] = np.zeros(
                shape=(
                    len(config["c_parallel_bin_edges_deg"]) - 1,
                    len(config["c_perpendicular_bin_edges_deg"]) - 1,
                )
            )

for airshower in corsika:
    event_header, cherenkov_bunches = airshower

    num_bunches = cherenkov_bunches.shape[0]

    if num_bunches == 0:
        print("zero")
        continue

    airshower_maximum_altitude_asl_m = 1e-2*np.median(
        cherenkov_bunches[:, cpw.IZEM]
    )

    underflow, altitude_bin, overflow = _find_bin_in_edges(
        value=airshower_maximum_altitude_asl_m,
        bin_edges=altitude_bin_edges
    )

    if underflow or overflow:
        # print("over-under-flow")
        continue

    num_thrown_in_altitude_bins[altitude_bin] += 1

    if (
        energies_in_altitude_bins[altitude_bin]
        >= max_energy_in_altitude_bin
    ):
        # print("full")
        continue

    energies_in_altitude_bins[altitude_bin] += energy


    xy_tree = scipy.spatial.KDTree(
        data=np.c_[
            1e-2*cherenkov_bunches[:, cpw.IX],
            1e-2*cherenkov_bunches[:, cpw.IY]
        ]
    )

    meets = xy_tree.query_ball_point(
        x=xy_supports,
        r=config["aperture_radius_m"]
    )

    for i, meet in enumerate(meets):
        view = cherenkov_bunches[meet, :]

        img = _project_to_image(
            cxs=view[:, cpw.ICX],
            cys=view[:, cpw.ICY],
            c_parallel_bin_edges=np.deg2rad(config["c_parallel_bin_edges_deg"]),
            c_perpendicular_bin_edges=np.deg2rad(config["c_perpendicular_bin_edges_deg"]),
            x=xy_supports[i, 0],
            y=xy_supports[i, 1],
        )

        az = i // num_radial_bins
        ra = i % num_radial_bins

        views[az][ra][altitude_bin] += img

    print(num_thrown_in_altitude_bins)


for az in range(num_azimuth_bins):
    for r in range(num_radial_bins):
        for alt in range(num_altitude_bins):

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.pcolormesh(views[az][r][alt].T)
            ax.set_aspect("equal")
            ax.set_xlabel("c para / deg")
            ax.set_ylabel("c perp / deg")
            fig.savefig("{:03d}_{:03d}_{:03d}_img.jpg".format(az, r, alt))
            plt.close(fig)