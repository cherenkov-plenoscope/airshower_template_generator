import numpy as np
import corsika_primary as cpw
import tempfile
import os
import io
import scipy
from scipy import spatial
import json
import tarfile
import gzip
import json_utils
import binning_utils
import glob
import rename_after_writing as rnafwr

from . import bins
from . import quality
from . import input_output
from . import projection


def make_jobs(work_dir):
    """
    Returns a list of jobs to be run by 'run_job(job)'.
    When run, each job populates a fraction of the look-up-table.
    Jobs can run in parallel.

    Parameters
    ----------
    work_dir : str, path
            The working-directory where the config-fiels are stored.
    """

    map_dir = os.path.join(work_dir, "map")

    sites = json_utils.read(os.path.join(work_dir, "sites.json"))
    particles = json_utils.read(os.path.join(work_dir, "particles.json"))
    binning = json_utils.read(os.path.join(work_dir, "binning.json"))
    run_config = json_utils.read(os.path.join(work_dir, "run_config.json"))

    explicit_binning = bins.make_explicit_binning(binning=binning)
    energy_bin_supports = explicit_binning["energy_GeV"]["supports"]

    az = binning["azimuth_deg"]
    assert image_pixels_are_square(binning=binning)
    assert (
        binning["aperture_radius_for_timing_m"] >= binning["aperture_radius_m"]
    )
    jobs = []
    for site_key in sites:
        for particle_key in particles:
            for energy_bin, energy_GeV in enumerate(energy_bin_supports):
                energy_key = "energy_bin_{:06d}".format(energy_bin)
                for energy_job in range(run_config["num_jobs_in_energy_bin"]):
                    job = {}
                    job["map_dir"] = os.path.join(
                        map_dir, site_key, particle_key, energy_key
                    )
                    job["site"] = sites[site_key]
                    job["particle"] = particles[particle_key]
                    job["energy_GeV"] = energy_GeV
                    job["energy_bin"] = energy_bin
                    job["energy_job"] = energy_job
                    job["binning"] = binning
                    job["run_config"] = run_config
                    jobs.append(job)
    return jobs


def parallel_pixel_width_rad(binning):
    para = binning["image_parallel_deg"]
    pixel_edge_deg = (para["stop_edge"] - para["start_edge"]) / para[
        "num_bins"
    ]
    return np.deg2rad(pixel_edge_deg)


def area_of_aperture_m2(binning):
    return np.pi * binning["aperture_radius_m"] ** 2


def solid_angle_of_pixel_sr(binning):
    assert image_pixels_are_square(binning=binning)
    pixel_edge_rad = parallel_pixel_width_rad(binning=binning)
    return pixel_edge_rad**2


def time_slice_duration_s(binning):
    tim = binning["time_s"]
    duration_s = (tim["stop_edge"] - tim["start_edge"]) / tim["num_bins"]
    return duration_s


def make_corsika_steering_card(
    site, particle, energy, num_airshower, random_seed
):
    i8 = np.int64
    f8 = np.float64

    run_id = random_seed + 1
    run = {
        "run_id": i8(run_id),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(site["observation_level_asl_m"]),
        "earth_magnetic_field_x_muT": f8(site["earth_magnetic_field_x_muT"]),
        "earth_magnetic_field_z_muT": f8(site["earth_magnetic_field_z_muT"]),
        "atmosphere_id": i8(site["atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(energy * 0.99),
            "stop_GeV": f8(energy * 1.01),
        },
        "random_seed": cpw.random.seed.make_simple_seed(run_id),
    }
    primaries = []
    for i in range(num_airshower):
        primary = {
            "particle_id": f8(particle["corsika_particle_id"]),
            "energy_GeV": f8(energy),
            "zenith_rad": f8(0.0),
            "azimuth_rad": f8(0.0),
            "depth_g_per_cm2": f8(0.0),
        }
        primaries.append(primary)
    steering = {
        "run": run,
        "primaries": primaries,
    }
    return steering


def image_pixels_are_square(binning):
    para = binning["image_parallel_deg"]
    density_para = para["num_bins"] / (para["stop_edge"] - para["start_edge"])
    perp = binning["image_perpendicular_deg"]
    density_perp = perp["num_bins"] / (perp["stop_edge"] - perp["start_edge"])
    return np.abs(density_para / density_perp - 1.0) < 1e-6


def zeros(binning, keys=[], dtype=np.float32):
    return np.zeros(
        shape=[binning[key]["num_bins"] for key in keys],
        dtype=dtype,
    )


def run_job(job):
    """
    Populates a part of the look-up-table.

    Parameters
    ----------
    job : dict
            The full controll to run the simulation and population of a part
            in the look-up-table.
    """
    os.makedirs(job["map_dir"], exist_ok=True)
    result_path = os.path.join(
        job["map_dir"], "{:06d}.tar".format(job["energy_job"])
    )
    explbin = bins.make_explicit_binning(binning=job["binning"])
    c_para_bin_edges_rad = np.deg2rad(explbin["image_parallel_deg"]["edges"])
    c_perp_bin_edges_rad = np.deg2rad(
        explbin["image_perpendicular_deg"]["edges"]
    )
    time_bin_edges_s = explbin["time_s"]["edges"]
    altitude_bin_edges_m = explbin["altitude_m"]["edges"]

    xy_supports = bins.full_coverage_xy_supports_on_observationlevel(
        binning=job["binning"]
    )

    num_airshowers_to_be_thrown = int(
        np.ceil(
            job["run_config"]["energy_to_be_thrown_in_job_GeV"]
            / job["energy_GeV"]
        )
    )

    max_num_airshower_to_collect_in_altitude_bin = int(
        np.ceil(
            job["run_config"]["max_energy_to_collect_in_altitude_bin_GeV"]
            / job["energy_GeV"]
        )
    )

    num_airshowers_in_altitude_bins = np.zeros(
        job["binning"]["altitude_m"]["num_bins"], dtype=np.int64
    )
    num_underflow = 0
    num_overflow = 0

    views = zeros(
        keys=[
            "azimuth_deg",
            "radius_m",
            "altitude_m",
            "image_parallel_deg",
            "image_perpendicular_deg",
        ],
        binning=job["binning"],
    )
    tiews = zeros(
        keys=[
            "azimuth_deg",
            "radius_m",
            "altitude_m",
            "image_parallel_deg",
            "time_s",
        ],
        binning=job["binning"],
    )

    steering_dict = make_corsika_steering_card(
        site=job["site"],
        particle=job["particle"],
        energy=job["energy_GeV"],
        num_airshower=num_airshowers_to_be_thrown,
        random_seed=job["energy_job"],
    )

    with tempfile.TemporaryDirectory(prefix="atg_") as tmp_dir:
        corsika_o_path = os.path.join(tmp_dir, "corsika.o")
        corsika_e_path = os.path.join(tmp_dir, "corsika.e")
        particle_path = os.path.join(tmp_dir, "par.dat")

        with cpw.CorsikaPrimary(
            corsika_path=job["run_config"]["corsika_primary_path"],
            steering_dict=steering_dict,
            stdout_path=corsika_o_path,
            stderr_path=corsika_e_path,
            particle_output_path=particle_path,
        ) as corsika_run:
            for airshower in corsika_run:
                _evth, cherenkov_reader = airshower

                cherenkov_bunches = np.vstack([b for b in cherenkov_reader])

                num_bunches = cherenkov_bunches.shape[0]

                if num_bunches == 0:
                    continue

                airshower_maximum_altitude_asl_m = 1e-2 * np.median(
                    cherenkov_bunches[:, cpw.I.BUNCH.EMISSOION_ALTITUDE_ASL_CM]
                )

                (
                    underflow,
                    altitude_bin,
                    overflow,
                ) = binning_utils.find_bin_in_edges(
                    value=airshower_maximum_altitude_asl_m,
                    bin_edges=altitude_bin_edges_m,
                )

                if underflow:
                    num_underflow += 1
                    continue

                if overflow:
                    num_overflow += 1
                    continue

                if (
                    num_airshowers_in_altitude_bins[altitude_bin]
                    >= max_num_airshower_to_collect_in_altitude_bin
                ):
                    continue

                num_airshowers_in_altitude_bins[altitude_bin] += 1

                xy_tree = scipy.spatial.KDTree(
                    data=np.c_[
                        1e-2 * cherenkov_bunches[:, cpw.I.BUNCH.X_CM],
                        1e-2 * cherenkov_bunches[:, cpw.I.BUNCH.Y_CM],
                    ]
                )

                for azi in range(job["binning"]["azimuth_deg"]["num_bins"]):
                    for rad in range(job["binning"]["radius_m"]["num_bins"]):
                        num_probing_apertures = len(xy_supports[azi][rad])
                        probing_aperture_weight = 1.0 / num_probing_apertures
                        for probe in range(num_probing_apertures):
                            meets = xy_tree.query_ball_point(
                                x=xy_supports[azi][rad][probe],
                                r=job["binning"]["aperture_radius_m"],
                            )

                            surround_meets = xy_tree.query_ball_point(
                                x=xy_supports[azi][rad][probe],
                                r=job["binning"][
                                    "aperture_radius_for_timing_m"
                                ],
                            )

                            num_cherenkov_photons_in_surrounding = len(
                                surround_meets
                            )

                            if num_cherenkov_photons_in_surrounding > 0:
                                med_time_ns = np.median(
                                    cherenkov_bunches[
                                        surround_meets, cpw.I.BUNCH.TIME_NS
                                    ]
                                )

                                view = cherenkov_bunches[meets, :]

                                MOMENTUM_TO_POINTING = -1.0
                                (
                                    cer_cpara,
                                    cer_cperp,
                                ) = projection.project_light_field_onto_source_image(
                                    cer_cx_rad=MOMENTUM_TO_POINTING
                                    * view[:, cpw.I.BUNCH.UX_1],
                                    cer_cy_rad=MOMENTUM_TO_POINTING
                                    * view[:, cpw.I.BUNCH.VY_1],
                                    cer_x_m=xy_supports[azi][rad][probe][0],
                                    cer_y_m=xy_supports[azi][rad][probe][1],
                                    primary_cx_rad=0.0,
                                    primary_cy_rad=0.0,
                                    primary_core_x_m=0.0,
                                    primary_core_y_m=0.0,
                                )
                                cer_bunch_size = view[
                                    :, cpw.I.BUNCH.BUNCH_SIZE_1
                                ]

                                image = np.histogram2d(
                                    x=cer_cpara,
                                    y=cer_cperp,
                                    weights=cer_bunch_size,
                                    bins=(
                                        c_para_bin_edges_rad,
                                        c_perp_bin_edges_rad,
                                    ),
                                )[0]

                                cer_relative_time_s = (
                                    view[:, cpw.I.BUNCH.TIME_NS] - med_time_ns
                                ) * 1e-9

                                time_image = np.histogram2d(
                                    x=cer_cpara,
                                    y=cer_relative_time_s,
                                    weights=cer_bunch_size,
                                    bins=(
                                        c_para_bin_edges_rad,
                                        time_bin_edges_s,
                                    ),
                                )[0]

                                views[azi][rad][altitude_bin] += (
                                    probing_aperture_weight * image
                                )
                                tiews[azi][rad][altitude_bin] += (
                                    probing_aperture_weight * time_image
                                )

        tmp_result_path = os.path.join(tmp_dir, "result.tar")
        with open(corsika_o_path, "rb") as fin:
            corsika_o = fin.read()
        with open(corsika_e_path, "rb") as fin:
            corsika_e = fin.read()
        input_output.write_map_result(
            path=tmp_result_path,
            job=job,
            cer_azi_rad_alt_par_per=views,
            cer_azi_rad_alt_par_tim=tiews,
            corsika_o=corsika_o,
            corsika_e=corsika_e,
            num_airshowers_in_altitude_bins=num_airshowers_in_altitude_bins,
        )
        rnafwr.move(tmp_result_path, result_path)
        return 1


def reduce(work_dir):
    """
    Reducing the results of the parallel jobs.
    The many results of the jobs are reduced into the final
    look-up-table and written to the 'work_dir'.

    Parameters
    ----------
    work_dir : str : path
            The working-directory of the look-up-table.
    """
    map_dir = os.path.join(work_dir, "map")
    reduce_dir = os.path.join(work_dir, "reduce")
    sites = json_utils.read(os.path.join(work_dir, "sites.json"))
    particles = json_utils.read(os.path.join(work_dir, "particles.json"))
    binning = json_utils.read(os.path.join(work_dir, "binning.json"))

    for site_key in sites:
        for particle_key in particles:
            cer = zeros(
                keys=[
                    "energy_GeV",
                    "azimuth_deg",
                    "radius_m",
                    "altitude_m",
                    "image_parallel_deg",
                    "image_perpendicular_deg",
                ],
                binning=binning,
            )

            ter = zeros(
                keys=[
                    "energy_GeV",
                    "azimuth_deg",
                    "radius_m",
                    "altitude_m",
                    "image_parallel_deg",
                    "time_s",
                ],
                binning=binning,
            )

            num_airshowers = np.zeros(
                shape=(
                    binning["energy_GeV"]["num_bins"],
                    binning["altitude_m"]["num_bins"],
                ),
                dtype=np.int64,
            )

            print("reduce intermediate map-results")
            for energy_bin in range(binning["energy_GeV"]["num_bins"]):
                energy_key = "energy_bin_{:06d}".format(energy_bin)
                map_site_particle_energy_dir = os.path.join(
                    map_dir, site_key, particle_key, energy_key
                )

                tmpl = os.path.join(map_site_particle_energy_dir, "*.tar")
                result_paths = glob.glob(tmpl)
                result_paths.sort()
                for result_path in result_paths:
                    print(result_path)

                    result = input_output.read_map_result(result_path)
                    cer[energy_bin] += result[
                        "cherenkov.histogram.azi_rad_alt_par_per"
                    ]
                    ter[energy_bin] += result[
                        "cherenkov.histogram.azi_rad_alt_par_tim"
                    ]
                    num_airshowers[energy_bin] += result[
                        "airshower.histogram.alt"
                    ]

            # normalize
            # =========

            print("normalize light-field")
            pixel_solid_angle_sr = solid_angle_of_pixel_sr(binning=binning)
            aperture_area_m2 = area_of_aperture_m2(binning=binning)
            pixel_width_rad = parallel_pixel_width_rad(binning=binning)
            t_duration_s = time_slice_duration_s(binning=binning)

            num_para = binning["image_parallel_deg"]["num_bins"]
            num_perp = binning["image_perpendicular_deg"]["num_bins"]
            num_time = binning["time_s"]["num_bins"]

            nan_img = np.nan * np.ones(shape=(num_para, num_perp))
            nan_tmg = np.nan * np.ones(shape=(num_para, num_time))

            for ene in range(binning["energy_GeV"]["num_bins"]):
                for azi in range(binning["azimuth_deg"]["num_bins"]):
                    for rad in range(binning["radius_m"]["num_bins"]):
                        for alt in range(binning["altitude_m"]["num_bins"]):
                            num = num_airshowers[ene, alt]
                            if num > 0:
                                norm_factor_imgae = (
                                    num
                                    * aperture_area_m2
                                    * pixel_solid_angle_sr
                                )
                                norm_factor_timage = (
                                    num
                                    * aperture_area_m2
                                    * pixel_width_rad
                                    * t_duration_s
                                )
                                cer[ene, azi, rad, alt] /= norm_factor_imgae
                                ter[ene, azi, rad, alt] /= norm_factor_timage
                            else:
                                cer[ene, azi, rad, alt] = nan_img
                                ter[ene, azi, rad, alt] = nan_tmg

            print("estimate leakage")
            leakage_mask = zeros(
                keys=[
                    "energy_GeV",
                    "azimuth_deg",
                    "radius_m",
                    "altitude_m",
                ],
                binning=binning,
                dtype=np.uint8,
            )
            for ene in range(binning["energy_GeV"]["num_bins"]):
                for azi in range(binning["azimuth_deg"]["num_bins"]):
                    for rad in range(binning["radius_m"]["num_bins"]):
                        for alt in range(binning["altitude_m"]["num_bins"]):
                            leak = quality.estimate_leakage(
                                image=cer[ene, azi, rad, alt],
                                num_pixel_outer_rim=1,
                            )
                            leakage_mask[ene, azi, rad, alt] = leak > 5e-5

            print("write result")
            out = {
                "binning": binning,
                "cherenkov.density.ene_azi_rad_alt_par_per": cer,
                "cherenkov.density.ene_azi_rad_alt_par_tim": ter,
                "airshower.histogram.ene_alt": num_airshowers,
                "quality.leakage.ene_azi_rad_alt": leakage_mask,
            }
            reduce_site_particle_dir = os.path.join(
                reduce_dir, site_key, particle_key
            )
            os.makedirs(reduce_site_particle_dir, exist_ok=True)
            input_output.write_raw(
                raw_look_up=out,
                path=os.path.join(reduce_site_particle_dir, "raw.tar"),
            )
