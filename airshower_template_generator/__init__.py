from . import examples
from . import bins
from . import query
from . import plot

import numpy as np
import corsika_primary_wrapper as cpw
import tempfile
import os
import io
import scipy
import json
from plenoirf import json_numpy
import glob

from queue_map_reduce import network_file_system as nfs
from scipy import spatial
import tarfile
import gzip


def init(
    work_dir,
    sites=examples.SITES,
    particles=examples.PARTICLES,
    binning=examples.BINNING,
    run_config=examples.RUN_CONFIG,
):
    os.makedirs(work_dir, exist_ok=True)
    json_numpy.write(path=os.path.join(work_dir, "sites.json"), out_dict=sites)
    json_numpy.write(
        path=os.path.join(work_dir, "particles.json"), out_dict=particles
    )
    json_numpy.write(
        path=os.path.join(work_dir, "binning.json"), out_dict=binning
    )
    json_numpy.write(
        path=os.path.join(work_dir, "run_config.json"), out_dict=run_config
    )


def make_jobs(work_dir):
    map_dir = os.path.join(work_dir, "map")

    sites = json_numpy.read(os.path.join(work_dir, "sites.json"))
    particles = json_numpy.read(os.path.join(work_dir, "particles.json"))
    binning = json_numpy.read(os.path.join(work_dir, "binning.json"))
    run_config = json_numpy.read(os.path.join(work_dir, "run_config.json"))

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
    return pixel_edge_rad ** 2


def time_slice_duration_s(binning):
    tim = binning["time_s"]
    duration_s = (tim["stop_edge"] - tim["start_edge"]) / tim["num_bins"]
    return duration_s


def make_corsika_steering_card(
    site, particle, energy, num_airshower, random_seed
):
    run_id = random_seed + 1
    assert run_id > 0
    steering = {
        "run": {
            "run_id": run_id,
            "event_id_of_first_event": 1,
            "observation_level_asl_m": site["observation_level_asl_m"],
            "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
            "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
            "atmosphere_id": site["atmosphere_id"],
        },
        "primaries": [],
    }

    for i in range(num_airshower):
        primary = {
            "particle_id": particle["particle_id"],
            "energy_GeV": energy,
            "zenith_rad": 0.0,
            "azimuth_rad": 0.0,
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw.simple_seed(i + run_id),
        }
        steering["primaries"].append(primary)
    return steering


def _project_to_image(
    cxs,
    cys,
    relative_time,
    bunch_size,
    c_para_bin_edges,
    c_perp_bin_edges,
    time_bin_edges,
    aperture_x,
    aperture_y,
):
    azimuth = np.arctan2(aperture_y, aperture_x)
    c_para = np.cos(-azimuth) * cxs - np.sin(-azimuth) * cys
    c_perp = np.sin(-azimuth) * cxs + np.cos(-azimuth) * cys
    image = np.histogram2d(
        x=c_para,
        y=c_perp,
        weights=bunch_size,
        bins=(c_para_bin_edges, c_perp_bin_edges),
    )[0]

    time_image = np.histogram2d(
        x=c_para,
        y=relative_time,
        weights=bunch_size,
        bins=(c_para_bin_edges, time_bin_edges),
    )[0]

    return image, time_image


def image_pixels_are_square(binning):
    para = binning["image_parallel_deg"]
    density_para = para["num_bins"] / (para["stop_edge"] - para["start_edge"])
    perp = binning["image_perpendicular_deg"]
    density_perp = perp["num_bins"] / (perp["stop_edge"] - perp["start_edge"])
    return np.abs(density_para / density_perp - 1.0) < 1e-6


def append_tar(tar_obj, name, payload_bytes):
    with io.BytesIO() as f:
        f.write(payload_bytes)
        f.seek(0)
        tarinfo = tarfile.TarInfo(name=name)
        tarinfo.size = len(payload_bytes)
        tar_obj.addfile(tarinfo=tarinfo, fileobj=f)


def zeros(binning, keys=[], dtype=np.float32):
    return np.zeros(
        shape=[binning[key]["num_bins"] for key in keys], dtype=dtype,
    )


def run_job(job):
    os.makedirs(job["map_dir"], exist_ok=True)
    result_path = os.path.join(
        job["map_dir"], "{:06d}.tar".format(job["energy_job"])
    )
    explbin = bins.make_explicit_binning(binning=job["binning"])
    c_para_bin_edges = np.deg2rad(explbin["image_parallel_deg"]["edges"])
    c_perp_bin_edges = np.deg2rad(explbin["image_perpendicular_deg"]["edges"])
    time_bin_edges = explbin["time_s"]["edges"]
    altitude_bin_edges_m = explbin["altitude_m"]["edges"]

    xy_supports = bins.xy_supports_on_observationlevel(binning=job["binning"])

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

        corsika_run = cpw.CorsikaPrimary(
            corsika_path=job["run_config"]["corsika_primary_path"],
            steering_dict=steering_dict,
            stdout_path=corsika_o_path,
            stderr_path=corsika_e_path,
        )

        for airshower in corsika_run:
            _, cherenkov_bunches = airshower

            num_bunches = cherenkov_bunches.shape[0]

            if num_bunches == 0:
                continue

            airshower_maximum_altitude_asl_m = 1e-2 * np.median(
                cherenkov_bunches[:, cpw.IZEM]
            )

            underflow, altitude_bin, overflow = bins.find_bin_in_edges(
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
                    1e-2 * cherenkov_bunches[:, cpw.IX],
                    1e-2 * cherenkov_bunches[:, cpw.IY],
                ]
            )

            for azi in range(job["binning"]["azimuth_deg"]["num_bins"]):
                meets = xy_tree.query_ball_point(
                    x=xy_supports[azi], r=job["binning"]["aperture_radius_m"]
                )

                surround_meets = xy_tree.query_ball_point(
                    x=xy_supports[azi],
                    r=job["binning"]["aperture_radius_for_timing_m"],
                )

                for rad in range(job["binning"]["radius_m"]["num_bins"]):

                    num_cherenkov_photons_in_surrounding = len(
                        surround_meets[rad]
                    )

                    if num_cherenkov_photons_in_surrounding > 0:

                        med_time_ns = np.median(
                            cherenkov_bunches[surround_meets[rad], cpw.ITIME]
                        )

                        view = cherenkov_bunches[meets[rad], :]
                        img, timg = _project_to_image(
                            cxs=view[:, cpw.ICX],
                            cys=view[:, cpw.ICY],
                            relative_time=(view[:, cpw.ITIME] - med_time_ns)
                            * 1e-9,
                            bunch_size=view[:, cpw.IBSIZE],
                            c_para_bin_edges=c_para_bin_edges,
                            c_perp_bin_edges=c_perp_bin_edges,
                            time_bin_edges=time_bin_edges,
                            aperture_x=xy_supports[azi][rad][0],
                            aperture_y=xy_supports[azi][rad][1],
                        )
                        views[azi][rad][altitude_bin] += img
                        tiews[azi][rad][altitude_bin] += timg

        tmp_result_path = os.path.join(tmp_dir, "result.tar")
        with tarfile.TarFile(tmp_result_path, "w") as tar_obj:
            append_tar(
                tar_obj=tar_obj,
                name="job.json",
                payload_bytes=json.dumps(
                    job, cls=json_numpy.Encoder, indent=4
                ).encode(encoding="ascii"),
            )
            append_tar(
                tar_obj=tar_obj,
                name="cherenkov.histogram.azi_rad_alt_par_per.order-c.float32.gz",
                payload_bytes=gzip.compress(data=views.tobytes(order="c")),
            )
            append_tar(
                tar_obj=tar_obj,
                name="cherenkov.histogram.azi_rad_alt_par_tim.order-c.float32.gz",
                payload_bytes=gzip.compress(data=tiews.tobytes(order="c")),
            )
            append_tar(
                tar_obj=tar_obj,
                name="airshower.histogram.alt.int64.gz",
                payload_bytes=gzip.compress(
                    data=num_airshowers_in_altitude_bins.tobytes(order="c"),
                ),
            )
            with open(corsika_o_path, "rb") as fin:
                append_tar(
                    tar_obj=tar_obj,
                    name="corsika.o.gz",
                    payload_bytes=gzip.compress(data=fin.read()),
                )
            with open(corsika_e_path, "rb") as fin:
                append_tar(
                    tar_obj=tar_obj,
                    name="corsika.e.gz",
                    payload_bytes=gzip.compress(data=fin.read()),
                )

        nfs.move(tmp_result_path, result_path)

        return 1


def _tar_read_and_reshape(tar_obj, name, shape, dtype=np.float32, order="c"):
    tinfo = tar_obj.next()
    assert tinfo.name == name
    raw = gzip.decompress(tar_obj.extractfile(tinfo).read())
    arr = np.frombuffer(raw, dtype=dtype)
    arr = arr.reshape(shape, order=order)
    return arr


def read_map_result(path):
    out = {}
    with tarfile.TarFile(path, "r") as tar_obj:

        tinfo = tar_obj.next()
        assert tinfo.name == "job.json"
        out["job"] = json.loads(tar_obj.extractfile(tinfo).read())
        _b = out["job"]["binning"]

        out["cherenkov.histogram.azi_rad_alt_par_per"] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="cherenkov.histogram.azi_rad_alt_par_per.order-c.float32.gz",
            shape=(
                _b["azimuth_deg"]["num_bins"],
                _b["radius_m"]["num_bins"],
                _b["altitude_m"]["num_bins"],
                _b["image_parallel_deg"]["num_bins"],
                _b["image_perpendicular_deg"]["num_bins"],
            ),
        )

        out["cherenkov.histogram.azi_rad_alt_par_tim"] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="cherenkov.histogram.azi_rad_alt_par_tim.order-c.float32.gz",
            shape=(
                _b["azimuth_deg"]["num_bins"],
                _b["radius_m"]["num_bins"],
                _b["altitude_m"]["num_bins"],
                _b["image_parallel_deg"]["num_bins"],
                _b["time_s"]["num_bins"],
            ),
        )

        out["airshower.histogram.alt"] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="airshower.histogram.alt.int64.gz",
            shape=(_b["altitude_m"]["num_bins"]),
            dtype=np.int64,
        )
    return out


def reduce(work_dir):
    map_dir = os.path.join(work_dir, "map")
    reduce_dir = os.path.join(work_dir, "reduce")
    sites = json_numpy.read(os.path.join(work_dir, "sites.json"))
    particles = json_numpy.read(os.path.join(work_dir, "particles.json"))
    binning = json_numpy.read(os.path.join(work_dir, "binning.json"))

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

            for energy_bin in range(binning["energy_GeV"]["num_bins"]):
                energy_key = "energy_bin_{:06d}".format(energy_bin)
                map_site_particle_energy_dir = os.path.join(
                    map_dir, site_key, particle_key, energy_key
                )

                tmpl = os.path.join(map_site_particle_energy_dir, "*.tar")
                for result_path in glob.glob(tmpl):
                    print(result_path)

                    result = read_map_result(result_path)
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

            out = {
                "binning": binning,
                "cherenkov.density.ene_azi_rad_alt_par_per": cer,
                "cherenkov.density.ene_azi_rad_alt_par_tim": ter,
                "airshower.histogram.ene_alt": num_airshowers,
            }
            reduce_site_particle_dir = os.path.join(
                reduce_dir, site_key, particle_key
            )
            os.makedirs(reduce_site_particle_dir, exist_ok=True)
            write_raw(out, os.path.join(reduce_site_particle_dir, "raw.tar"))


def write_raw(raw_look_up, path):
    _b = raw_look_up["binning"]

    cer = raw_look_up["cherenkov.density.ene_azi_rad_alt_par_per"]
    assert cer.dtype == np.float32
    assert cer.shape[0] == _b["energy_GeV"]["num_bins"]
    assert cer.shape[1] == _b["azimuth_deg"]["num_bins"]
    assert cer.shape[2] == _b["radius_m"]["num_bins"]
    assert cer.shape[3] == _b["altitude_m"]["num_bins"]
    assert cer.shape[4] == _b["image_parallel_deg"]["num_bins"]
    assert cer.shape[5] == _b["image_perpendicular_deg"]["num_bins"]

    ter = raw_look_up["cherenkov.density.ene_azi_rad_alt_par_tim"]
    assert ter.dtype == np.float32
    assert ter.shape[0] == _b["energy_GeV"]["num_bins"]
    assert ter.shape[1] == _b["azimuth_deg"]["num_bins"]
    assert ter.shape[2] == _b["radius_m"]["num_bins"]
    assert ter.shape[3] == _b["altitude_m"]["num_bins"]
    assert ter.shape[4] == _b["image_parallel_deg"]["num_bins"]
    assert ter.shape[5] == _b["time_s"]["num_bins"]

    num = raw_look_up["airshower.histogram.ene_alt"]
    assert num.dtype == np.int64
    assert num.shape[0] == _b["energy_GeV"]["num_bins"]
    assert num.shape[1] == _b["altitude_m"]["num_bins"]

    with tempfile.TemporaryDirectory(prefix="atg_") as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "raw_look_up.tar")
        with tarfile.TarFile(tmp_path, "w") as tar_obj:
            append_tar(
                tar_obj=tar_obj,
                name="binning.json",
                payload_bytes=json.dumps(
                    raw_look_up["binning"], indent=4
                ).encode(encoding="ascii"),
            )
            append_tar(
                tar_obj=tar_obj,
                name="cherenkov.density.ene_azi_rad_alt_par_per.order-c.float32.gz",
                payload_bytes=gzip.compress(data=cer.tobytes(order="c")),
            )
            append_tar(
                tar_obj=tar_obj,
                name="cherenkov.density.ene_azi_rad_alt_par_tim.order-c.float32.gz",
                payload_bytes=gzip.compress(data=ter.tobytes(order="c")),
            )
            append_tar(
                tar_obj=tar_obj,
                name="airshower.histogram.ene_alt.int64.gz",
                payload_bytes=gzip.compress(data=num.tobytes(order="c"),),
            )
        nfs.move(src=tmp_path, dst=path)


def read_raw(path):
    out = {}
    with tarfile.TarFile(path, "r") as tar_obj:
        tinfo = tar_obj.next()
        assert tinfo.name == "binning.json"
        out["binning"] = json.loads(tar_obj.extractfile(tinfo).read())
        _b = out["binning"]

        out[
            "cherenkov.density.ene_azi_rad_alt_par_per"
        ] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="cherenkov.density.ene_azi_rad_alt_par_per.order-c.float32.gz",
            shape=(
                _b["energy_GeV"]["num_bins"],
                _b["azimuth_deg"]["num_bins"],
                _b["radius_m"]["num_bins"],
                _b["altitude_m"]["num_bins"],
                _b["image_parallel_deg"]["num_bins"],
                _b["image_perpendicular_deg"]["num_bins"],
            ),
        )

        out[
            "cherenkov.density.ene_azi_rad_alt_par_tim"
        ] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="cherenkov.density.ene_azi_rad_alt_par_tim.order-c.float32.gz",
            shape=(
                _b["energy_GeV"]["num_bins"],
                _b["azimuth_deg"]["num_bins"],
                _b["radius_m"]["num_bins"],
                _b["altitude_m"]["num_bins"],
                _b["image_parallel_deg"]["num_bins"],
                _b["time_s"]["num_bins"],
            ),
        )

        out["airshower.histogram.ene_alt"] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="airshower.histogram.ene_alt.int64.gz",
            shape=(_b["energy_GeV"]["num_bins"], _b["altitude_m"]["num_bins"]),
            dtype=np.int64,
        )
    out["explicit_binning"] = bins.make_explicit_binning(out["binning"])
    return out
