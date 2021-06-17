import tarfile
import gzip
import io
import tempfile
import os
import numpy as np
from queue_map_reduce import network_file_system as nfs
import json
from . import bins


def _tar_append(tar_obj, name, payload_bytes):
    with io.BytesIO() as f:
        f.write(payload_bytes)
        f.seek(0)
        tarinfo = tarfile.TarInfo(name=name)
        tarinfo.size = len(payload_bytes)
        tar_obj.addfile(tarinfo=tarinfo, fileobj=f)


def _tar_read_and_reshape(tar_obj, name, shape, dtype=np.float32, order="c"):
    tinfo = tar_obj.next()
    assert tinfo.name == name
    raw = gzip.decompress(tar_obj.extractfile(tinfo).read())
    arr = np.frombuffer(raw, dtype=dtype)
    arr = arr.reshape(shape, order=order)
    return arr


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

    q_leakage = raw_look_up["quality.leakage.ene_azi_rad_alt"]
    assert q_leakage.dtype == np.uint8
    assert q_leakage.shape[0] == _b["energy_GeV"]["num_bins"]
    assert q_leakage.shape[1] == _b["azimuth_deg"]["num_bins"]
    assert q_leakage.shape[2] == _b["radius_m"]["num_bins"]
    assert q_leakage.shape[3] == _b["altitude_m"]["num_bins"]

    with tempfile.TemporaryDirectory(prefix="atg_") as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "raw_look_up.tar")
        with tarfile.TarFile(tmp_path, "w") as tar_obj:
            _tar_append(
                tar_obj=tar_obj,
                name="binning.json",
                payload_bytes=json.dumps(
                    raw_look_up["binning"], indent=4
                ).encode(encoding="ascii"),
            )
            _tar_append(
                tar_obj=tar_obj,
                name="cherenkov.density.ene_azi_rad_alt_par_per.order-c.float32.gz",
                payload_bytes=gzip.compress(data=cer.tobytes(order="c")),
            )
            _tar_append(
                tar_obj=tar_obj,
                name="cherenkov.density.ene_azi_rad_alt_par_tim.order-c.float32.gz",
                payload_bytes=gzip.compress(data=ter.tobytes(order="c")),
            )
            _tar_append(
                tar_obj=tar_obj,
                name="airshower.histogram.ene_alt.int64.gz",
                payload_bytes=gzip.compress(data=num.tobytes(order="c"),),
            )
            _tar_append(
                tar_obj=tar_obj,
                name="quality.leakage.ene_azi_rad_alt.uint8.gz",
                payload_bytes=gzip.compress(
                    data=q_leakage.tobytes(order="c"),
                ),
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

        out["quality.leakage.ene_azi_rad_alt"] = _tar_read_and_reshape(
            tar_obj=tar_obj,
            name="quality.leakage.ene_azi_rad_alt.uint8.gz",
            shape=(
                _b["energy_GeV"]["num_bins"],
                _b["azimuth_deg"]["num_bins"],
                _b["radius_m"]["num_bins"],
                _b["altitude_m"]["num_bins"],
            ),
            dtype=np.uint8,
        )

    out["explicit_binning"] = bins.make_explicit_binning(out["binning"])
    return out


def write_map_result(
    path,
    job,
    cer_azi_rad_alt_par_per,
    cer_azi_rad_alt_par_tim,
    corsika_o,
    corsika_e,
    num_airshowers_in_altitude_bins,
):
    tmp_path = path + ".tmp"
    with tarfile.TarFile(tmp_path, "w") as tar_obj:
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
            payload_bytes=gzip.compress(
                data=cer_azi_rad_alt_par_per.tobytes(order="c")
            ),
        )
        append_tar(
            tar_obj=tar_obj,
            name="cherenkov.histogram.azi_rad_alt_par_tim.order-c.float32.gz",
            payload_bytes=gzip.compress(
                data=cer_azi_rad_alt_par_tim.tobytes(order="c")
            ),
        )
        append_tar(
            tar_obj=tar_obj,
            name="airshower.histogram.alt.int64.gz",
            payload_bytes=gzip.compress(
                data=num_airshowers_in_altitude_bins.tobytes(order="c"),
            ),
        )
        append_tar(
            tar_obj=tar_obj,
            name="corsika.o.gz",
            payload_bytes=gzip.compress(data=corsika_o),
        )
        append_tar(
            tar_obj=tar_obj,
            name="corsika.e.gz",
            payload_bytes=gzip.compress(data=corsika_e),
        )

    nfs.move(tmp_path, path)


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
