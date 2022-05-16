import numpy as np
import binning_utils


def make_explicit_binning(binning):
    _b = binning
    out = {}
    out["energy_GeV"] = {}
    out["energy_GeV"]["supports"] = np.geomspace(
        _b["energy_GeV"]["start_support"],
        _b["energy_GeV"]["stop_support"],
        _b["energy_GeV"]["num_bins"],
    )
    out["energy_GeV"]["log10_supports"] = np.log10(
        out["energy_GeV"]["supports"]
    )

    out["azimuth_deg"] = {}
    out["azimuth_deg"]["edges"] = np.linspace(
        0.0, 360.0, _b["azimuth_deg"]["num_bins"] + 1
    )
    out["azimuth_deg"]["supports"] = binning_utils.centers(
        bin_edges=out["azimuth_deg"]["edges"]
    )

    out["radius_m"] = {}
    out["radius_m"]["supports"] = np.linspace(
        _b["radius_m"]["start_support"],
        _b["radius_m"]["stop_support"],
        _b["radius_m"]["num_bins"],
    )

    for key in [
        "altitude_m",
        "image_parallel_deg",
        "image_perpendicular_deg",
        "time_s",
    ]:
        out[key] = {}
        out[key]["edges"] = np.linspace(
            _b[key]["start_edge"],
            _b[key]["stop_edge"],
            _b[key]["num_bins"] + 1,
        )
        out[key]["supports"] = binning_utils.centers(
            bin_edges=out[key]["edges"]
        )

    return out


def full_coverage_xy_supports_on_observationlevel(binning):
    """
    1dim) azimuth
    2dim) radius
    3dim) probes on azimuth-arc
    """
    eb = make_explicit_binning(binning=binning)
    probing_aperute_diameter_m = 2.0 * binning["aperture_radius_m"]

    xy_supports = []
    for azi in range(binning["azimuth_deg"]["num_bins"]):
        xy_supports.append([])
        for rad in range(binning["radius_m"]["num_bins"]):
            xy_supports[azi].append([])

    azi_bin_width_deg = 360.0 / binning["azimuth_deg"]["num_bins"]

    for azi, azi_center_deg in enumerate(eb["azimuth_deg"]["supports"]):
        for rad, r_m in enumerate(eb["radius_m"]["supports"]):

            azi_circumference_m = 2.0 * np.pi * r_m
            azi_arc_length_m = (
                azi_circumference_m / binning["azimuth_deg"]["num_bins"]
            )

            num_probing_apertures_on_arc = (
                azi_arc_length_m // probing_aperute_diameter_m
            )
            if num_probing_apertures_on_arc == 0:
                num_probing_apertures_on_arc += 1

            arc_azimuths_deg = np.linspace(
                azi_center_deg - 0.5 * azi_bin_width_deg,
                azi_center_deg + 0.5 * azi_bin_width_deg,
                num_probing_apertures_on_arc,
            )

            for azi_deg in arc_azimuths_deg:
                _x = np.cos(np.deg2rad(azi_deg)) * r_m
                _y = np.sin(np.deg2rad(azi_deg)) * r_m
                xy_supports[azi][rad].append([_x, _y])

    for azi, azi_center_deg in enumerate(eb["azimuth_deg"]["supports"]):
        for rad, r_m in enumerate(eb["radius_m"]["supports"]):
            xy_supports[azi][rad] = np.array(xy_supports[azi][rad])

    return xy_supports


def find_energy_bins(explicit_binning, energy_GeV):
    ene = binning_utils.find_bins_in_centers(
        bin_centers=explicit_binning["energy_GeV"]["log10_supports"],
        value=np.log10(energy_GeV),
    )
    if ene["overflow"] or ene["underflow"]:
        raise IndexError("energy {:.3e}GeV out of range".format(energy_GeV))
    return ene


def find_azimuth_bins(explicit_binning, azimuth_deg):
    azi = binning_utils.find_bins_in_centers(
        bin_centers=explicit_binning["azimuth_deg"]["edges"],
        value=modulo_azimuth_range(azimuth_deg=azimuth_deg),
    )
    if azi["overflow"] or azi["underflow"]:
        raise IndexError("azimuth {:.3e}deg out of range".format(azimuth_deg))
    num_azimuths_supports = len(explicit_binning["azimuth_deg"]["supports"])
    if azi["upper_bin"] == num_azimuths_supports:
        azi["upper_bin"] = 0
    return azi


def find_altitude_bins(explicit_binning, altitude_m):
    alt = binning_utils.find_bins_in_centers(
        bin_centers=explicit_binning["altitude_m"]["supports"],
        value=altitude_m,
    )
    if alt["overflow"] or alt["underflow"]:
        raise IndexError("altitude {:.3e}m out of range".format(altitude_m))
    return alt


def find_radius_bins(explicit_binning, radius_m):
    rad = binning_utils.find_bins_in_centers(
        bin_centers=explicit_binning["radius_m"]["supports"], value=radius_m
    )
    if rad["overflow"] or rad["underflow"]:
        raise IndexError("radius {:.3e}m out of range".format(radius_m))
    return rad


def find_bins(explicit_binning, energy_GeV, altitude_m, azimuth_deg, radius_m):
    ene = find_energy_bins(explicit_binning, energy_GeV)
    azi = find_azimuth_bins(explicit_binning, azimuth_deg)
    alt = find_altitude_bins(explicit_binning, altitude_m)
    rad = find_radius_bins(explicit_binning, radius_m)

    return {
        "energy_GeV": [
            {"bin": ene["lower_bin"], "weight": ene["lower_weight"]},
            {"bin": ene["upper_bin"], "weight": ene["upper_weight"]},
        ],
        "altitude_m": [
            {"bin": alt["lower_bin"], "weight": alt["lower_weight"]},
            {"bin": alt["upper_bin"], "weight": alt["upper_weight"]},
        ],
        "azimuth_deg": [
            {"bin": azi["lower_bin"], "weight": azi["lower_weight"]},
            {"bin": azi["upper_bin"], "weight": azi["upper_weight"]},
        ],
        "radius_m": [
            {"bin": rad["lower_bin"], "weight": rad["lower_weight"]},
            {"bin": rad["upper_bin"], "weight": rad["upper_weight"]},
        ],
    }


def modulo_azimuth_range(azimuth_deg):
    start_deg = 0.0
    stop_deg = 360.0
    out_azimuth_deg = float(azimuth_deg)

    while out_azimuth_deg < start_deg:
        out_azimuth_deg += 360.0

    while out_azimuth_deg >= stop_deg:
        out_azimuth_deg -= 360.0

    return out_azimuth_deg
