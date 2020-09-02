import numpy as np


def find_bin_in_edges(bin_edges, value):
    upper_bin_edge = int(np.digitize([value], bin_edges)[0])
    if upper_bin_edge == 0:
        return True, 0, False
    if upper_bin_edge == bin_edges.shape[0]:
        return False, upper_bin_edge - 1, True
    return False, upper_bin_edge - 1, False


def find_bins_in_centers(bin_centers, value):
    underflow, lower_bin, overflow = find_bin_in_edges(
        bin_edges=bin_centers, value=value
    )

    upper_bin = lower_bin + 1
    if underflow:
        lower_weight = 0.0
    elif overflow:
        lower_weight = 1.0
    else:
        dist_to_lower = value - bin_centers[lower_bin]
        dist_to_upper = bin_centers[upper_bin] - value
        bin_range = bin_centers[upper_bin] - bin_centers[lower_bin]
        lower_weight = 1 - dist_to_lower / bin_range

    return {
        "underflow": underflow,
        "overflow": overflow,
        "lower_bin": lower_bin,
        "upper_bin": lower_bin + 1,
        "lower_weight": lower_weight,
        "upper_weight": 1.0 - lower_weight,
    }


def bin_centers(bin_edges, weight_lower_edge=0.5):
    assert weight_lower_edge >= 0.0 and weight_lower_edge <= 1.0
    weight_upper_edge = 1.0 - weight_lower_edge
    return (
        weight_lower_edge * bin_edges[:-1] + weight_upper_edge * bin_edges[1:]
    )


def make_explicit_binning(binning):
    _b = binning
    out = {}
    for key in _b:
        if _b[key].__class__ == dict:
            if _b[key]["space"] == "lin":
                space = np.linspace
            elif _b[key]["space"] == "geom":
                space = np.geomspace
            out[key] = {}
            out[key]["edges"] = space(
                _b[key]["start"], _b[key]["stop"], _b[key]["num_bins"] + 1,
            )
            out[key]["centers"] = bin_centers(bin_edges=out[key]["edges"])
        else:
            out[key] = binning[key]
    return out


def xy_supports_on_observationlevel(binning):
    _b = binning
    _eb = make_explicit_binning(binning=_b)

    xy_supports = np.zeros(
        shape=(_b["azimuth_deg"]["num_bins"], _b["radius_m"]["num_bins"], 2)
    )
    radius_m_supports = _eb["radius_m"]["centers"]
    azimuth_deg_supports = _eb["azimuth_deg"]["centers"]
    for azi, a_deg in enumerate(azimuth_deg_supports):
        for rad, r_m in enumerate(radius_m_supports):
            xy_supports[azi][rad][0] = np.cos(np.deg2rad(a_deg)) * r_m
            xy_supports[azi][rad][1] = np.sin(np.deg2rad(a_deg)) * r_m
    return xy_supports


def find_bins(explicit_binning, energy, altitude, azimuth, radius):
    _eb = explicit_binning

    ene = find_bins_in_centers(
        bin_centers=_eb["energy_GeV"]["centers"], value=energy
    )
    if ene["overflow"] or ene["underflow"]:
        raise IndexError("energy out of range")

    alt = find_bins_in_centers(
        bin_centers=_eb["altitude_m"]["centers"], value=altitude
    )
    if alt["overflow"] or alt["underflow"]:
        raise IndexError("altitude out of range")

    azi = find_bins_in_centers(
        bin_centers=_eb["azimuth_deg"]["centers"], value=azimuth
    )
    if azi["overflow"] or azi["underflow"]:
        raise IndexError("azimuth out of range")

    rad = find_bins_in_centers(
        bin_centers=_eb["radius_m"]["centers"], value=radius
    )
    if rad["overflow"] or rad["underflow"]:
        raise IndexError("radius out of range")

    return {
        "energy_bins": [
            {"bin": ene["lower_bin"], "weight": ene["lower_weight"]},
            {"bin": ene["upper_bin"], "weight": ene["upper_weight"]},
        ],
        "altitude_bins": [
            {"bin": alt["lower_bin"], "weight": alt["lower_weight"]},
            {"bin": alt["upper_bin"], "weight": alt["upper_weight"]},
        ],
        "azimuth_bins": [
            {"bin": azi["lower_bin"], "weight": azi["lower_weight"]},
            {"bin": azi["upper_bin"], "weight": azi["upper_weight"]},
        ],
        "radius_bins": [
            {"bin": rad["lower_bin"], "weight": rad["lower_weight"]},
            {"bin": rad["upper_bin"], "weight": rad["upper_weight"]},
        ],
    }
