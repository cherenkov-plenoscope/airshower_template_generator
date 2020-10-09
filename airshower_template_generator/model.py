import numpy as np
import json
from . import query
from . import bins

SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def gaussian_bell_1d(c_deg, peak_deg, width_deg):
    return (
        1.0
        / (width_deg * SQRT_TWO_PI)
        * np.exp(-0.5 * (c_deg - peak_deg) ** 2 / width_deg ** 2)
    )


def lorentz_transversal(c_deg, peak_deg, width_deg):
    return width_deg / (np.pi * (width_deg ** 2 + (c_deg - peak_deg) ** 2))


def lorentz_moyal_longitidinal(
    c_deg, peak_deg, width_deg,
):
    lam = (c_deg - peak_deg) / width_deg
    return (
        1.0 / (width_deg * SQRT_TWO_PI) * np.exp(-0.5 * (lam + np.exp(-lam)))
    )


def my_moyal_model(c_deg, moyal_peak_deg, tail_direction, width_deg):
    moyal_density = (1.0 - tail_direction) * lorentz_moyal_longitidinal(
        c_deg=c_deg, peak_deg=moyal_peak_deg, width_deg=width_deg,
    )
    gaussian_density = tail_direction * lorentz_moyal_longitidinal(
        c_deg=-c_deg, peak_deg=-moyal_peak_deg, width_deg=width_deg,
    )
    return moyal_density + gaussian_density


def read_bell_model_lut(path):
    with open(path, "rt") as fin:
        dd = json.loads(fin.read())
    lut = dd
    lut["explicit_binning"] = bins.make_explicit_binning(lut["binning"])
    lut["bell_par.ene_azi_rad_alt"] = np.array(lut["bell_par.ene_azi_rad_alt"])
    lut["bell_per.ene_azi_rad_alt"] = np.array(lut["bell_per.ene_azi_rad_alt"])
    lut["population.ene_alt"] = np.array(lut["population.ene_alt"])
    lut["max_rad_at.ene_azi_alt"] = np.array(lut["max_rad_at.ene_azi_alt"])
    return lut


def query_bell_model(
    bell_model_lut, energy_GeV, azimuth_deg, radius_m, altitude_m
):
    lut = bell_model_lut
    b = bins.find_bins(
        explicit_binning=lut["explicit_binning"],
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )

    avg_par = np.zeros(2, dtype=np.float32)
    avg_per = np.zeros(2, dtype=np.float32)

    lutpar = lut["bell_par.ene_azi_rad_alt"]
    lutper = lut["bell_per.ene_azi_rad_alt"]

    weights = []
    for ene in b["energy_GeV"]:
        for alt in b["altitude_m"]:

            assert lut["population.ene_alt"][ene["bin"], alt["bin"]]

            for azi in b["azimuth_deg"]:
                for rad in b["radius_m"]:

                    assert (
                        rad["bin"]
                        <= lut["max_rad_at.ene_azi_alt"][
                            ene["bin"], azi["bin"], alt["bin"]
                        ]
                    )

                    slice_par = lutpar[
                        ene["bin"], azi["bin"], rad["bin"], alt["bin"]
                    ]
                    slice_per = lutper[
                        ene["bin"], azi["bin"], rad["bin"], alt["bin"]
                    ]

                    weight = (
                        ene["weight"]
                        * alt["weight"]
                        * azi["weight"]
                        * rad["weight"]
                    )
                    weights.append(weight)
                    avg_par = avg_par + slice_par * weight
                    avg_per = avg_per + slice_per * weight
    sum_weights = np.sum(weights)
    return (avg_par / sum_weights, avg_per / sum_weights)
