import numpy as np
import json
from . import query
from . import bins
from . import projection

SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def gaussian_bell_1d(c_deg, peak_deg, width_deg):
    return (
        gaussian_bell_1d_max_one(c_deg, peak_deg, width_deg) /
        (width_deg * SQRT_TWO_PI)
    )

def gaussian_bell_1d_max_one(c_deg, peak_deg, width_deg):
    return np.exp(-0.5 * (c_deg - peak_deg) ** 2 / width_deg ** 2)


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



class DiscPotential:
    def __init__(self, r0, r1):
        assert r0 > 0.0
        assert r1 > r0
        self.r0 = r0
        self.r1 = r1

    def potential(self, r):
        if r > self.r0:
            return ((r - self.r0) / (self.r1 - self.r0)) ** 2
        else:
            return 0.0

    def activation(self, r):
        if r <= self.r0:
            return 0.0
        elif r > self.r1:
            return 1.0
        else:
            gap = self.r1 - self.r0
            return (r - self.r0) / gap


class LightField:
    def __init__(
        self,
        cx,
        cy,
        x,
        y,
        time_slice,
    ):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.time_slice = time_slice


class BellLightFieldFitter:
    def __init__(
        self,
        bell_model_lut,
        light_field,
        energy_GeV,
        altitude_m,
    ):
        self.bell_model_lut = bell_model_lut
        self.light_field = light_field
        self.energy_GeV = energy_GeV
        self.altitude_m = altitude_m

        eb = self.bell_model_lut["explicit_binning"]
        ene = bins.find_bins_in_centers(
            bin_centers=eb["energy_GeV"]["supports"],
            value=energy_GeV
        )
        assert ana["underflow"] == False and ene["overflow"] == False
        self.ene = ene

        alt = bins.find_bins_in_centers(
            bin_centers=eb["altitude_m"]["supports"],
            value=altitude_m
        )
        assert alt["underflow"] == False and alt["overflow"] == False
        self.alt = alt

        assert self.bell_model_lut["population.ene_alt"][self.ene, self.alt]


    def max_core_radius(self, azimuth_deg):

        bb = bins.find_bins(
            explicit_binning=self.bell_model_lut["explicit_binning"],
            energy_GeV=self.energy_GeV,
            altitude_m=self.altitude_m,
            azimuth_deg=azimuth_deg,
            radius_m=0.0,
        )

        min_rad = self.bell_model_lut["binning"]["radius_m"]["num_bins"]
        for bene in bb["energy_GeV"]:
            for balt in bb["altitude_m"]:
                for bazi in bb["azimuth_deg"]:
                    _max_rad = self.bell_model_lut["max_rad_at.ene_azi_alt"][
                        bene["bin"], bazi["bin"], balt["bin"]
                    ]
                    if _max_rad < min_rad:
                        min_rad = _max_rad

        return self.bell_model_lut["explicit_binning"]["radius_m"]["supports"][min_rad]


    def model(self, source_cx, source_cy, core_x, core_y):
        num_ph = self.light_field.x.shape[0]

        ph_core_x = core_x - self.light_field.x
        ph_core_y = core_y - self.light_field.y

        ph_core_azimuth_rad = np.arctan2(y=ph_core_x, x=ph_core_x)
        ph_core_azimuth_deg = np.rad2deg(ph_core_azimuth_rad)
        ph_core_radius = np.hypot(ph_core_x, ph_core_y)

        ph_bell_par = []
        ph_bell_per = []
        for ph in range(num_ph):
            par, per = query_bell_model(
                bell_model_lut=self.bell_model_lut,
                energy_GeV=self.energy_GeV,
                azimuth_deg=ph_core_azimuth_deg[ph],
                radius_m=ph_core_radius[ph],
                altitude_m=self.altitude_m
            )
            ph_bell_par.append(par)
            ph_bell_per.append(per)
        ph_bell_par = np.array(ph_bell_par)
        ph_bell_per = np.array(ph_bell_per)

        WRT_DOWNWARDS = -1.0
        c_para, c_perp = projection.project_light_field_onto_source_image(
            cer_cx_rad=WRT_DOWNWARDS * self.light_field.cx,
            cer_cy_rad=WRT_DOWNWARDS * self.light_field.cy,
            cer_x_m=self.light_field.x,
            cer_y_m=self.light_field.y,
            primary_cx_rad=WRT_DOWNWARDS * source_cx,
            primary_cy_rad=WRT_DOWNWARDS * source_cy,
            primary_core_x_m=core_x,
            primary_core_y_m=core_y,
        )

        par_weights = []
        per_weights = []
        for ph in range(num_ph):
            par_w = gaussian_bell_1d_max_one(
                c_deg=c_para[ph],
                peak_deg=ph_bell_par[ph, 0],
                width_deg=ph_bell_par[ph, 1]
            )
            per_w = gaussian_bell_1d_max_one(
                c_deg=c_perp[ph],
                peak_deg=ph_bell_per[ph, 0],
                width_deg=ph_bell_per[ph, 1]
            )
            par_weights.append(par_w)
            per_weights.append(per_w)

        w = np.sum(par_weights) * np.sum(per_weights) / (num_ph ** 2)

        return 1.0 - w
