import numpy as np
import json
import skimage
import sklearn
from sklearn import linear_model
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BellLightFieldFitter:
    def __init__(
        self,
        bell_model_lut,
        split_light_field,
        energy_GeV,
        altitude_m,
    ):
        self.bell_model_lut = bell_model_lut
        self.split_light_field = split_light_field
        self.energy_GeV = energy_GeV
        self.altitude_m = altitude_m

        ene_range = bins.find_energy_bins(
            explicit_binning=self.bell_model_lut["explicit_binning"],
            energy_GeV=energy_GeV
        )
        self.ene_bins = [ene_range["lower_bin"], ene_range["upper_bin"]]
        self.ene_weights = [ene_range["lower_weight"], ene_range["upper_weight"]]

        alt_range = bins.find_altitude_bins(
            explicit_binning=self.bell_model_lut["explicit_binning"],
            altitude_m=altitude_m
        )
        self.alt_bins = [alt_range["lower_bin"], alt_range["upper_bin"]]
        self.alt_weights = [alt_range["lower_weight"], alt_range["upper_weight"]]

        for ene in self.ene_bins :
            for alt in self.alt_bins:
                assert self.bell_model_lut["population.ene_alt"][ene, alt]


    def max_core_radius(self):

        min_rad = self.bell_model_lut["binning"]["radius_m"]["num_bins"] - 1

        for ene in self.ene_bins :
            for alt in self.alt_bins:
                for azi in range(self.bell_model_lut["binning"]["azimuth_deg"]["num_bins"]):
                    _max_rad = self.bell_model_lut["max_rad_at.ene_azi_alt"][
                        ene, azi, alt
                    ]
                    if _max_rad < min_rad:
                        min_rad = _max_rad

        return self.bell_model_lut["explicit_binning"]["radius_m"]["supports"][min_rad]


    def model(self, source_cx, source_cy, core_x, core_y):

        # query bell model for each paxel
        # -------------------------------
        num_ph = self.split_light_field.number_photons

        core_x_wrt_paxel = core_x - self.split_light_field.paxel_x
        core_y_wrt_paxel = core_y - self.split_light_field.paxel_y

        core_azimuth_rad = np.arctan2(core_y_wrt_paxel, core_x_wrt_paxel)
        core_azimuth_deg_wrt_paxel = np.rad2deg(core_azimuth_rad)
        core_radius_wrt_paxel = np.hypot(core_x_wrt_paxel, core_y_wrt_paxel)

        pax_bell_par = []
        pax_bell_per = []
        for pax in range(self.split_light_field.number_paxel):
            par, per = query_bell_model(
                bell_model_lut=self.bell_model_lut,
                energy_GeV=self.energy_GeV,
                azimuth_deg=core_azimuth_deg_wrt_paxel[pax],
                radius_m=core_radius_wrt_paxel[pax],
                altitude_m=self.altitude_m
            )
            pax_bell_par.append(par)
            pax_bell_per.append(per)
        pax_bell_par = np.array(pax_bell_par)
        pax_bell_per = np.array(pax_bell_per)

        # transform images into source-frame
        # ----------------------------------
        WRT_DOWNWARDS = -1.0
        pax_img_par_wrt_source = []
        pax_img_per_wrt_source = []
        for pax in range(self.split_light_field.number_paxel):
            ones = np.ones(self.split_light_field.image_sequences[pax].shape[0])
            c_para, c_perp = projection.project_light_field_onto_source_image(
                cer_cx_rad=WRT_DOWNWARDS * self.split_light_field.image_sequences[pax][:, 0],
                cer_cy_rad=WRT_DOWNWARDS * self.split_light_field.image_sequences[pax][:, 1],
                cer_x_m=ones * self.split_light_field.paxel_x[pax],
                cer_y_m=ones * self.split_light_field.paxel_y[pax],
                primary_cx_rad=WRT_DOWNWARDS * source_cx,
                primary_cy_rad=WRT_DOWNWARDS * source_cy,
                primary_core_x_m=core_x,
                primary_core_y_m=core_y,
            )
            pax_img_par_wrt_source.append(c_para)
            pax_img_per_wrt_source.append(c_perp)

        """
        c_par_bin_edges = np.linspace(-4.5, 4.5, 90+1)
        c_par_bin_centers = 0.5 * (c_par_bin_edges[0:-1] + c_par_bin_edges[1:])

        c_per_bin_edges = np.linspace(-4.5, 4.5, 90+1)
        c_per_bin_centers = 0.5 * (c_per_bin_edges[0:-1] + c_per_bin_edges[1:])

        # histograms
        # ----------
        pax_hist_par_wrt_source = []
        pax_hist_per_wrt_source = []
        for pax in range(self.split_light_field.number_paxel):
            h_par = np.histogram(
                a=np.rad2deg(pax_img_par_wrt_source[pax]),
                bins=c_par_bin_edges
            )[0]
            pax_hist_par_wrt_source.append(h_par)
            h_per = np.histogram(
                a=np.rad2deg(pax_img_per_wrt_source[pax]),
                bins=c_per_bin_edges
            )[0]
            pax_hist_per_wrt_source.append(h_per)
        pax_hist_par_wrt_source = np.array(pax_hist_par_wrt_source, dtype=np.float32)
        pax_hist_per_wrt_source = np.array(pax_hist_per_wrt_source, dtype=np.float32)

        # normalize histograms
        # --------------------
        for pax in range(self.split_light_field.number_paxel):

            h_par_sum = np.max(pax_hist_par_wrt_source[pax])
            if h_par_sum > 0:
                pax_hist_par_wrt_source[pax, :] *= (1/h_par_sum)

            h_per_sum = np.max(pax_hist_per_wrt_source[pax])
            if h_per_sum > 0:
                pax_hist_per_wrt_source[pax, :] *= (1/h_per_sum)


        # overlap with histograms
        # -----------------------
        w_par = 0.0
        w_per = 0.0
        for pax in range(self.split_light_field.number_paxel):
            hist_par_actual = pax_hist_par_wrt_source[pax]
            hist_par_model = gaussian_bell_1d_max_one(
                c_deg=c_par_bin_centers,
                peak_deg=pax_bell_par[pax, 0],
                width_deg=pax_bell_par[pax, 1]
            )
            #hist_par_model /= np.sum(hist_par_model)
            w_par += np.sum(np.abs(hist_par_actual - hist_par_model)) * len(pax_img_par_wrt_source[pax])

            hist_per_actual = pax_hist_per_wrt_source[pax]
            hist_per_model = gaussian_bell_1d_max_one(
                c_deg=c_per_bin_centers,
                peak_deg=pax_bell_per[pax, 0],
                width_deg=pax_bell_per[pax, 1]
            )
            #hist_per_model /= np.sum(hist_per_model)
            w_per += np.sum(np.abs(hist_per_actual - hist_per_model)) * len(pax_img_par_wrt_source[pax])

            if pax == 17:
                print("=============")
                print("num: ", len(pax_img_par_wrt_source[pax]))
                print("par peak/width deg: ", pax_bell_par[pax, 0], pax_bell_par[pax, 1])
                print("per peak/width deg: ", pax_bell_per[pax, 0], pax_bell_per[pax, 1])
                print(np.round(hist_par_actual, 2))
                print(np.round(hist_par_model, 2))

        w = np.log10((w_par + w_per))
        print(w)
        return w
        """

        # overlap with single photons
        # ---------------------------
        par_weights = 0
        per_weights = 0
        par_maxs = 0
        per_maxs = 0
        for pax in range(self.split_light_field.number_paxel):
            num_ph_pax = pax_img_par_wrt_source[pax].shape[0]
            _par_w = gaussian_bell_1d(
                c_deg=np.rad2deg(pax_img_par_wrt_source[pax]),
                peak_deg=pax_bell_par[pax, 0],
                width_deg=pax_bell_par[pax, 1]
            )
            _par_max = 1.0/(SQRT_TWO_PI * pax_bell_par[pax, 1])
            par_weights += (num_ph_pax * _par_max) - np.sum(_par_w)

            _per_w = gaussian_bell_1d(
                c_deg=np.rad2deg(pax_img_per_wrt_source[pax]),
                peak_deg=pax_bell_per[pax, 0],
                width_deg=pax_bell_per[pax, 1]
            )
            _per_max = 1.0/(SQRT_TWO_PI * pax_bell_per[pax, 1])
            per_weights += (num_ph_pax * _per_max) - np.sum(_per_w)

        w = (par_weights + per_weights) / (2.0 * num_ph)
        print("=============")
        print("w: ", w)

        return w


        """
        if w > 0.8:
            for pax in range(self.split_light_field.number_paxel):
                fig = plt.figure()
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax.set_aspect("equal")
                ax.plot(
                    np.rad2deg(pax_img_par_wrt_source[pax]),
                    np.rad2deg(pax_img_per_wrt_source[pax]),
                    "xk"
                )
                ax.set_xlabel("c_para / deg")
                ax.set_ylabel("c_perp / deg")
                ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
                fig.savefig("_projection_pax{:03d}".format(pax))
                plt.close(fig)
        """

