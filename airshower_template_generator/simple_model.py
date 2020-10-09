import numpy as np
import airshower_template_generator as atg
from iminuit import Minuit
import sys
import os
import pandas as pd
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# input
# -----
lut = atg.read_raw("2020-09-26_gamma_lut/reduce/namibia/gamma/raw.tar")
binning = lut["binning"]
ebinning = lut["explicit_binning"]


# population
# ----------
population_ene_alt = np.zeros(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["altitude_m"]["num_bins"],
    ),
    dtype=np.int8
)
E_MIN_GEV = 1e3
for ene in range(binning["energy_GeV"]["num_bins"]):
    for alt in range(binning["altitude_m"]["num_bins"]):
        num_actual = lut["airshower.histogram.ene_alt"][ene, alt]
        num_expected = E_MIN_GEV / ebinning["energy_GeV"]["supports"][ene]
        population_ene_alt[ene, alt] = num_actual >= num_expected


# project images on para and perp axis
# ------------------------------------
cer_para = np.zeros(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["azimuth_deg"]["num_bins"],
        binning["radius_m"]["num_bins"],
        binning["altitude_m"]["num_bins"],
        binning["image_parallel_deg"]["num_bins"],
    ),
    dtype=np.float32
)

cer_perp = np.zeros(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["azimuth_deg"]["num_bins"],
        binning["radius_m"]["num_bins"],
        binning["altitude_m"]["num_bins"],
        binning["image_perpendicular_deg"]["num_bins"],
    ),
    dtype=np.float32
)

for ene in range(binning["energy_GeV"]["num_bins"]):
    for azi in range(binning["azimuth_deg"]["num_bins"]):
        for rad in range(binning["radius_m"]["num_bins"]):
            for alt in range(binning["altitude_m"]["num_bins"]):
                img = lut["cherenkov.density.ene_azi_rad_alt_par_per"][
                        ene, azi, rad, alt, :, :
                    ]
                cer_para[ene, azi, rad, alt, :] = np.sum(img, axis=1)
                cer_perp[ene, azi, rad, alt, :] = np.sum(img, axis=0)

# model
# -----

class BellFit:
    def __init__(self, cer, supports):
        self.sum = np.sum(cer)
        self.cer = np.array(cer) / self.sum
        eb = atg.bins.make_explicit_binning(binning)
        self.supports = supports

    def fcn(self, peak_deg, width_deg):
        cer_model = atg.model.gaussian_bell_1d(
            c_deg=self.supports,
            peak_deg=peak_deg,
            width_deg=width_deg,
        )
        cer_model /= np.sum(cer_model)
        square_diff_image = (self.cer - cer_model) ** 2
        diff = np.sqrt(np.sum(square_diff_image))
        return diff


bell_para =  np.nan * np.ones(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["azimuth_deg"]["num_bins"],
        binning["radius_m"]["num_bins"],
        binning["altitude_m"]["num_bins"],
        2
    )
)

bell_perp = np.nan * np.ones(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["azimuth_deg"]["num_bins"],
        binning["radius_m"]["num_bins"],
        binning["altitude_m"]["num_bins"],
        2
    )
)

num_para = binning["image_parallel_deg"]["num_bins"]
para_edge_idx = int(num_para*0.95)

max_valid_radius = np.zeros(
    shape=(
        binning["energy_GeV"]["num_bins"],
        binning["azimuth_deg"]["num_bins"],
        binning["altitude_m"]["num_bins"],
    ),
    dtype=np.int8
)

# fit simple bell model
for ene in range(binning["energy_GeV"]["num_bins"]):
    for alt in range(binning["altitude_m"]["num_bins"]):
        if not population_ene_alt[ene, alt]:
            continue

        for azi in range(binning["azimuth_deg"]["num_bins"]):
            for rad in range(binning["radius_m"]["num_bins"]):

                print(ene, azi, rad, alt)

                cer_para_slice = cer_para[ene, azi, rad, alt]

                inner = np.sum(cer_para_slice[0:para_edge_idx])
                outer = np.sum(cer_para_slice[para_edge_idx:])

                if outer > 0.1 * inner:
                    print("leakage", ene, azi, rad, alt)
                    # stop radius component all together. Break to next azimuth
                    break
                else:
                    max_valid_radius[ene, azi, alt] += 1


                bell_fit_para = BellFit(
                    cer=cer_para_slice,
                    supports=ebinning["image_parallel_deg"]["supports"],
                )

                mpara = Minuit(
                    fcn=bell_fit_para.fcn,
                    peak_deg=2.0,
                    width_deg=1.0,
                    limit_peak_deg=[-0.0, 5.5],
                    limit_width_deg=[0.01, 10.0],
                    error_peak_deg=0.01,
                    error_width_deg=0.01,
                    errordef=Minuit.LEAST_SQUARES,
                    print_level=0,
                )
                mpara.migrad()
                bell_para[ene, azi, rad, alt, 0] = mpara.values["peak_deg"]
                bell_para[ene, azi, rad, alt, 1] = mpara.values["width_deg"]

                bell_fit_perp = BellFit(
                    cer=cer_perp[ene, azi, rad, alt],
                    supports=ebinning["image_perpendicular_deg"]["supports"],
                )

                mperp = Minuit(
                    fcn=bell_fit_perp.fcn,
                    peak_deg=0.0,
                    width_deg=0.5,
                    limit_peak_deg=[-0.5, 0.5],
                    limit_width_deg=[0.01, 10.0],
                    error_peak_deg=0.01,
                    error_width_deg=0.01,
                    errordef=Minuit.LEAST_SQUARES,
                    print_level=0,
                )
                mperp.migrad()
                bell_perp[ene, azi, rad, alt, 0] = mperp.values["peak_deg"]
                bell_perp[ene, azi, rad, alt, 1] = mperp.values["width_deg"]


# detect outliers
# ---------------
D_PEAK_MAX = 0.3
outlier_cords = []
for ene in range(binning["energy_GeV"]["num_bins"]):

    ene_start = np.max([ene - 1, 0])
    ene_stop = np.min([ene + 1, binning["energy_GeV"]["num_bins"]])
    fene = np.arange(ene_start, ene_stop, 1)

    for alt in range(binning["altitude_m"]["num_bins"]):
        if not population_ene_alt[ene, alt]:
            continue

        alt_start = np.max([alt - 1, 0])
        alt_stop = np.min([alt + 1, binning["altitude_m"]["num_bins"]])
        falt = np.arange(alt_start, alt_stop, 1)

        for azi in range(binning["azimuth_deg"]["num_bins"]):

            azi_start = np.max([azi - 1, 0])
            azi_stop = np.min([azi + 1, binning["azimuth_deg"]["num_bins"]])
            fazi = np.arange(azi_start, azi_stop, 1)

            _rad_num_bins = max_valid_radius[ene, azi, alt]

            for rad in range(_rad_num_bins):

                rad_start = np.max([rad - 1, 0])
                rad_stop = np.min([rad + 1, _rad_num_bins])
                frad = np.arange(rad_start, rad_stop, 1)

                nn = []
                for nene in fene:
                    for nalt in falt:
                        for nazi in fazi:
                            for nrad in frad:

                                nn.append(bell_para[nene, nazi, nrad, nalt, 0])

                med = np.median(nn)
                nom = bell_para[ene, azi, rad, alt, 0]
                if np.abs(med - nom) > D_PEAK_MAX:
                    outlier_cords.append((ene, azi, rad, alt))

# replace outliers
# ----------------
bell_para_fix = np.array(bell_para)
for cord in outlier_cords:
    ene, azi, rad, alt = cord

    rad_l = rad - 1
    rad_u = rad + 1

    reps0 = []
    reps1 = []
    if rad_l >= 0:
        reps0.append(bell_para[ene, azi, rad_l, alt, 0])
        reps1.append(bell_para[ene, azi, rad_l, alt, 1])

    _rad_num_bins = max_valid_radius[ene, azi, alt]
    if rad_u < _rad_num_bins:
        reps0.append(bell_para[ene, azi, rad_u, alt, 0])
        reps1.append(bell_para[ene, azi, rad_u, alt, 1])

    bell_para_fix[ene, azi, rad, alt, 0] = np.mean(reps0)
    bell_para_fix[ene, azi, rad, alt, 1] = np.mean(reps1)


scale = 2
azi = 0
for ene in range(binning["energy_GeV"]["num_bins"]):
    for alt in range(binning["altitude_m"]["num_bins"]):
        if not population_ene_alt[ene, alt]:
            continue

        fig = plt.figure(figsize=(16/scale,9/scale), dpi=100)
        ax_para = fig.add_axes([0.1, 0.1, 0.8, 0.4])

        ax_para.plot(
            ebinning["radius_m"]["supports"],
            bell_para[ene, azi, :, alt, 0],
            "rx",
        )
        ax_para.plot(
            ebinning["radius_m"]["supports"],
            bell_para[ene, azi, :, alt, 1],
            "ro",
        )

        ax_para.plot(
            ebinning["radius_m"]["supports"],
            bell_para_fix[ene, azi, :, alt, 0],
            "kx",
        )
        ax_para.plot(
            ebinning["radius_m"]["supports"],
            bell_para_fix[ene, azi, :, alt, 1],
            "ko",
        )
        ax_para.set_xlim(
            [
                binning["radius_m"]["start_support"],
                binning["radius_m"]["stop_support"]
            ]
        )
        ax_para.set_ylim([0, 5.0])
        ax_para.set_ylabel("longitidinal direction / 1$^{\\circ}$")
        ax_para.set_xlabel("core-disatance / m")
        ax_para.spines["top"].set_color("none")
        ax_para.spines["right"].set_color("none")
        ax_para.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)


        ax_perp = fig.add_axes([0.1, 0.55, 0.8, 0.4])
        ax_perp.plot(
            ebinning["radius_m"]["supports"],
            bell_perp[ene, azi, :, alt, 0],
            "kx",
        )
        ax_perp.plot(
            ebinning["radius_m"]["supports"],
            bell_perp[ene, azi, :, alt, 1],
            "ko",
        )
        ax_perp.set_xlim(
            [
                binning["radius_m"]["start_support"],
                binning["radius_m"]["stop_support"]
            ]
        )
        ax_perp.set_ylim([-.05, .2])
        ax_perp.set_ylabel("transversal direction / 1$^{\\circ}$")
        ax_perp.spines["top"].set_color("none")
        ax_perp.spines["right"].set_color("none")
        ax_perp.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

        ax_perp.set_title(
            "energy {: 4.1f}GeV, altitude {: 4.1f}km".format(
                ebinning["energy_GeV"]["supports"][ene],
                1e-3*ebinning["altitude_m"]["supports"][alt]
            )
        )

        ax_perp.plot(200, 0.17, "ko")
        ax_perp.text(s="width", x=210, y=0.17)
        ax_perp.plot(300, 0.17, "kx")
        ax_perp.text(s="center", x=310, y=0.17)
        ax_perp.plot(400, 0.17, "ro")
        ax_perp.text(s="outlier", x=410, y=0.17)


        fig.savefig(
            "simple_bell_model_ene{:02d}_azi{:02d}_alt{:02d}.jpg".format(
                ene, azi, alt
            )
        )
        plt.close(fig)
