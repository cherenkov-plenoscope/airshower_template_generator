import numpy as np
import airshower_template_generator as atg


SQRT_TWO_PI = np.sqrt(2.0 * np.pi)

"""
n: normalisation
p: peak along axis-u
l: length_of_moyal
"""


def gaussian_bell_1d(c_deg, peak_deg, width_deg):
    return (
        1.0
        / (width_deg * SQRT_TWO_PI)
        * np.exp(-0.5 * (c_deg - peak_deg) ** 2 / width_deg ** 2)
    )


def lorentz_transversal(c_deg, peak_deg, width_deg):
    return width_deg / (np.pi * (width_deg ** 2 + (c_deg - peak_deg) ** 2))


def lorentz_moyal_longitidinal(
    c_deg, moyal_peak_deg, width_deg,
):
    lam = (c_deg - moyal_peak_deg) / width_deg
    return (
        1.0 / (width_deg * SQRT_TWO_PI) * np.exp(-0.5 * (lam + np.exp(-lam)))
    )


"""
def modified_lorentz_moyal_longitidinal(
    c_deg, moyal_peak_deg, width_deg, transition_angle_deg=0.5
):
    bell_weight = 1.0 - (moyal_peak_deg / transition_angle_deg) ** 0.5
    if bell_weight > 1.0:
        bell_weight = 1.0
    if bell_weight < 0.0:
        bell_weight = 0.0
    moyal_weight = 1.0 - bell_weight

    return (
        gaussian_bell_1d(
            c_deg=c_deg, peak_deg=moyal_peak_deg, width_deg=width_deg,
        )
        * bell_weight
        + lorentz_moyal_longitidinal(
            c_deg=c_deg, moyal_peak_deg=moyal_peak_deg, width_deg=width_deg,
        )
        * moyal_weight
    )


def amp(
    c_para_deg, c_perp_deg, moyal_peak_deg, width_moyal_deg, width_deg,
):
    assert width_moyal_deg >= width_deg

    # Chapter 4, Equation 4.18, page: 76

    L_Moyal = modified_lorentz_moyal_longitidinal(
        c_deg=c_para_deg,
        moyal_peak_deg=moyal_peak_deg,
        width_deg=width_moyal_deg,
    )

    bell_weight = 1.0 - moyal_peak_deg ** 0.5
    if bell_weight > 1.0:
        bell_weight = 1.0
    if bell_weight < 0.0:
        bell_weight = 0.0
    moyal_weight = 1.0 - bell_weight

    T_Lorentz = gaussian_bell_1d(
        c_deg=c_perp_deg,
        peak_deg=0.0,
        width_deg=width_moyal_deg * bell_weight + moyal_weight * width_deg,
    )

    return L_Moyal * T_Lorentz
    # return L_Moyal * (T_Lorentz / (np.pi * (T_Lorentz ** 2 + c_perp_deg ** 2)))
"""


"""
def simple_hillas_ellipse(
    c_para_deg, c_perp_deg, para_peak_deg, para_width_deg, perp_width_deg,
):
    w_para = gaussian_bell_1d(
        c_deg=c_para_deg,
        peak_deg=para_peak_deg,
        width_deg=para_width_deg
    )
    w_perp = gaussian_bell_1d(
        c_deg=c_perp_deg,
        peak_deg=0.0,
        width_deg=perp_width_deg
    )
    return w_para * w_perp



def make_per_par_image(model, model_parameters, binning, supersampling=2):
    assert supersampling >= 1
    num_para = binning["image_parallel_deg"]["num_bins"]
    num_perp = binning["image_perpendicular_deg"]["num_bins"]

    super_para_supports_deg = np.linspace(
        binning["image_parallel_deg"]["start_edge"],
        binning["image_parallel_deg"]["stop_edge"],
        (binning["image_parallel_deg"]["num_bins"] * supersampling)
    )
    super_perp_supports_deg = np.linspace(
        binning["image_perpendicular_deg"]["start_edge"],
        binning["image_perpendicular_deg"]["stop_edge"],
        (binning["image_perpendicular_deg"]["num_bins"] * supersampling)
    )

    img = np.zeros(shape=(num_para, num_perp))
    for i_super_para, c_super_para_deg in enumerate(super_para_supports_deg):
        for i_super_perp, c_super_perp_deg in enumerate(super_perp_supports_deg):

            i_para = i_super_para // supersampling
            i_perp = i_super_perp // supersampling

            img[i_para, i_perp] += model(
                c_para_deg=c_super_para_deg,
                c_perp_deg=c_super_perp_deg,
                **model_parameters
            )
    img /= (supersampling*supersampling)
    return img
"""


#!/usr/bin/python
import sys
import os
import pandas as pd
import numpy as np
import json
import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
c_para_deg = np.linspace(-0.5, 2.5, 101)

fig = plt.figure(figsize=(16, 9), dpi=120)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_title("moyal")
for peak in np.linspace(0.0, 2.0, 10):
    width = 0.3
    ax.plot(
        c_para_deg,
        lorentz_moyal_longitidinal(
            c_deg=c_para_deg, moyal_peak_deg=peak, width_deg=width,
        ),
        "b",
    )
    ax.plot(
        c_para_deg,
        gaussian_bell_1d(c_deg=c_para_deg, peak_deg=peak, width_deg=width,),
        "r",
    )
    ax.plot(
        c_para_deg,
        modified_lorentz_moyal_longitidinal(
            c_deg=c_para_deg, moyal_peak_deg=peak, width_deg=width,
        ),
        "g",
    )
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)
ax.set_xlabel("c-parallel / $^{\\circ}$")
plt.savefig("moyal.jpg")
plt.close(fig)


c_perp_deg = np.linspace(-1.0, 1.0, 101)

fig = plt.figure(figsize=(16, 9), dpi=120)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_title("lorentz_transversal")

width = 0.3
ax.plot(
    c_perp_deg,
    lorentz_transversal(c_deg=c_perp_deg, peak_deg=0.0, width_deg=width,),
    "b",
)
ax.plot(
    c_perp_deg,
    gaussian_bell_1d(c_deg=c_perp_deg, peak_deg=0.0, width_deg=width,),
    "r",
)
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)
ax.set_xlabel("c-perpendicular / $^{\\circ}$")
plt.savefig("lorentz_transversal.jpg")
plt.close(fig)



for ii, moyal_peak_deg in enumerate(np.linspace(0.0, 3.5, 15)):
    binning = atg.examples.BINNING

    amp_model_parameters = {
        "moyal_peak_deg": moyal_peak_deg,
        "width_moyal_deg": 0.1,
        "width_deg": 0.1,
    }

    hillas_model_parameters = {
        "para_peak_deg": moyal_peak_deg,
        "para_width_deg": 0.5,
        "perp_width_deg": 0.1,
    }

    img = make_per_par_image(
        model=simple_hillas_ellipse,
        binning=binning,
        model_parameters=hillas_model_parameters,
        supersampling=1
    )
    atg.plot.write_image(
        path="moyal_img_{:06d}.jpg".format(ii),
        binning=binning,
        image=img,
        x_key="image_parallel_deg",
        y_key="image_perpendicular_deg",
    )
"""

from iminuit import Minuit

"""
class BT:
    def __init__(self, image_look_up_table, model, binning, supersampling=2):
        self.image_look_up_table = np.array(image_look_up_table) / np.sum(image_look_up_table)
        self.model = model
        self.binning = binning
        self.supersampling = supersampling

    def fcn(self, para_peak_deg, para_width_deg, perp_width_deg):
        image_model = make_per_par_image(
            model=self.model,
            model_parameters={
                "para_peak_deg": para_peak_deg,
                "para_width_deg": para_width_deg,
                "perp_width_deg": perp_width_deg
            },
            binning=self.binning,
            supersampling=self.supersampling
        )
        image_model /= np.sum(image_model)
        square_diff_image = (self.image_look_up_table - image_model)**2
        diff = np.sum(square_diff_image)
        return diff
"""

"""
def fit_model_to_look_up_table(model, lut, supersampling=1):
    b = lut["binning"]
    eb = lut["explicit_binning"]

    results = {}
    for ene in np.arange(0, b["energy_GeV"]["num_bins"], 4):
        results[ene] = {}
        for azi in np.arange(0, b["azimuth_deg"]["num_bins"], 3):
            results[ene][azi] = {}
            for rad in np.arange(0, b["radius_m"]["num_bins"], 8):
                results[ene][azi][rad] = {}
                for alt in np.arange(0, b["altitude_m"]["num_bins"], 4):
                    results[ene][azi][rad][alt] = {}

                    print(ene, azi, rad, alt)

                    target_parameters = {
                        "energy_GeV": eb["energy_GeV"]["supports"][ene],
                        "azimuth_deg": eb["azimuth_deg"]["supports"][azi],
                        "radius_m":  eb["radius_m"]["supports"][rad],
                        "altitude_m":  eb["altitude_m"]["supports"][alt],
                    }

                    img = lut["cherenkov.density.ene_azi_rad_alt_par_per"][
                        ene, azi, rad, alt
                    ]

                    B = BT(
                        image_look_up_table=img,
                        model=model,
                        binning=b,
                        supersampling=supersampling
                    )

                    mm = Minuit(
                        fcn=B.fcn,
                        para_peak_deg=2.0,
                        limit_para_peak_deg=[-0.5, 4.5],
                        para_width_deg=2.0,
                        limit_para_width_deg=[0.0, 4.5],
                        perp_width_deg=1.0,
                        limit_perp_width_deg=[0.0, 2.0],
                        print_level=0,
                    )
                    mm.migrad()
                    model_parameters = dict(mm.values)
                    print(model_parameters)

                    results[ene][azi][rad][alt]["target"] = target_parameters
                    results[ene][azi][rad][alt]["model"] = model_parameters

                    atg.plot.write_image(
                        path="fit_{:02d}_{:02d}_{:02d}_{:02d}_lut.jpg".format(ene, azi, rad, alt),
                        binning=b,
                        image=img,
                        x_key="image_parallel_deg",
                        y_key="image_perpendicular_deg",
                    )
                    mod = make_per_par_image(
                        model=model,
                        model_parameters=model_parameters,
                        binning=b
                    )
                    atg.plot.write_image(
                        path="fit_{:02d}_{:02d}_{:02d}_{:02d}_mod.jpg".format(ene, azi, rad, alt),
                        binning=b,
                        image=mod,
                        x_key="image_parallel_deg",
                        y_key="image_perpendicular_deg",
                    )

    return results
"""


def my_model(c_deg, moyal_peak_deg, tail_direction, width_deg):
        moyal_density = (1.0 - tail_direction) * lorentz_moyal_longitidinal(
            c_deg=c_deg,
            moyal_peak_deg=moyal_peak_deg,
            width_deg=width_deg,
        )
        gaussian_density = tail_direction * lorentz_moyal_longitidinal(
            c_deg=-c_deg,
            moyal_peak_deg=-moyal_peak_deg,
            width_deg=width_deg,
        )
        return moyal_density + gaussian_density



class MoyalFit:
    def __init__(self, density_c_parallel, binning):
        self.sum = np.sum(density_c_parallel)
        self.density_c_parallel = np.array(density_c_parallel) / self.sum
        eb = atg.bins.make_explicit_binning(binning)
        self.supports_image_parallel_deg = eb["image_parallel_deg"]["supports"]

    def fcn(self, moyal_peak_deg, tail_direction, width_deg):
        density_c_parallel_model = my_model(
            c_deg=self.supports_image_parallel_deg,
            moyal_peak_deg=moyal_peak_deg,
            tail_direction=tail_direction,
            width_deg=width_deg,
        )

        density_c_parallel_model /= np.sum(density_c_parallel_model)

        square_diff_image = (
            self.density_c_parallel - density_c_parallel_model
        ) ** 2
        diff = np.sqrt(np.sum(square_diff_image))

        return diff


def project_lut_on_c_parallel(lut):
    binning = lut["binning"]

    density_c_parallel = np.zeros(
        shape=(
            binning["energy_GeV"]["num_bins"],
            binning["azimuth_deg"]["num_bins"],
            binning["radius_m"]["num_bins"],
            binning["altitude_m"]["num_bins"],
            binning["image_parallel_deg"]["num_bins"],
        ),
        dtype=np.float32
    )

    i_middle = 0.5 * binning["image_perpendicular_deg"]["num_bins"]
    i_start = int(np.floor(i_middle - 1))
    i_stop = int(np.ceil(i_middle + 1))
    span = i_stop - i_start
    assert span == 2

    kernel = np.array([1, 2, 5, 2, 1])
    kernel = kernel/np.sum(kernel)

    for ene in range(binning["energy_GeV"]["num_bins"]):
        for azi in range(binning["azimuth_deg"]["num_bins"]):
            for rad in range(binning["radius_m"]["num_bins"]):
                for alt in range(binning["altitude_m"]["num_bins"]):

                    img = lut["cherenkov.density.ene_azi_rad_alt_par_per"][
                        ene, azi, rad, alt
                    ]

                    middle_slice = img[:, i_start:i_stop]

                    arr = np.sum(middle_slice, axis=1) * (1.0/span)

                    smoo = np.convolve(arr, kernel, mode="same")

                    density_c_parallel[ene, azi, rad, alt] = smoo

    return density_c_parallel


lut = atg.read_raw("lookup_2020-09-10/reduce/namibia/gamma/raw.tar")
binning = lut["binning"]
eb = lut["explicit_binning"]

density_c_parallel = project_lut_on_c_parallel(lut=lut)

IDX_MOYAL_PEAK_DEG = 0
IDX_WIDTH_DEG = 1
IDX_TAIL_DIRECTION = 2
IDX_SUM = 3
NUM_MOYAL_PARAMETERS = 4

path = "moyal_fit_smoo_lin.ene_azi_rad_alt.c_order.float32"

shape = (
    binning["energy_GeV"]["num_bins"],
    binning["azimuth_deg"]["num_bins"],
    binning["radius_m"]["num_bins"],
    binning["altitude_m"]["num_bins"],
    NUM_MOYAL_PARAMETERS,
)

if os.path.exists(path):

    with open(path, "rb") as fin:
        results = np.frombuffer(fin.read(), dtype=np.float32)
    results = np.reshape(results, shape)

else:
    results = np.zeros(
        shape=shape,
        dtype=np.float32
    )

    ene_range = np.arange(0, binning["energy_GeV"]["num_bins"], 1)
    azi_range = np.arange(0, binning["azimuth_deg"]["num_bins"], 1)
    rad_range = np.arange(0, binning["radius_m"]["num_bins"], 1)
    alt_range = np.arange(0, binning["altitude_m"]["num_bins"], 1)

    for ene in ene_range:
        for azi in azi_range:
            for rad in rad_range:
                for alt in alt_range:

                    print(ene, azi, rad, alt)

                    moyal_fit = MoyalFit(
                        density_c_parallel=density_c_parallel[ene, azi, rad, alt],
                        binning=binning,
                    )

                    mm = Minuit(
                        fcn=moyal_fit.fcn,
                        moyal_peak_deg=2.0,
                        width_deg=1.0,
                        tail_direction=0.5,
                        limit_moyal_peak_deg=[-0.0, 5.5],
                        limit_width_deg=[0.0, 10.0],
                        limit_tail_direction=[0.0, 1.0],
                        errordef=Minuit.LEAST_SQUARES,
                        print_level=0,
                    )
                    mm.migrad()

                    results[ene, azi, rad, alt, IDX_MOYAL_PEAK_DEG] = mm.values["moyal_peak_deg"]
                    results[ene, azi, rad, alt, IDX_WIDTH_DEG] = mm.values["width_deg"]
                    results[ene, azi, rad, alt, IDX_TAIL_DIRECTION] = mm.values["tail_direction"]
                    results[ene, azi, rad, alt, IDX_SUM] = moyal_fit.sum

    with open(path, "wb") as fout:
        fout.write(results.tobytes())


for ene in np.arange(0, binning["energy_GeV"]["num_bins"], 2):
    for azi in np.arange(0, binning["azimuth_deg"]["num_bins"], 3):
        for rad in np.arange(0, binning["radius_m"]["num_bins"], 4):
            for alt in np.arange(0, binning["altitude_m"]["num_bins"], 2):

                if lut["airshower.histogram.ene_alt"][ene, alt] < 100:
                    continue

                I_lut = density_c_parallel[ene, azi, rad, alt]
                I_lut /= np.sum(I_lut)

                I_best_model = my_model(
                    c_deg=eb["image_parallel_deg"]["supports"],
                    moyal_peak_deg=results[ene, azi, rad, alt, IDX_MOYAL_PEAK_DEG],
                    width_deg=results[ene, azi, rad, alt, IDX_WIDTH_DEG],
                    tail_direction=results[ene, azi, rad, alt, IDX_TAIL_DIRECTION],
                )
                I_best_model /= np.sum(I_best_model)

                plt.figure()
                plt.plot(
                    eb["image_parallel_deg"]["supports"],
                    I_lut,
                     "r",
                     alpha=0.5,
                )
                plt.plot(
                    eb["image_parallel_deg"]["supports"],
                    I_best_model,
                    "b",
                     alpha=0.5,
                )
                plt.savefig(
                    "fit_{:02d}_{:02d}_{:02d}_{:02d}_mod.jpg".format(
                        ene, azi, rad, alt
                    )
                )
                plt.close("all")
