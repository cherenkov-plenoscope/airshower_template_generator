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
    """
    Chapter 4, Equation 4.18, page: 76
    """

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


def make_image(binning, moyal_peak_deg, width_moyal_deg, width_deg):
    num_para = binning["image_parallel_deg"]["num_bins"]
    num_perp = binning["image_perpendicular_deg"]["num_bins"]

    _eb = atg.bins.make_explicit_binning(binning=binning)
    para_supports = _eb["image_parallel_deg"]["supports"]
    perp_supports = _eb["image_perpendicular_deg"]["supports"]

    img = np.zeros(shape=(num_para, num_perp))
    for i_para, c_para in enumerate(para_supports):
        for i_perp, c_perp in enumerate(perp_supports):
            img[i_para, i_perp] = amp(
                c_para_deg=c_para,
                c_perp_deg=c_perp,
                moyal_peak_deg=moyal_peak_deg,
                width_moyal_deg=width_moyal_deg,
                width_deg=width_deg,
            )

    return img


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


for ii, moyal_peak_deg in enumerate(np.linspace(0.0, 1.5, 15)):
    binning = atg.examples.BINNING
    img = make_image(
        binning=binning,
        moyal_peak_deg=moyal_peak_deg,
        width_moyal_deg=0.1,
        width_deg=0.1,
    )
    atg.plot.write_image(
        path="moyal_img_{:06d}.jpg".format(ii),
        binning=binning,
        image=img,
        x_key="image_parallel_deg",
        y_key="image_perpendicular_deg",
    )
