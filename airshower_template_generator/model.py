import numpy as np


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


def my_model(c_deg, moyal_peak_deg, tail_direction, width_deg):
    moyal_density = (1.0 - tail_direction) * lorentz_moyal_longitidinal(
        c_deg=c_deg, peak_deg=moyal_peak_deg, width_deg=width_deg,
    )
    gaussian_density = tail_direction * lorentz_moyal_longitidinal(
        c_deg=-c_deg, peak_deg=-moyal_peak_deg, width_deg=width_deg,
    )
    return moyal_density + gaussian_density
