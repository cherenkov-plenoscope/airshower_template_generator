import numpy as np
from . import bins


def query_image(lut, energy_GeV, altitude_m, azimuth_deg, radius_m):
    b = bins.find_bins(
        explicit_binning=lut["explicit_binning"],
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )

    avg_img = np.zeros(
        shape=(
            lut["binning"]["image_parallel_deg"]["num_bins"],
            lut["binning"]["image_perpendicular_deg"]["num_bins"],
        ),
        dtype=np.float32,
    )

    cpd = lut["cherenkov_photon_density"]

    weights = []
    for ene in b["energy_GeV"]:
        for alt in b["altitude_m"]:
            for azi in b["azimuth_deg"]:
                for rad in b["radius_m"]:
                    img = cpd[ene["bin"], azi["bin"], rad["bin"], alt["bin"]]

                    weight = (
                        ene["weight"]
                        * alt["weight"]
                        * azi["weight"]
                        * rad["weight"]
                    )
                    weights.append(weight)
                    avg_img = avg_img + img * weight
    return avg_img / np.sum(weights)


def benchmark(lut, num_queries=1000):
    _b = lut["binning"]
    _eb = lut["explicit_binning"]
    request = 0
    while request < num_queries:
        energy_GeV = np.random.uniform(
            low=_b["energy_GeV"]["start_support"],
            high=_b["energy_GeV"]["stop_support"],
            size=1,
        )[0]

        altitude_m = np.random.uniform(
            low=_eb["altitude_m"]["supports"][0],
            high=_eb["altitude_m"]["supports"][-1],
            size=1,
        )[0]

        azimuth_deg = np.random.uniform(low=0.0, high=360.0, size=1)[0]

        radius_m = np.random.uniform(
            low=_b["radius_m"]["start_support"],
            high=_b["radius_m"]["stop_support"],
            size=1,
        )[0]

        _ = query_image(
            lut=lut,
            energy_GeV=energy_GeV,
            altitude_m=altitude_m,
            azimuth_deg=azimuth_deg,
            radius_m=radius_m,
        )
        request += 1
    return True
