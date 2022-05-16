import airshower_template_generator as atg
import numpy as np
import pytest


assert_close = np.testing.assert_almost_equal


binning = {
    "energy_GeV": {"start_support": 1e0, "stop_support": 1e3, "num_bins": 3,},
    "azimuth_deg": {"num_bins": 3},
    "radius_m": {"start_support": 0.0, "stop_support": 3.0, "num_bins": 4},
    "altitude_m": {"start_edge": 0, "stop_edge": 30e3, "num_bins": 3},
    "image_parallel_deg": {
        "start_edge": -0.5,
        "stop_edge": 2.5,
        "num_bins": 96,
    },
    "image_perpendicular_deg": {
        "start_edge": -0.5,
        "stop_edge": 0.5,
        "num_bins": 32,
    },
    "time_s": {"start_edge": -32e-9, "stop_edge": 32e-9, "num_bins": 16,},
    "aperture_radius_m": 5.0,
}


def test_find_bins():
    explicit_binning = atg.bins.make_explicit_binning(binning=binning)

    assert len(explicit_binning["radius_m"]["supports"]) == 4

    b = atg.bins.find_bins(
        explicit_binning=explicit_binning,
        energy_GeV=1.0e0,
        altitude_m=5.0e3,
        azimuth_deg=0.0,
        radius_m=0.5,
    )

    assert b["energy_GeV"][0]["bin"] == 0
    assert b["energy_GeV"][0]["weight"] == 1.0
    assert b["energy_GeV"][1]["bin"] == 1
    assert b["energy_GeV"][1]["weight"] == 0.0

    assert b["azimuth_deg"][0]["bin"] == 0
    assert b["azimuth_deg"][1]["bin"] == 1

    assert b["altitude_m"][0]["bin"] == 0
    assert b["altitude_m"][1]["bin"] == 1

    assert b["radius_m"][0]["bin"] == 0
    assert b["radius_m"][0]["weight"] == 0.5
    assert b["radius_m"][1]["bin"] == 1
    assert b["radius_m"][1]["weight"] == 0.5


def test_energy_out_of_range():
    explicit_binning = atg.bins.make_explicit_binning(binning=binning)

    def find(energy_GeV):
        b = atg.bins.find_bins(
            explicit_binning=explicit_binning,
            energy_GeV=energy_GeV,
            altitude_m=5.0e3,
            azimuth_deg=0.0,
            radius_m=0.5,
        )

    with pytest.raises(IndexError):
        find(energy_GeV=0.99e-1)

    find(energy_GeV=1e0)

    find(energy_GeV=1e2)

    with pytest.raises(IndexError):
        find(energy_GeV=1e3)

    with pytest.raises(IndexError):
        find(energy_GeV=1.01e3)


def test_altitude_out_of_range():
    explicit_binning = atg.bins.make_explicit_binning(binning=binning)

    assert len(explicit_binning["altitude_m"]["supports"]) == 3

    def find(altitude_m):
        b = atg.bins.find_bins(
            explicit_binning=explicit_binning,
            energy_GeV=1e2,
            altitude_m=altitude_m,
            azimuth_deg=0.0,
            radius_m=0.5,
        )
        return b["altitude_m"]

    with pytest.raises(IndexError):
        r = find(altitude_m=-1.0)

    with pytest.raises(IndexError):
        r = find(altitude_m=0.0)

    r = find(altitude_m=5e3)
    assert r[0]["bin"] == 0
    assert r[0]["weight"] == 1.0
    assert r[1]["bin"] == 1
    assert r[1]["weight"] == 0.0

    r = find(altitude_m=24.99e3)
    assert r[0]["bin"] == 1
    assert r[0]["weight"] < 0.1

    assert r[1]["bin"] == 2
    assert r[1]["weight"] > 0.9

    with pytest.raises(IndexError):
        find(altitude_m=30e3)

    with pytest.raises(IndexError):
        find(altitude_m=30.1e3)
