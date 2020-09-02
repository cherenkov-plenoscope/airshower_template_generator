import airshower_template_generator as atg
import numpy as np
import pytest


assert_close = np.testing.assert_almost_equal


def test_finding_bin_index_in_edges():
    VALUE = 0
    UNDERFLOW = 1
    BIN_IDX = 2
    OVERFLOW = 3
    bin_edges = np.linspace(0, 1, 5)
    scenarios = [
        (-0.1, True, 0, False),
        (0.0, False, 0, False),
        (0.1, False, 0, False),
        (0.23, False, 0, False),
        (0.26, False, 1, False),
        (0.49, False, 1, False),
        (0.51, False, 2, False),
        (0.74, False, 2, False),
        (0.76, False, 3, False),
        (0.99, False, 3, False),
        (1.0, False, 4, True),
        (1.1, False, 4, True),
    ]

    for scenario in scenarios:
        print(scenario)
        underflow, bin_idx, overflow = atg.bins.find_bin_in_edges(
            bin_edges=bin_edges, value=scenario[VALUE]
        )

        assert scenario[UNDERFLOW] == underflow
        assert scenario[BIN_IDX] == bin_idx
        assert scenario[OVERFLOW] == overflow


def test_finding_bin_index_in_centers():
    VALUE = 0
    UNDERFLOW = 1
    LOWER_IDX = 2
    OVERFLOW = 3
    UPPER_IDX = 4
    LOWER_WEIGHT = 5
    bin_centers = np.linspace(0, 1, 6)
    # 0    1     2    3     4    5
    # .0,  .2,  .4,   .6,   .8   1.
    scenarios = [
        (-0.1, True, 0, False, 1, 0.0),
        (0.0, False, 0, False, 1, 1.0),
        (0.1, False, 0, False, 1, 0.5),
        (0.19, False, 0, False, 1, 0.05),
        (0.3, False, 1, False, 2, 0.5),
        (0.41, False, 2, False, 3, 0.95),
        (0.5, False, 2, False, 3, 0.5),
        (0.75, False, 3, False, 4, 0.25),
        (0.8, False, 4, False, 5, 1.0),
        (0.95, False, 4, False, 5, 0.25),
        (1.0, False, 5, True, 6, 1.0),
        (1.1, False, 5, True, 6, 1.0),
    ]

    for scenario in scenarios:
        print(scenario)
        match = atg.bins.find_bins_in_centers(
            bin_centers=bin_centers, value=scenario[VALUE]
        )

        assert scenario[UNDERFLOW] == match["underflow"]
        assert scenario[OVERFLOW] == match["overflow"]
        assert scenario[LOWER_IDX] == match["lower_bin"]
        assert scenario[UPPER_IDX] == match["upper_bin"]
        assert_close(scenario[LOWER_WEIGHT], match["lower_weight"])
        assert_close(match["lower_weight"] + match["upper_weight"], 1.0)


def test_finding_bin_index_in_circular():
    VALUE = 0
    UNDERFLOW = 1
    LOWER_IDX = 2
    OVERFLOW = 3
    UPPER_IDX = 4
    LOWER_WEIGHT = 5
    bin_centers = np.linspace(0, 1, 3)
    # 0., .5, 1.
    scenarios = [
        (0.0, False, 0, False, 1, 1.0),
        (-1e-6, True, 0, False, 1, 0.0),
        (1 - 1e-6, False, 1, False, 2, 2e-6),
        (1, False, 2, True, 3, 1.0),
    ]

    for scenario in scenarios:
        print(scenario)
        match = atg.bins.find_bins_in_centers(
            bin_centers=bin_centers, value=scenario[VALUE]
        )

        assert scenario[UNDERFLOW] == match["underflow"]
        assert scenario[OVERFLOW] == match["overflow"]
        assert scenario[LOWER_IDX] == match["lower_bin"]
        assert scenario[UPPER_IDX] == match["upper_bin"]
        assert_close(scenario[LOWER_WEIGHT], match["lower_weight"])
        assert_close(match["lower_weight"] + match["upper_weight"], 1.0)


binning = {
    "energy_GeV": {
        "start_support": 1e0,
        "stop_support": 1e3,
        "num_bins": 3,
    },
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
