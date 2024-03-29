import os


BINNING = {
    "energy_GeV": {"start_support": 1e0, "stop_support": 1e3, "num_bins": 15},
    "azimuth_deg": {"num_bins": 3},
    "radius_m": {"start_support": 0.0, "stop_support": 640.0, "num_bins": 64},
    "altitude_m": {"start_edge": 5000.0, "stop_edge": 29000.0, "num_bins": 24},
    "image_parallel_deg": {
        "start_edge": -0.5,
        "stop_edge": 4.5,
        "num_bins": 120,
    },
    "image_perpendicular_deg": {
        "start_edge": -0.5,
        "stop_edge": 0.5,
        "num_bins": 24,
    },
    "time_s": {"start_edge": -24e-9, "stop_edge": 24e-9, "num_bins": 24},
    "aperture_radius_m": 5.0,
    "aperture_radius_for_timing_m": 50.0,
}

RUN_CONFIG = {
    "corsika_primary_path": os.path.abspath(
        os.path.join(
            "build",
            "corsika",
            "modified",
            "corsika-75600",
            "run",
            "corsika75600Linux_QGSII_urqmd",
        )
    ),
    "energy_to_be_thrown_in_job_GeV": 1e2,
    "max_energy_to_collect_in_altitude_bin_GeV": 1e1,
    "num_jobs_in_energy_bin": 4,
}
