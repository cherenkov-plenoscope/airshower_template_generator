import os

SITES = {
    "namibia": {
        "observation_level_asl_m": 2300,
        "earth_magnetic_field_x_muT": 12.5,
        "earth_magnetic_field_z_muT": -25.9,
        "atmosphere_id": 10,
        "geomagnetic_cutoff_rigidity_GV": 12.5,
    },
}

PARTICLES = {
    "gamma": {"particle_id": 1,},
}

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
    "max_num_airshowers_in_job": 100,
    "max_fraction_of_airshowers_to_collect_in_altitude_bin": 0.33,
    "num_jobs_in_energy_bin": 4,
}
