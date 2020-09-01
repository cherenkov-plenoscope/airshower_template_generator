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
    "gamma": {
        "particle_id": 1,
    },
}

BINNING = {
    "energy_GeV": {"start": 1e0, "stop": 2e0, "num_bins": 3},
    "azimuth_deg": {"start": 0.0, "stop": 240.0, "num_bins": 3},
    "radius_m": {"start": 0.0, "stop": 250.0, "num_bins": 25},
    "altitude_m": {"start": 5e3, "stop": 30e3, "num_bins": 25},
    "image_parallel_deg": {
        "start": -0.5,
        "stop": 2.5,
        "num_bins": 96,
    },
    "image_perpendicular_deg": {
        "start": -0.5,
        "stop": 0.5,
        "num_bins": 32,
    },
    "aperture_radius_m": 8.0,
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
