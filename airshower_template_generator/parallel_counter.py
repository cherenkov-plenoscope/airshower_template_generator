"""
Count the number of airshowers produced in a certain energy/altitude-bin during
parallel production using the filesystem.
"""
import os
import uuid


ENERGY_TMPLATE = "ene_{:06d}"
ALTITUDE_TMPLATE = "alt_{:06d}"


def init(counter_path, binning):
    os.makedirs(counter_path, exist_ok=True)

    for energy_bin_idx in range(binning["energy_GeV"]["num_bins"]):
        ene_name = ENERGY_TMPLATE.format(energy_bin_idx)
        os.makedirs(os.path.join(counter_path, ene_name), exist_ok=True)

        for altitude_bin_idx in range(binning["altitude_m"]["num_bins"]):
            alt_name = ALTITUDE_TMPLATE.format(altitude_bin_idx)
            os.makedirs(
                os.path.join(counter_path, ene_name, alt_name), exist_ok=True
            )


def increment(counter_path, energy_bin_idx, altitude_bin_idx):
    ene_name = ENERGY_TMPLATE.format(energy_bin_idx)
    alt_name = ALTITUDE_TMPLATE.format(altitude_bin_idx)
    count_uuid = uuid.uuid4().__str__()
    count_path = os.path.join(counter_path, ene_name, alt_name, count_uuid)
    with open(count_path, "w") as f:
        pass


def get(counter_path, energy_bin_idx, altitude_bin_idx):
    ene_name = ENERGY_TMPLATE.format(energy_bin_idx)
    alt_name = ALTITUDE_TMPLATE.format(altitude_bin_idx)
    bin_dir = os.path.join(counter_path, ene_name, alt_name)
    return len(os.listdir(bin_dir))
