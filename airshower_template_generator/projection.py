import numpy as np


def project_light_field_onto_source_image(
    cer_cx_rad,
    cer_cy_rad,
    cer_x_m,
    cer_y_m,
    primary_cx_rad,
    primary_cy_rad,
    primary_core_x_m,
    primary_core_y_m,
):
    cer_x_wrt_core = cer_x_m - primary_core_x_m
    cer_y_wrt_core = cer_y_m - primary_core_y_m

    cer_cx_wrt_primary = cer_cx_rad - primary_cx_rad
    cer_cy_wrt_primary = cer_cy_rad - primary_cy_rad

    azimuth = np.arctan2(cer_y_wrt_core, cer_x_wrt_core)
    derotate = -1.0 * azimuth

    cos_d = np.cos(derotate)
    sin_d = np.sin(derotate)

    cer_cpara = cos_d * cer_cx_wrt_primary - sin_d * cer_cy_wrt_primary
    cer_cperp = sin_d * cer_cx_wrt_primary + cos_d * cer_cy_wrt_primary

    return cer_cpara, cer_cperp
