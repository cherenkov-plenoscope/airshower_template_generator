import airshower_template_generator as atg
import numpy as np


def is_near(a, b, absolute=1e-6):
    return np.abs(a - b) < absolute


def test_zero():
    cpara, cperp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=0.0,
        cer_cy_rad=0.0,
        cer_x_m=0.0,
        cer_y_m=0.0,
        primary_cx_rad=0.0,
        primary_cy_rad=0.0,
        primary_core_x_m=0.0,
        primary_core_y_m=0.0,
    )
    assert cpara == 0.0
    assert cperp == 0.0


def test_xy_does_not_matter_when_pointing_to_zero():
    for cer_x in np.linspace(-1e3, 1e3, 10):
        for cer_y in np.linspace(-1e3, 1e3, 10):
            (
                cpara,
                cperp,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=0.0,
                cer_cy_rad=0.0,
                cer_x_m=cer_x,
                cer_y_m=cer_y,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=0.0,
            )
            assert is_near(cpara, 0.0)
            assert is_near(cperp, 0.0)


def test_core_xy_does_not_matter_when_pointing_to_zero():
    for core_x in np.linspace(-1e3, 1e3, 10):
        for core_y in np.linspace(-1e3, 1e3, 10):
            (
                cpara,
                cperp,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=0.0,
                cer_cy_rad=0.0,
                cer_x_m=0.0,
                cer_y_m=0.0,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=core_x,
                primary_core_y_m=core_y,
            )
            assert is_near(cpara, 0.0)
            assert is_near(cperp, 0.0)


def test_parallel_on_x_axis():
    """
               cY, c_perpendicular
               ^
          -----|------------
         |     |            |
         |     |            |
        -------0-----------------------> cX, c_parallel
         |                  |
         |                  |
          ------------------
    """
    for cer_cx in np.linspace(-0.1, 0.1, 10):
        for cer_cy in np.linspace(-0.1, 0.1, 10):
            (
                cpara,
                cperp,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=cer_cx,
                cer_cy_rad=cer_cy,
                cer_x_m=1.0,
                cer_y_m=0.0,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=0.0,
            )
            assert is_near(cpara, cer_cx)
            assert is_near(cperp, cer_cy)


def test_parallel_on_y_axis():
    """
               c_perpendicular
               ^
          -----|------------
         |     |            |
         |     |            |
        -------0-----------------------> cY, c_parallel
         |     |            |
         |     |            |
          -----|------------
               V
               cX
    """
    offset_y = 1.0
    for cer_cx in np.linspace(-0.1, 0.1, 10):
        for cer_cy in np.linspace(-0.1, 0.1, 10):
            (
                cpara,
                cperp,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=cer_cx,
                cer_cy_rad=cer_cy,
                cer_x_m=0.0,
                cer_y_m=offset_y,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=0.0,
            )
            assert is_near(cpara, cer_cy)
            assert is_near(cperp, -cer_cx)

            (
                cpara2,
                cperp2,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=cer_cx,
                cer_cy_rad=cer_cy,
                cer_x_m=0.0,
                cer_y_m=0.0,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=-1.0 * offset_y,
            )
            assert is_near(cpara, cpara2)
            assert is_near(cperp, cperp2)


def test_c_parallel_anitparallel_to_y_axis():
    """
               cX, c_perpendicular
               ^
          -----|------------
         |     |            |
         |     |            |
    cY <-------0-----------------------> c_parallel
         |                  |
         |                  |
          ------------------

    """
    offset_y = 1.0
    for cer_cx in np.linspace(-0.1, 0.1, 10):
        for cer_cy in np.linspace(-0.1, 0.1, 10):
            (
                cpara,
                cperp,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=cer_cx,
                cer_cy_rad=cer_cy,
                cer_x_m=0.0,
                cer_y_m=-1.0 * offset_y,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=0.0,
            )
            assert is_near(cpara, -cer_cy)
            assert is_near(cperp, cer_cx)

            (
                cpara2,
                cperp2,
            ) = atg.projection.project_light_field_onto_source_image(
                cer_cx_rad=cer_cx,
                cer_cy_rad=cer_cy,
                cer_x_m=0.0,
                cer_y_m=0.0,
                primary_cx_rad=0.0,
                primary_cy_rad=0.0,
                primary_core_x_m=0.0,
                primary_core_y_m=offset_y,
            )
            assert is_near(cpara, cpara2)
            assert is_near(cperp, cperp2)
