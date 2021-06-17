from . import query
from . import bins
import numpy as np
import sebastians_matplotlib_addons as splt
import matplotlib.pyplot as plt
import os


def write_image(
    path,
    binning,
    image,
    x_key="image_parallel_deg",
    y_key="image_perpendicular_deg",
    figure_style=splt.FIGURE_16_9
):
    _b = binning
    explbins = bins.make_explicit_binning(_b)
    w_deg = _b[x_key]["num_bins"]
    h_deg = _b[y_key]["num_bins"]
    w = 0.8
    h = h_deg / w_deg * w * 16 / 9
    fig = splt.figure(figure_style)
    ax = splt.add_axes(fig=fig, span=[0.1, 0.1, w, h])
    ax.set_title("{:.3e} ph".format(np.sum(image)))
    ax.pcolor(
        explbins[x_key]["edges"],
        explbins[y_key]["edges"],
        image.T,
        cmap="inferno",
    )
    ax.grid(color="white", linestyle="-", linewidth=0.66, alpha=0.3)
    ax.set_xlabel("radial / $^{\\circ}$")
    plt.savefig(path)
    plt.close(fig)


def axis_size(x_start, x_stop, y_start, y_stop, figsize, dpi):
    num_cols = figsize[0] * dpi
    num_rows = figsize[1] * dpi
    x_s_rel = x_start / num_cols
    x_e_rel = x_stop / num_cols
    y_s_rel = y_start / num_rows
    y_e_rel = y_stop / num_rows
    out = (x_s_rel, y_s_rel, x_e_rel - x_s_rel, y_e_rel - y_s_rel)
    return out


def rm_splines(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def add_slider_axes(ax, start, stop, value, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    rm_splines(ax=ax)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(label)
    ax.plot([0, 1], [value, value], "k", linewidth=5)


def write_view(path, energy_GeV, altitude_m, azimuth_deg, radius_m, lut):
    """
    Writes an overview-figure of a slice in the look-up-table.
    Shows:
    -   Direction-histogram (c-parallel vs. c-perpendicular)
    -   Direction-time-histogram (c-parallel vs. rel. arrival-time)
    -   The position of the query in the look-up-table's space, and the
            population of the look-up-table.

    Parameters
    ----------
    path : str, path
            Output path to write figure to.
    energy_GeV : float
            The energy to look up.
    altitude_m : float
            The altitude of the airshower's maximum to look up.
    azimuth_deg : float
            The azimuth-angle of the shower's core w.r.t. the aperture-plane.
    radius_m : float
            The radial distance of the shower's core w.r.t. the aperture-plane.
    lut : dict
            The look-up-table.
    """
    image_integrated = query.query_par_per(
        lut=lut,
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )
    timage_integrated = query.query_par_tim(
        lut=lut,
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )
    _b = lut["binning"]
    num_showers = np.array(lut["airshower.histogram.ene_alt"])
    num_photons = np.array(num_showers)

    explbins = bins.make_explicit_binning(_b)
    b = bins.find_bins(
        explicit_binning=explbins,
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )

    lookup_population = 0.1 * (num_photons > 10)
    lookup_population_pos = lookup_population.copy()
    image_integrated_size = np.max(image_integrated)
    for ene in b["energy_GeV"]:
        for alt in b["altitude_m"]:
            lookup_population_pos[ene["bin"], alt["bin"]] += (
                0.9 * (ene["weight"] + alt["weight"]) / 2
            )

    aperture_radius_m = _b["aperture_radius_m"]
    max_core_radius = _b["radius_m"]["stop_support"]

    aperture_x = radius_m * np.cos(np.deg2rad(azimuth_deg))
    aperture_y = radius_m * np.sin(np.deg2rad(azimuth_deg))

    energy_bin_centers = explbins["energy_GeV"]["supports"]
    altitude_bin_edges = explbins["altitude_m"]["edges"]

    figsize = (16, 9)
    dpi = 120

    _img_para_span = (
        _b["image_parallel_deg"]["stop_edge"]
        - _b["image_parallel_deg"]["start_edge"]
    )
    _img_perp_span = (
        _b["image_perpendicular_deg"]["stop_edge"]
        - _b["image_perpendicular_deg"]["start_edge"]
    )

    _img_w = 0.9
    _img_h = _img_w * (_img_perp_span / _img_para_span)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # para vs. perp
    # =============
    ax_img = fig.add_axes([0.05, 0.67, _img_w, _img_h * (16 / 9)])
    ax_img.pcolor(
        explbins["image_parallel_deg"]["edges"],
        explbins["image_perpendicular_deg"]["edges"],
        image_integrated.T,
        cmap="inferno",
    )
    rm_splines(ax=ax_img)
    ax_img.grid(color="white", linestyle="-", linewidth=0.66, alpha=0.3)
    ax_img.set_xlabel("radial / $^{\\circ}$")

    # para vs. time
    # =============
    ax_tmg = fig.add_axes([0.05, 0.34, _img_w, _img_h * (16 / 9)])
    ax_tmg.pcolor(
        explbins["image_parallel_deg"]["edges"],
        explbins["time_s"]["edges"],
        timage_integrated.T,
        cmap="inferno",
    )
    rm_splines(ax=ax_tmg)
    ax_tmg.grid(color="white", linestyle="-", linewidth=0.66, alpha=0.3)
    ax_tmg.set_xlabel("radial / $^{\\circ}$")

    YS = 75
    ax_aperture = fig.add_axes(
        axis_size(75, 75 + 330, YS, 75 + 330, figsize, dpi)
    )
    splt.ax_add_circle(
        ax=ax_aperture,
        x=0,
        y=0,
        r=max_core_radius,
        linestyle=":",
        color="k"
    )
    rm_splines(ax=ax_aperture)
    ax_aperture.set_xlim([-max_core_radius, max_core_radius])
    ax_aperture.set_ylim([-max_core_radius, max_core_radius])
    ax_aperture.set_yticklabels([])

    splt.ax_add_circle(
        ax=ax_aperture,
        x=aperture_x,
        y=aperture_y,
        r=aperture_radius_m,
        linestyle="-",
        color="k"
    )
    ax_aperture.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_aperture.set_xlabel("x / m")
    ax_aperture.set_ylabel("y / m")

    ax_ap_text = fig.add_axes(
        axis_size(75 + 330, 75 + 330 + 300, YS, 75 + 330, figsize, dpi)
    )
    ax_ap_text.set_axis_off()
    ax_ap_text.text(0.1, 1.0, "x {:0.1f}m".format(aperture_x))
    ax_ap_text.text(0.1, 0.9, "y {:0.1f}m".format(aperture_y))
    ax_ap_text.text(
        0.1, 0.7, "azimuth {:0.1f}$^{{\\circ}}$".format(azimuth_deg)
    )
    ax_ap_text.text(0.1, 0.6, "radius {:0.1f}m".format(radius_m))
    ax_ap_text.text(
        0.1, 0.5, "apertur-radius {:0.1f}m".format(aperture_radius_m)
    )
    ax_ap_text.text(
        0.1,
        0.3,
        "azimuth-bin [{: 3d}, {: 3d}]".format(
            b["azimuth_deg"][0]["bin"], b["azimuth_deg"][1]["bin"],
        ),
    )
    ax_ap_text.text(
        0.1,
        0.2,
        "radius-bin [{: 3d}, {: 3d}]".format(
            b["radius_m"][0]["bin"], b["radius_m"][1]["bin"],
        ),
    )
    ax_ap_text.text(
        0.1,
        0.1,
        "energy-bin [{: 3d}, {: 3d}]".format(
            b["energy_GeV"][0]["bin"], b["energy_GeV"][1]["bin"],
        ),
    )
    ax_ap_text.text(
        0.1,
        0.0,
        "altitude-bin [{: 3d}, {: 3d}]".format(
            b["altitude_m"][0]["bin"], b["altitude_m"][1]["bin"],
        ),
    )

    # energy-altitude-population
    # --------------------------
    ax_population = fig.add_axes(
        axis_size(880, 880 + 330, YS, 75 + 330, figsize, dpi)
    )
    rm_splines(ax=ax_population)
    ax_population.pcolor(lookup_population_pos.T, cmap="binary", vmax=1.0)
    ax_population.set_xlabel("energy-bins")
    ax_population.set_ylabel("altitude-bins")

    ax_population.set_xticks(np.arange(len(energy_bin_centers)))
    ax_population.set_yticks(np.arange(len(altitude_bin_edges)))
    ax_population.set_xticklabels([])
    ax_population.set_yticklabels([])
    ax_population.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)

    # azimuth-radius-population
    # -------------------------
    az_ra_popu = np.zeros(
        shape=(_b["azimuth_deg"]["num_bins"], _b["radius_m"]["num_bins"])
    )
    for azi in b["azimuth_deg"]:
        for rad in b["radius_m"]:
            az_ra_popu[azi["bin"], rad["bin"]] += (
                0.9 * (azi["weight"] + rad["weight"]) / 2
            )
    ax_az_ra_popu = fig.add_axes(
        axis_size(680, 680 + 60, YS, 75 + 330, figsize, dpi)
    )
    rm_splines(ax=ax_az_ra_popu)
    ax_az_ra_popu.pcolor(az_ra_popu.T, cmap="binary", vmax=1.0)
    ax_az_ra_popu.set_xlabel("azimuth-bins")
    ax_az_ra_popu.set_ylabel("radius-bins")

    ax_az_ra_popu.set_xticks(np.arange(_b["azimuth_deg"]["num_bins"]))
    ax_az_ra_popu.set_yticks(np.arange(_b["radius_m"]["num_bins"]))
    ax_az_ra_popu.set_xticklabels([])
    ax_az_ra_popu.set_yticklabels([])
    ax_az_ra_popu.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.3)

    ax_size = fig.add_axes(axis_size(1300, 1350, YS, 75 + 330, figsize, dpi))
    add_slider_axes(
        ax=ax_size,
        start=1e3,
        stop=1e9,
        value=image_integrated_size,
        log=True,
        label="photons / m$^{{-2}}$ sr$^{{-1}}$\n{:.1e}".format(
            image_integrated_size
        ),
    )

    ax_altitude_slider = fig.add_axes(
        axis_size(1450, 1500, YS, 75 + 330, figsize, dpi)
    )
    add_slider_axes(
        ax=ax_altitude_slider,
        start=altitude_bin_edges[0] * 1e-3,
        stop=altitude_bin_edges[-1] * 1e-3,
        value=altitude_m * 1e-3,
        log=False,
        label="altitude / km\n{:.1f}".format(altitude_m * 1e-3),
    )

    ax_energy_slider = fig.add_axes(
        axis_size(1600, 1920 - 275, YS, 75 + 330, figsize, dpi)
    )
    add_slider_axes(
        ax=ax_energy_slider,
        start=energy_bin_centers[0],
        stop=energy_bin_centers[-1],
        value=energy_GeV,
        log=True,
        label="energy / GeV\n{:.1f}".format(energy_GeV),
    )
    fig.savefig(path)
    plt.close(fig)


def move_linear(view_stations, num_steps_per_station=60):
    num_steps = num_steps_per_station
    views = []
    station = 0
    while station < (len(view_stations) - 1):
        start = view_stations[station]
        stop = view_stations[station + 1]
        block_views = np.array(
            [
                np.geomspace(start[0], stop[0], num_steps),
                np.linspace(start[1], stop[1], num_steps),
                np.linspace(start[2], stop[2], num_steps),
                np.linspace(start[3], stop[3], num_steps),
            ]
        )
        views.append(block_views.T)
        station += 1
    return np.concatenate(views)


def example_view_path():
    p = []
    p.append([1.25, 12.5e3, 0.0, 0.0])
    p.append([1.25, 12.5e3, 0.0, 250.0])
    p.append([1.25, 12.5e3, 0.0, 75.0])
    p.append([9.0, 12.5e3, 0.0, 75.0])
    p.append([9.0, 7.5e3, 0.0, 75.0])
    p.append([9.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 12.5e3, 360.0, 290.0])
    p.append([5.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 15.5e3, 0.0, 75.0])
    p.append([5.0, 10.5e3, 0.0, 75.0])
    p.append([1.25, 12.5e3, 0.0, 0.0])
    return p


def make_jobs_walk(lut, out_dir, views, image_file_format="jpg"):
    jobs = []
    os.makedirs(out_dir, exist_ok=True)
    for idx, view in enumerate(views):
        print(idx)
        out_path = os.path.join(
            out_dir, "{:06d}.{:s}".format(idx, image_file_format)
        )
        job = {
            "path": out_path,
            "binning": lut["binning"],
            "energy_GeV": view[0],
            "altitude_m": view[1],
            "azimuth_deg": view[2],
            "radius_m": view[3],
            "airshower.histogram.ene_alt": lut["airshower.histogram.ene_alt"],
            "image_integrated": query.query_par_per(
                lut=lut,
                energy_GeV=view[0],
                altitude_m=view[1],
                azimuth_deg=view[2],
                radius_m=view[3],
            ),
        }
        jobs.append(job)
    return jobs


def run_job(job):
    write_view(
        path=job["path"],
        energy_GeV=job["energy_GeV"],
        altitude_m=job["altitude_m"],
        azimuth_deg=job["azimuth_deg"],
        radius_m=job["radius_m"],
        lut=None,
        binning=job["binning"],
        image_integrated=job["image_integrated"],
        num_airshowers_ene_alt=job["airshower.histogram.ene_alt"],
    )
