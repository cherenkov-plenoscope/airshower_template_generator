import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from . import query
from . import bins

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def axis_size(x_start, x_stop, y_start, y_stop, figsize, dpi):
    num_cols = figsize[0] * dpi
    num_rows = figsize[1] * dpi
    x_s_rel = x_start / num_cols
    x_e_rel = x_stop / num_cols
    y_s_rel = y_start / num_rows
    y_e_rel = y_stop / num_rows
    out = (x_s_rel, y_s_rel, x_e_rel - x_s_rel, y_e_rel - y_s_rel)
    return out


def add_circle(ax, x, y, r, linestyle):
    phis = np.linspace(0, 2 * np.pi, 512)
    xs = r * np.cos(phis) + x
    ys = r * np.sin(phis) + y
    ax.plot(xs, ys, linestyle)


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


def write_view(
    path,
    energy_GeV,
    altitude_m,
    azimuth_deg,
    radius_m,
    image_integrated=None,
    lut=None,
    binning=None,
    num_airshowers=None,
):

    if image_integrated is None:
        image_integrated = query.query_image(
            lut=lut,
            energy_GeV=energy_GeV,
            altitude_m=altitude_m,
            azimuth_deg=azimuth_deg,
            radius_m=radius_m,
        )
        _b = lut["binning"]
        num_showers = np.array(lut["num_airshowers"])
        num_photons = np.array(num_showers)
    else:
        num_showers = num_airshowers
        num_photons = np.array(num_showers)
        _b = binning

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
    img_height = 0.5

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax_img = fig.add_axes(
        axis_size(75, 1920 - 75, 480, 480 + ((1920 - 150) / 3), figsize, dpi)
    )
    ax_img.pcolor(
        explbins["image_parallel_deg"]["edges"],
        explbins["image_perpendicular_deg"]["edges"],
        image_integrated.T,
        cmap="inferno",
    )
    rm_splines(ax=ax_img)
    ax_img.grid(color="white", linestyle="-", linewidth=0.66, alpha=0.3)
    ax_img.set_xlabel("radial / $^{\\circ}$")
    ax_img.set_yticks(np.linspace(-0.5, 0.5, 5))
    ax_img.set_xticks(np.linspace(-0.5, 2.5, 13))

    ax_aperture = fig.add_axes(
        axis_size(75, 75 + 330, 75, 75 + 330, figsize, dpi)
    )
    add_circle(ax=ax_aperture, x=0, y=0, r=max_core_radius, linestyle="k:")
    rm_splines(ax=ax_aperture)
    ax_aperture.set_xlim([-max_core_radius, max_core_radius])
    ax_aperture.set_ylim([-max_core_radius, max_core_radius])
    ax_aperture.set_yticklabels([])

    add_circle(
        ax=ax_aperture,
        x=aperture_x,
        y=aperture_y,
        r=aperture_radius_m,
        linestyle="k-",
    )
    ax_aperture.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_aperture.set_xlabel("x / m")
    ax_aperture.set_ylabel("y / m")

    ax_ap_text = fig.add_axes(
        axis_size(75 + 330, 75 + 330 + 300, 75, 75 + 330, figsize, dpi)
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
        axis_size(880, 880 + 330, 75, 75 + 330, figsize, dpi)
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
        axis_size(680, 680 + 60, 75, 75 + 330, figsize, dpi)
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

    ax_size = fig.add_axes(axis_size(1300, 1350, 75, 75 + 330, figsize, dpi))
    add_slider_axes(
        ax=ax_size,
        start=1e3,
        stop=1e9,
        value=image_integrated_size,
        log=True,
        label="photons / m$^{{-2}}$ sr$^{{-1}}$\n{:.0f}k".format(
            image_integrated_size / 1e3
        ),
    )

    ax_altitude_slider = fig.add_axes(
        axis_size(1450, 1500, 75, 75 + 330, figsize, dpi)
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
        axis_size(1600, 1920 - 275, 75, 75 + 330, figsize, dpi)
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
    p.append([0.5, 12.5e3, 0.0, 0.0])
    p.append([0.5, 12.5e3, 0.0, 250.0])
    p.append([0.5, 12.5e3, 0.0, 75.0])
    p.append([20.0, 12.5e3, 0.0, 75.0])
    p.append([20.0, 7.5e3, 0.0, 75.0])
    p.append([20.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 12.5e3, 360.0, 290.0])
    p.append([5.0, 12.5e3, 0.0, 75.0])
    p.append([5.0, 15.5e3, 0.0, 75.0])
    p.append([5.0, 10.5e3, 0.0, 75.0])
    p.append([0.5, 12.5e3, 0.0, 0.0])
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
            "num_airshowers": lut["num_airshowers"],
            "image_integrated": query.query_image(
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
        num_airshowers=job["num_airshowers"],
    )
