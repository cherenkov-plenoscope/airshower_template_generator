import numpy as np
import json
import skimage
import sklearn
from sklearn import linear_model
from . import query
from . import bins
from . import projection

SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def fit_ellipse(cx, cy, ts, find_time_gradient=False):
    center_x = np.median(cx)
    center_y = np.median(cy)

    ccx = cx - center_x
    ccy = cy - center_y

    rots = np.zeros(360)
    azis = np.linspace(0.0, np.pi, 360, endpoint=False)
    for ii, img_azi in enumerate(azis):
        cos = np.cos(img_azi)
        sin = np.sin(img_azi)
        ccax = ccx * cos - ccy * sin
        rots[ii] = np.std(ccax)
    wheremin = np.argmin(rots)
    wheremax = np.argmax(rots)

    min_azi = azis[wheremin]
    cos = np.cos(img_azi)
    sin = np.sin(img_azi)
    ccax = ccx * cos - ccy * sin

    if find_time_gradient:
        try:
            rr = sklearn.linear_model.RANSACRegressor()
            rr.fit(X=ccax.reshape([ccax.shape[0], 1]), y=ts)
            time_slope_slices_per_rad = rr.estimator_.coef_[0]
            time_slope_ns_per_deg = 0.5*time_slope_slices_per_rad/np.rad2deg(1)
        except ValueError:
            time_slope_ns_per_deg = 0.0
        print("time_slope: ", time_slope_ns_per_deg, "ns/deg")
    else:
        time_slope_ns_per_deg = 0.0

    return [center_x, center_y], azis[wheremin], rots[wheremin], rots[wheremax], time_slope_ns_per_deg


def fit_ellipse_using_covariance_matrix(cx, cy, ts, find_time_gradient=False):
    center_x = np.median(cx)
    center_y = np.median(cy)

    cov_matrix = np.cov(np.c_[cx, cy].T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_vals)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0
    major_axis = eigen_vecs[:, major_idx]
    major_std = np.sqrt(eigen_vals[major_idx])
    minor_axis = eigen_vecs[:, minor_idx]
    minor_std = np.sqrt(eigen_vals[minor_idx])

    azimuth = np.arctan2(major_axis[0], major_axis[1])
    time_slope_ns_per_deg = 0.0

    return [center_x, center_y], azimuth, major_std, minor_std, time_slope_ns_per_deg



IMAGE_BINNING = {
    "radius_deg": 4.25,
    "num_bins": 128,
}

def draw_line(line_model, image_binning=IMAGE_BINNING):
    radius = np.deg2rad(image_binning["radius_deg"])
    cen_x = line_model[0][0]
    cen_y = line_model[0][1]
    azi = line_model[1]

    time_slope_ns_per_deg = line_model[4]
    if time_slope_ns_per_deg >= 0.0:
        azi = azi
    else:
        azi =  azi + np.pi

    off_x = radius * np.sin(azi)
    off_y = radius * np.cos(azi)
    start_x = cen_x + off_x
    start_y = cen_y + off_y

    if np.abs(time_slope_ns_per_deg) < 1.0:
        stop_x = cen_x - off_x
        stop_y = cen_y - off_y
    else:
        stop_x = cen_x
        stop_y = cen_y

    pix_per_rad = image_binning["num_bins"]/(2.0*radius)
    center_px = image_binning["num_bins"]//2

    start_x_px = int(np.round(start_x * pix_per_rad)) + center_px
    start_y_px = int(np.round(start_y * pix_per_rad)) + center_px
    stop_x_px = int(np.round(stop_x * pix_per_rad)) + center_px
    stop_y_px = int(np.round(stop_y * pix_per_rad)) + center_px

    rr, cc, aa = skimage.draw.line_aa(
        r0=start_y_px, c0=start_x_px, r1=stop_y_px, c1=stop_x_px
    )

    valid_rr = np.logical_and((rr >= 0), (rr < image_binning["num_bins"]))
    valid_cc = np.logical_and((cc >= 0), (cc < image_binning["num_bins"]))
    valid = np.logical_and(valid_rr, valid_cc)

    return rr[valid], cc[valid], aa[valid]


def make_image(split_light_field, image_binning=IMAGE_BINNING):
    slf = split_light_field
    line_models = []
    for pax in range(slf.number_paxel):
        img = slf.image_sequences[pax]
        num_ph = img.shape[0]
        if num_ph > 2:
            line_models.append(
                (
                    fit_ellipse_using_covariance_matrix(cx=img[:, 0], cy=img[:, 1], ts=img[:, 2]),
                    float(num_ph)
                )
            )

    out = np.zeros(shape=(image_binning["num_bins"], image_binning["num_bins"]))
    for line_model in line_models:
        rr, cc, aa = draw_line(
            line_model=line_model[0],
            image_binning=image_binning
        )
        out[rr, cc] += aa * line_model[1]
    return out


def argmax_image_cx_cy_deg(image, image_binning=IMAGE_BINNING):
    ib = image_binning
    _cxcy_bin_edges = np.linspace(
        -ib["radius_deg"],
        ib["radius_deg"],
        ib["num_bins"] + 1
    )
    _cxcy_bin_centers = 0.5 * (_cxcy_bin_edges[0:-1] + _cxcy_bin_edges[1:])
    _resp = np.unravel_index(np.argmax(image), image.shape)
    reco_cx_deg = _cxcy_bin_centers[_resp[1]]
    reco_cy_deg = _cxcy_bin_centers[_resp[0]]
    return reco_cx_deg, reco_cy_deg


def add_image_to_ax(ax, image, image_binning=IMAGE_BINNING):
    ib = image_binning
    assert image.shape[0] == ib["num_bins"]
    assert image.shape[1] == ib["num_bins"]

    c_bin_edges = np.linspace(
        -ib["radius_deg"], ib["radius_deg"], ib["num_bins"] + 1
    )
    return ax.pcolor(c_bin_edges, c_bin_edges, image)


def write_img(path, loph, light_field_geometry, image_binning=IMAGE_BINNING,
    true_cx=None, true_cy=None, true_x=None, true_y=None):
    lfg = light_field_geometry
    fov_radius_deg = np.rad2deg(
        0.5*lfg.sensor_plane2imaging_system.max_FoV_diameter
    )
    ib = image_binning

    slf = SplitLightField(
        loph_record=loph,
        light_field_geometry=lfg
    )
    img = make_image(split_light_field=slf, image_binning=ib)

    scale = 1.5
    fig = plt.figure(figsize=(16/scale, 9/scale), dpi=100*scale)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for pax in range(slf.number_paxel):
        ax.plot(
            np.rad2deg(slf.image_sequences[pax][:, 0]),
            np.rad2deg(slf.image_sequences[pax][:, 1]),
            "xb",
            alpha=0.03
        )

    c_bin_edges = np.linspace(
        -ib["radius_deg"], ib["radius_deg"], ib["num_bins"] + 1
    )
    ax.pcolor(c_bin_edges, c_bin_edges, img, cmap="Reds")

    phi = np.linspace(0, 2*np.pi, 1000)
    ax.plot(fov_radius_deg*np.cos(phi), fov_radius_deg*np.sin(phi), "k")

    info_str = "reco. Cherenkov: {: 4d} p.e.".format(
        loph["photons"]["channels"].shape[0]
    )

    ax.set_title(
        "reco. Cherenkov: {: 4d}p.e.".format(
            loph["photons"]["channels"].shape[0],
        )
    )

    ax.set_aspect("equal")
    ax.set_xlabel("cx / deg")
    ax.set_ylabel("cy / deg")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

    fig.savefig(path)
    plt.close(fig)


def gaussian_bell_1d(c_deg, peak_deg, width_deg):
    return (
        gaussian_bell_1d_max_one(c_deg, peak_deg, width_deg) /
        (width_deg * SQRT_TWO_PI)
    )

def gaussian_bell_1d_max_one(c_deg, peak_deg, width_deg):
    return np.exp(-0.5 * (c_deg - peak_deg) ** 2 / width_deg ** 2)


def lorentz_transversal(c_deg, peak_deg, width_deg):
    return width_deg / (np.pi * (width_deg ** 2 + (c_deg - peak_deg) ** 2))


def lorentz_moyal_longitidinal(
    c_deg, peak_deg, width_deg,
):
    lam = (c_deg - peak_deg) / width_deg
    return (
        1.0 / (width_deg * SQRT_TWO_PI) * np.exp(-0.5 * (lam + np.exp(-lam)))
    )


def my_moyal_model(c_deg, moyal_peak_deg, tail_direction, width_deg):
    moyal_density = (1.0 - tail_direction) * lorentz_moyal_longitidinal(
        c_deg=c_deg, peak_deg=moyal_peak_deg, width_deg=width_deg,
    )
    gaussian_density = tail_direction * lorentz_moyal_longitidinal(
        c_deg=-c_deg, peak_deg=-moyal_peak_deg, width_deg=width_deg,
    )
    return moyal_density + gaussian_density


def read_bell_model_lut(path):
    with open(path, "rt") as fin:
        dd = json.loads(fin.read())
    lut = dd
    lut["explicit_binning"] = bins.make_explicit_binning(lut["binning"])
    lut["bell_par.ene_azi_rad_alt"] = np.array(lut["bell_par.ene_azi_rad_alt"])
    lut["bell_per.ene_azi_rad_alt"] = np.array(lut["bell_per.ene_azi_rad_alt"])
    lut["population.ene_alt"] = np.array(lut["population.ene_alt"])
    lut["max_rad_at.ene_azi_alt"] = np.array(lut["max_rad_at.ene_azi_alt"])
    return lut


def query_bell_model(
    bell_model_lut, energy_GeV, azimuth_deg, radius_m, altitude_m
):
    lut = bell_model_lut
    b = bins.find_bins(
        explicit_binning=lut["explicit_binning"],
        energy_GeV=energy_GeV,
        altitude_m=altitude_m,
        azimuth_deg=azimuth_deg,
        radius_m=radius_m,
    )

    avg_par = np.zeros(2, dtype=np.float32)
    avg_per = np.zeros(2, dtype=np.float32)

    lutpar = lut["bell_par.ene_azi_rad_alt"]
    lutper = lut["bell_per.ene_azi_rad_alt"]

    weights = []
    for ene in b["energy_GeV"]:
        for alt in b["altitude_m"]:

            assert lut["population.ene_alt"][ene["bin"], alt["bin"]]

            for azi in b["azimuth_deg"]:
                for rad in b["radius_m"]:

                    assert (
                        rad["bin"]
                        <= lut["max_rad_at.ene_azi_alt"][
                            ene["bin"], azi["bin"], alt["bin"]
                        ]
                    )

                    slice_par = lutpar[
                        ene["bin"], azi["bin"], rad["bin"], alt["bin"]
                    ]
                    slice_per = lutper[
                        ene["bin"], azi["bin"], rad["bin"], alt["bin"]
                    ]

                    weight = (
                        ene["weight"]
                        * alt["weight"]
                        * azi["weight"]
                        * rad["weight"]
                    )
                    weights.append(weight)
                    avg_par = avg_par + slice_par * weight
                    avg_per = avg_per + slice_per * weight
    sum_weights = np.sum(weights)
    return (avg_par / sum_weights, avg_per / sum_weights)



class DiscPotential:
    def __init__(self, r0, r1):
        assert r0 > 0.0
        assert r1 > r0
        self.r0 = r0
        self.r1 = r1

    def potential(self, r):
        if r > self.r0:
            return ((r - self.r0) / (self.r1 - self.r0)) ** 2
        else:
            return 0.0

    def activation(self, r):
        if r <= self.r0:
            return 0.0
        elif r > self.r1:
            return 1.0
        else:
            gap = self.r1 - self.r0
            return (r - self.r0) / gap


class LightField:
    def __init__(
        self,
        cx,
        cy,
        x,
        y,
        time_slice,
    ):
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.time_slice = time_slice


class SplitLightField:
    def __init__(self, loph_record, light_field_geometry):
        lr = loph_record
        lfg = light_field_geometry
        self.number_photons = lr["photons"]["arrival_time_slices"].shape[0]

        self.number_paxel = lfg.number_paxel
        self.paxel_x = lfg.paxel_pos_x
        self.paxel_y = lfg.paxel_pos_y

        self.image_sequences = [[] for pax in range(self.number_paxel)]
        ph_pixel, ph_paxel = lfg.pixel_and_paxel_of_lixel(
            lixel=lr["photons"]["channels"]
        )
        ph_cx = lfg.cx_mean[lr["photons"]["channels"]]
        ph_cy = lfg.cy_mean[lr["photons"]["channels"]]
        self.median_cx = np.median(ph_cx)
        self.median_cy = np.median(ph_cy)
        ph_ts = lr["photons"]["arrival_time_slices"]

        for ph in range(self.number_photons):
            pax = ph_paxel[ph]
            self.image_sequences[pax].append([ph_cx[ph], ph_cy[ph], ph_ts[ph]])

        for pax in range(self.number_paxel):
            if len(self.image_sequences[pax]) > 0:
                self.image_sequences[pax] = np.array(self.image_sequences[pax])
            else:
                self.image_sequences[pax] = np.zeros(shape=(0, 3))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BellLightFieldFitter:
    def __init__(
        self,
        bell_model_lut,
        split_light_field,
        energy_GeV,
        altitude_m,
    ):
        self.bell_model_lut = bell_model_lut
        self.split_light_field = split_light_field
        self.energy_GeV = energy_GeV
        self.altitude_m = altitude_m

        ene_range = bins.find_energy_bins(
            explicit_binning=self.bell_model_lut["explicit_binning"],
            energy_GeV=energy_GeV
        )
        self.ene_bins = [ene_range["lower_bin"], ene_range["upper_bin"]]
        self.ene_weights = [ene_range["lower_weight"], ene_range["upper_weight"]]

        alt_range = bins.find_altitude_bins(
            explicit_binning=self.bell_model_lut["explicit_binning"],
            altitude_m=altitude_m
        )
        self.alt_bins = [alt_range["lower_bin"], alt_range["upper_bin"]]
        self.alt_weights = [alt_range["lower_weight"], alt_range["upper_weight"]]

        for ene in self.ene_bins :
            for alt in self.alt_bins:
                assert self.bell_model_lut["population.ene_alt"][ene, alt]


    def max_core_radius(self):

        min_rad = self.bell_model_lut["binning"]["radius_m"]["num_bins"] - 1

        for ene in self.ene_bins :
            for alt in self.alt_bins:
                for azi in range(self.bell_model_lut["binning"]["azimuth_deg"]["num_bins"]):
                    _max_rad = self.bell_model_lut["max_rad_at.ene_azi_alt"][
                        ene, azi, alt
                    ]
                    if _max_rad < min_rad:
                        min_rad = _max_rad

        return self.bell_model_lut["explicit_binning"]["radius_m"]["supports"][min_rad]


    def model(self, source_cx, source_cy, core_x, core_y):

        # query bell model for each paxel
        # -------------------------------
        num_ph = self.split_light_field.number_photons

        core_x_wrt_paxel = core_x - self.split_light_field.paxel_x
        core_y_wrt_paxel = core_y - self.split_light_field.paxel_y

        core_azimuth_rad = np.arctan2(core_y_wrt_paxel, core_x_wrt_paxel)
        core_azimuth_deg_wrt_paxel = np.rad2deg(core_azimuth_rad)
        core_radius_wrt_paxel = np.hypot(core_x_wrt_paxel, core_y_wrt_paxel)

        pax_bell_par = []
        pax_bell_per = []
        for pax in range(self.split_light_field.number_paxel):
            par, per = query_bell_model(
                bell_model_lut=self.bell_model_lut,
                energy_GeV=self.energy_GeV,
                azimuth_deg=core_azimuth_deg_wrt_paxel[pax],
                radius_m=core_radius_wrt_paxel[pax],
                altitude_m=self.altitude_m
            )
            pax_bell_par.append(par)
            pax_bell_per.append(per)
        pax_bell_par = np.array(pax_bell_par)
        pax_bell_per = np.array(pax_bell_per)

        # transform images into source-frame
        # ----------------------------------
        WRT_DOWNWARDS = -1.0
        pax_img_par_wrt_source = []
        pax_img_per_wrt_source = []
        for pax in range(self.split_light_field.number_paxel):
            ones = np.ones(self.split_light_field.image_sequences[pax].shape[0])
            c_para, c_perp = projection.project_light_field_onto_source_image(
                cer_cx_rad=WRT_DOWNWARDS * self.split_light_field.image_sequences[pax][:, 0],
                cer_cy_rad=WRT_DOWNWARDS * self.split_light_field.image_sequences[pax][:, 1],
                cer_x_m=ones * self.split_light_field.paxel_x[pax],
                cer_y_m=ones * self.split_light_field.paxel_y[pax],
                primary_cx_rad=WRT_DOWNWARDS * source_cx,
                primary_cy_rad=WRT_DOWNWARDS * source_cy,
                primary_core_x_m=core_x,
                primary_core_y_m=core_y,
            )
            pax_img_par_wrt_source.append(c_para)
            pax_img_per_wrt_source.append(c_perp)

        """
        c_par_bin_edges = np.linspace(-4.5, 4.5, 90+1)
        c_par_bin_centers = 0.5 * (c_par_bin_edges[0:-1] + c_par_bin_edges[1:])

        c_per_bin_edges = np.linspace(-4.5, 4.5, 90+1)
        c_per_bin_centers = 0.5 * (c_per_bin_edges[0:-1] + c_per_bin_edges[1:])

        # histograms
        # ----------
        pax_hist_par_wrt_source = []
        pax_hist_per_wrt_source = []
        for pax in range(self.split_light_field.number_paxel):
            h_par = np.histogram(
                a=np.rad2deg(pax_img_par_wrt_source[pax]),
                bins=c_par_bin_edges
            )[0]
            pax_hist_par_wrt_source.append(h_par)
            h_per = np.histogram(
                a=np.rad2deg(pax_img_per_wrt_source[pax]),
                bins=c_per_bin_edges
            )[0]
            pax_hist_per_wrt_source.append(h_per)
        pax_hist_par_wrt_source = np.array(pax_hist_par_wrt_source, dtype=np.float32)
        pax_hist_per_wrt_source = np.array(pax_hist_per_wrt_source, dtype=np.float32)

        # normalize histograms
        # --------------------
        for pax in range(self.split_light_field.number_paxel):

            h_par_sum = np.max(pax_hist_par_wrt_source[pax])
            if h_par_sum > 0:
                pax_hist_par_wrt_source[pax, :] *= (1/h_par_sum)

            h_per_sum = np.max(pax_hist_per_wrt_source[pax])
            if h_per_sum > 0:
                pax_hist_per_wrt_source[pax, :] *= (1/h_per_sum)


        # overlap with histograms
        # -----------------------
        w_par = 0.0
        w_per = 0.0
        for pax in range(self.split_light_field.number_paxel):
            hist_par_actual = pax_hist_par_wrt_source[pax]
            hist_par_model = gaussian_bell_1d_max_one(
                c_deg=c_par_bin_centers,
                peak_deg=pax_bell_par[pax, 0],
                width_deg=pax_bell_par[pax, 1]
            )
            #hist_par_model /= np.sum(hist_par_model)
            w_par += np.sum(np.abs(hist_par_actual - hist_par_model)) * len(pax_img_par_wrt_source[pax])

            hist_per_actual = pax_hist_per_wrt_source[pax]
            hist_per_model = gaussian_bell_1d_max_one(
                c_deg=c_per_bin_centers,
                peak_deg=pax_bell_per[pax, 0],
                width_deg=pax_bell_per[pax, 1]
            )
            #hist_per_model /= np.sum(hist_per_model)
            w_per += np.sum(np.abs(hist_per_actual - hist_per_model)) * len(pax_img_par_wrt_source[pax])

            if pax == 17:
                print("=============")
                print("num: ", len(pax_img_par_wrt_source[pax]))
                print("par peak/width deg: ", pax_bell_par[pax, 0], pax_bell_par[pax, 1])
                print("per peak/width deg: ", pax_bell_per[pax, 0], pax_bell_per[pax, 1])
                print(np.round(hist_par_actual, 2))
                print(np.round(hist_par_model, 2))

        w = np.log10((w_par + w_per))
        print(w)
        return w
        """

        # overlap with single photons
        # ---------------------------
        par_weights = 0
        per_weights = 0
        par_maxs = 0
        per_maxs = 0
        for pax in range(self.split_light_field.number_paxel):
            num_ph_pax = pax_img_par_wrt_source[pax].shape[0]
            _par_w = gaussian_bell_1d(
                c_deg=np.rad2deg(pax_img_par_wrt_source[pax]),
                peak_deg=pax_bell_par[pax, 0],
                width_deg=pax_bell_par[pax, 1]
            )
            _par_max = 1.0/(SQRT_TWO_PI * pax_bell_par[pax, 1])
            par_weights += (num_ph_pax * _par_max) - np.sum(_par_w)

            _per_w = gaussian_bell_1d(
                c_deg=np.rad2deg(pax_img_per_wrt_source[pax]),
                peak_deg=pax_bell_per[pax, 0],
                width_deg=pax_bell_per[pax, 1]
            )
            _per_max = 1.0/(SQRT_TWO_PI * pax_bell_per[pax, 1])
            per_weights += (num_ph_pax * _per_max) - np.sum(_per_w)

        w = (par_weights + per_weights) / (2.0 * num_ph)
        print("=============")
        print("w: ", w)

        return w


        """
        if w > 0.8:
            for pax in range(self.split_light_field.number_paxel):
                fig = plt.figure()
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax.set_aspect("equal")
                ax.plot(
                    np.rad2deg(pax_img_par_wrt_source[pax]),
                    np.rad2deg(pax_img_per_wrt_source[pax]),
                    "xk"
                )
                ax.set_xlabel("c_para / deg")
                ax.set_ylabel("c_perp / deg")
                ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
                fig.savefig("_projection_pax{:03d}".format(pax))
                plt.close(fig)
        """

