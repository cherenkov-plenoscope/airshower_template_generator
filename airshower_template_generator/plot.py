import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_views(views, config):

    (
        c_para_bin_edges_deg,
        c_perp_bin_edges_deg,
    ) = _image_bin_edges_deg_para_perp(config["binning"])

    for azi in range(config["binning"]["azimuth_deg"]["num_bins"]):
        for rad in range(config["binning"]["radius_m"]["num_bins"]):
            for alt in range(config["binning"]["altitude_m"]["num_bins"]):

                fig = plt.figure()
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax.pcolormesh(
                    c_para_bin_edges_deg,
                    c_perp_bin_edges_deg,
                    views[azi, rad, alt].T,
                    cmap="inferno",
                )
                ax.set_aspect("equal")
                ax.set_xlabel("c para / deg")
                ax.set_ylabel("c perp / deg")
                fig.savefig(
                    "{:03d}_{:03d}_{:03d}_img.jpg".format(azi, rad, alt)
                )
                plt.close(fig)
