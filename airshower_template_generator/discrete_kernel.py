import numpy as np

GAUSS_5X5 = (1.0 / 273.0) * np.array(
    [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ]
)

USEFUL_SIGMA = 1.92


def gauss1d(num_steps, edge_sigma=USEFUL_SIGMA):
    x = np.linspace(-edge_sigma, edge_sigma, num_steps)
    z = np.exp(-0.5 * x ** 2)
    z = z / np.sum(z)
    return z


def gauss2d(num_steps, edge_sigma=USEFUL_SIGMA):
    kernel_1d = gauss1d(
        num_steps=num_steps,
        edge_sigma=edge_sigma,
    )
    kernel_1d = kernel_1d / np.max(kernel_1d)
    out = np.zeros(shape=(num_steps, num_steps))
    for ix in range(num_steps):
        for iy in range(num_steps):
            out[ix, iy] = kernel_1d[ix] * kernel_1d[iy]
    out = out / np.sum(out)
    return out
