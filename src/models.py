"""Генерация синтетических 2D геоэлектрических моделей."""

import numpy as np


def create_graben_model(n_pickets=30, depth=8000, nz=49, with_graben=True):
    z_edges = np.logspace(0, np.log10(depth), nz + 1)
    z_edges[0] = 0.0
    dz = np.diff(z_edges)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    rho = np.full((n_pickets, nz), 500.0)   # фундамент

    for j in range(nz):
        if z_edges[j + 1] < 600:
            rho[:, j] = 200.0
        elif z_edges[j] < 1800:
            rho[:, j] = 50.0

    if with_graben:
        center = n_pickets // 2
        half_width = 6
        for i in range(center - half_width, center + half_width + 1):
            offset = 800 * (1 - abs(i - center) / half_width)
            for j in range(nz):
                if z_edges[j + 1] < 1800 + offset:
                    rho[i, j] = 5.0

    return rho, z_edges, z_centers, dz