"""2D расчёт прямой задачи МТЗ через SimPEG (с fallback-эмуляцией)."""

import numpy as np

from .models import create_graben_model


def compute_2d_simpeg(rho_2d, dz, T_array, n_pickets):
    try:
        from discretize import TensorMesh
        from simpeg.electromagnetics.natural_source import (
            receivers, sources, Survey, Simulation3DPrimarySecondary)
        print("SimPEG обнаружен, выполняется 2D расчёт...")
    except ImportError:
        print("SimPEG не установлен. Используется эмуляция 2D решения.")
        return None, None

    try:
        nx = 3
        dx = [200.0] * nx
        dy = [200.0] * n_pickets
        dz_earth_rev = dz[::-1]
        n_air = 5
        h_air = 10000.0
        dz_air = [h_air] * n_air
        dz_total = np.r_[dz_earth_rev, dz_air]
        z_bottom = -np.sum(dz)
        mesh = TensorMesh([dx, dy, dz_total], origin=[0, 0, z_bottom])

        rho_earth_rev = rho_2d[:, ::-1].T
        sigma_earth = 1.0 / rho_earth_rev
        sigma_air_val = 1e-18
        sigma = np.ones((nx, n_pickets, len(dz_total))) * sigma_air_val
        sigma[:, :, :len(dz)] = sigma_earth.T[np.newaxis, :, :]

        rho_bg, _, _, _ = create_graben_model(n_pickets=n_pickets,
                                              depth=np.sum(dz), nz=len(dz),
                                              with_graben=False)
        rho_bg_rev = rho_bg[:, ::-1].T
        sigma_bg_earth = 1.0 / rho_bg_rev
        sigma_bg = np.ones_like(sigma) * sigma_air_val
        sigma_bg[:, :, :len(dz)] = sigma_bg_earth.T[np.newaxis, :, :]

        frequencies = 1.0 / T_array
        y_coords = np.arange(n_pickets) * dy[0]
        x_center = np.sum(dx) / 2.0
        rx_locs = np.array([[x_center, y, 0] for y in y_coords])
        rx_xy = receivers.Impedance(rx_locs, orientation='xy')
        rx_yx = receivers.Impedance(rx_locs, orientation='yx')
        rx_zy = receivers.Tipper(rx_locs, orientation='zy')
        receiver_list = [rx_xy, rx_yx, rx_zy]
        source_list = [sources.Planewave(receiver_list, frequency=freq) for freq in frequencies]
        survey = Survey(source_list)
        simulation = Simulation3DPrimarySecondary(
            mesh, survey=survey, sigma=sigma, sigmaPrimary=sigma_bg)
        print("Запуск 2D расчёта...")
        dpred = simulation.dpred()
        data = dpred.reshape((len(frequencies), 3, n_pickets))
        ix = nx // 2
        Zxy_2d = data[:, 0, ix, :]
        Zyx_2d = data[:, 1, ix, :]
        print("2D расчёт завершён.")
        return Zxy_2d, Zyx_2d
    except Exception as e:
        print(f"Ошибка 2D расчёта: {e}. Переход к эмуляции.")
        return None, None