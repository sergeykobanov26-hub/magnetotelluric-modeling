"""Численное 1D решение прямой задачи МТЗ методом конечных разностей."""

import numpy as np

from .constants import mu


def numerical_solution_1d(rho_layers, h_layers, T_array):
    n_air = 10
    h_air_total = 110e3
    h_air = np.full(n_air, h_air_total / n_air)
    rho_air = np.full(n_air, 1e18)
    h_e = np.concatenate([h_air, h_layers])
    rho_e = np.concatenate([rho_air, rho_layers])
    N_e = len(h_e) - 1

    Zxy = np.zeros(len(T_array), dtype=complex)
    Zyx = np.zeros(len(T_array), dtype=complex)
    rho_xy = np.zeros(len(T_array))
    rho_yx = np.zeros(len(T_array))
    phi_xy = np.zeros(len(T_array))
    phi_yx = np.zeros(len(T_array))

    for idx, T in enumerate(T_array):
        omega = 2 * np.pi / T

        # H-поляризация (Zyx)
        N_h = len(rho_layers) - 1
        A = np.zeros((N_h + 1, N_h + 1), dtype=complex)
        F = np.zeros(N_h + 1, dtype=complex)
        A[0, 0] = 1.0; F[0] = 1.0
        A[N_h, N_h] = 1.0; F[N_h] = 0.0
        for i in range(1, N_h):
            A[i, i-1] = rho_layers[i-1] / ((h_layers[i-1] + h_layers[i]) * h_layers[i])
            A[i, i]   = (1j * omega * mu
                         - (rho_layers[i] / h_layers[i+1] + rho_layers[i-1] / h_layers[i])
                         / (h_layers[i-1] + h_layers[i]))
            A[i, i+1] = rho_layers[i] / ((h_layers[i-1] + h_layers[i]) * h_layers[i+1])
        Hx = np.linalg.solve(A, F)
        dHx_dz = (Hx[1] - Hx[0]) / h_layers[0]
        Ey = rho_layers[0] * dHx_dz
        Zyx[idx] = -Ey / Hx[0]

        # E-поляризация (Zxy)
        A = np.zeros((N_e + 1, N_e + 1), dtype=complex)
        F = np.zeros(N_e + 1, dtype=complex)
        A[0, 0] = 1.0; F[0] = 1.0
        A[N_e, N_e] = 1.0; F[N_e] = 0.0
        for i in range(1, N_e):
            A[i, i-1] = 1.0 / ((h_e[i-1] + h_e[i]) * h_e[i])
            A[i, i]   = (1j * omega * mu / rho_e[i]
                         - (1.0 / h_e[i+1] + 1.0 / h_e[i]) / (h_e[i-1] + h_e[i]))
            A[i, i+1] = 1.0 / ((h_e[i-1] + h_e[i]) * h_e[i+1])
        Ex = np.linalg.solve(A, F)
        dEx_dz = (Ex[n_air + 1] - Ex[n_air]) / h_layers[0]
        Hy = dEx_dz / (1j * omega * mu)
        Zxy[idx] = Ex[n_air] / Hy

        rho_xy[idx] = np.abs(Zxy[idx])**2 / (omega * mu)
        phi_xy[idx] = np.arctan2(np.imag(Zxy[idx]), np.real(Zxy[idx])) * 180 / np.pi
        rho_yx[idx] = np.abs(Zyx[idx])**2 / (omega * mu)
        phi_yx[idx] = np.arctan2(np.imag(Zyx[idx]), np.real(Zyx[idx])) * 180 / np.pi

    return rho_xy, phi_xy, Zxy, rho_yx, phi_yx, Zyx