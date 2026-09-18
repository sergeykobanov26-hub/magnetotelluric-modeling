"""Аналитическое 1D решение прямой задачи МТЗ (рекуррентный метод)."""

import numpy as np

from .constants import mu


def analytical_solution_1d(rho_layers, h_layers, T_array):
    NT = len(T_array)
    Z = np.zeros(NT, dtype=complex)
    rho_T = np.zeros(NT)
    phi_T = np.zeros(NT)
    for idx, T in enumerate(T_array):
        omega = 2 * np.pi / T
        N = len(rho_layers)
        R = 1.0 + 0.0j
        for m in range(N - 2, -1, -1):
            k = np.sqrt(-1j * omega * mu / rho_layers[m])
            A = np.sqrt(rho_layers[m] / rho_layers[m + 1])
            B = np.exp(-2 * k * h_layers[m]) * (R - A) / (R + A)
            R = (1 + B) / (1 - B)
        k1 = np.sqrt(-1j * omega * mu / rho_layers[0])
        Z[idx] = -1j * omega * mu * R / k1
        rho_T[idx] = np.abs(Z[idx])**2 / (omega * mu)
        phi_T[idx] = np.arctan2(np.imag(Z[idx]), np.real(Z[idx])) * 180 / np.pi
    return rho_T, phi_T, Z