"""Функции визуализации для проекта МТЗ."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def plot_model(rho, z_edges):
    ny, nz = rho.shape
    y_edges = np.arange(ny + 1) - 0.5
    Y, Z = np.meshgrid(y_edges, z_edges)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(Y, Z, np.log10(rho.T), cmap='jet', shading='flat')
    plt.colorbar(label='log₁₀(ρ, Ом·м)')
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.xlabel('Номер пикета')
    plt.ylabel('Глубина, м')
    plt.title('Геоэлектрический разрез: модель грабена')
    plt.tight_layout()
    plt.show()


def plot_spacetime_single(ax, T, x_vals, data, title, cmap='jet',
                          log_norm=True, vmin=None, vmax=None, cbar_label=''):
    if log_norm:
        data_pos = np.abs(data)
        data_pos[data_pos < 1e-10] = 1e-10
        norm = LogNorm()
    else:
        data_pos = data
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(x_vals, T, data_pos, norm=norm, cmap=cmap, shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel('Пикет')
    ax.set_ylabel('Период, с')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label=cbar_label)    