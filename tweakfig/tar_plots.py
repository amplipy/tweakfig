"""
tweakfig.tar_plots
==================

Specialized plotting functions for Tunneling Andreev Reflection (TAR) analysis.
These are domain-specific utilities for visualizing conductance and kappa data.

Note: These functions are experimental and may require additional dependencies.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter as savgol
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter


# =============================================================================
# Color Utilities (shared with plots.py)
# =============================================================================

def colorFader(c1, c2, mix=0):
    """
    Linearly interpolate between two colors.
    
    Parameters
    ----------
    c1, c2 : color
        Start and end colors (any matplotlib format).
    mix : float
        Interpolation factor (0 = c1, 1 = c2).
    
    Returns
    -------
    str
        Hex color string.
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


# =============================================================================
# Conductance Calculations
# =============================================================================

def calc_conductance(G, **kwargs):
    """
    Calculate conductance components from raw conductance data.
    
    Parameters
    ----------
    G : ndarray
        Conductance array with shape (..., 2) where G[..., 0] is total 
        and G[..., 1] is Andreev component.
    
    Returns
    -------
    dict
        Dictionary with keys: 'total', 'electron', 'Andreev', 'GA_GT'.
    """
    return {
        "total": G[:, :, 0] / 2,
        "electron": (G[:, :, 0] - 2 * G[:, :, 1]) / 2,
        "Andreev": G[:, :, 1],
        "GA_GT": 2 * G[:, :, 1] / G[:, :, 0],
    }


def calc_kappa(G_T, en_ind=30, _plot=False):
    """
    Calculate kappa (logarithmic derivative) at a specific energy index.
    
    Parameters
    ----------
    G_T : ndarray
        Conductance array.
    en_ind : int
        Energy index for calculation.
    _plot : bool
        If True, create a plot of the result.
    """
    x = G_T[:, 0]
    y = G_T[:, en_ind]
    
    dlogx = savgol(np.log(x), window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(y), window_length=5, polyorder=2, deriv=1)
    
    kappa = dlogy / dlogx
    
    if _plot:
        fig, ax = plt.subplots()
        ax.plot(x, kappa)
        plt.show()
    
    return kappa


def calc_kappa_all(G_T, x, thresh=1e-4):
    """
    Calculate kappa for all energies.
    
    Computes dlog(G(z,E)) / dlog(G[z,0]) for the full energy range.
    
    Parameters
    ----------
    G_T : ndarray
        Conductance array.
    x : ndarray
        Reference conductance values.
    thresh : float
        Threshold below which to mask values as NaN.
    
    Returns
    -------
    ndarray
        Kappa values with masked low-conductance regions.
    """
    dlogx = savgol(np.log(x), axis=0, window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(G_T), axis=0, window_length=5, polyorder=2, deriv=1)
    kappa = np.apply_along_axis(lambda x: x / dlogx, 0, dlogy)
    return np.where(G_T > thresh, kappa, np.nan)


def calc_true_kappa(G_T, x):
    """
    Calculate kappa using true coupling coefficient.
    
    Parameters
    ----------
    G_T : ndarray
        Conductance array.
    x : ndarray
        True coupling coefficient (e.g., t^2 from calculations).
    
    Returns
    -------
    ndarray
        Normalized kappa values.
    """
    dlogx = savgol(x, axis=0, window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(G_T), axis=0, window_length=5, polyorder=2, deriv=1)
    unnormalized = np.apply_along_axis(lambda y: y / dlogx, 0, dlogy)
    kappa_normalized = np.apply_along_axis(lambda y: y / y[0], 1, unnormalized)
    return kappa_normalized


# =============================================================================
# Convenience Helpers
# =============================================================================

def normalize(x):
    """Normalize array by its first element."""
    return x / x[0]


_nred = normalize  # Alias for backward compatibility


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_conductance(G_plot, d2, figax=None, noxlabel=False):
    """
    Plot conductance components (total, electron, Andreev ratio).
    
    Parameters
    ----------
    G_plot : ndarray
        Array of conductance data to plot.
    d2 : dict
        Dictionary containing 'es' (energies) and 'delta' (gap).
    figax : tuple, optional
        (fig, axes) to plot on. Creates new figure if None.
    noxlabel : bool
        If True, suppress x-axis labels.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if created, None otherwise.
    """
    es = d2["es"]
    delta = d2["delta"]

    if figax is None:
        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(10, 3))
    else:
        fig, axes = figax

    # Color gradient
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0.1, 1, len(G_plot)))

    for color, G in zip(colors, G_plot):
        axes[0].semilogy(es / delta, G[:, 0], linewidth=1.5, color=color)
        axes[1].semilogy(es / delta, G[:, 0] - 2 * G[:, 1], color=color)
        axes[1].plot(es / delta, 2 * G[:, 1], color='red', alpha=0.2)
        axes[2].semilogy(es / delta, 2 * G[:, 1] / G[:, 0], color=color, alpha=0.4)

    if not noxlabel:
        for ax in axes:
            ax.set_xlabel(r"Energy ($\Delta$)", labelpad=3.0)

    axes[0].set_ylabel(r"$G \ (G_{0})$", labelpad=3.0)
    axes[1].set_ylabel(r"$G \ (G_{0})$", labelpad=3.0)
    axes[2].set_ylabel(r"$G_{A}/G$", labelpad=3.0)

    plt.tight_layout()
    
    if figax is None:
        return fig


def plot_kappas(G_p, d2, figax=None, refwidth=2.5):
    """
    Plot kappa components (total, Andreev, electron).
    
    Parameters
    ----------
    G_p : ndarray
        Conductance data array.
    d2 : dict
        Dictionary containing 'es' (energies) and 'delta' (gap).
    figax : tuple, optional
        (fig, axes) to plot on. Creates new figure if None.
    refwidth : float
        Reference width for figure sizing.
    
    Returns
    -------
    list
        [fig, results_dict] where results_dict contains computed kappa values.
    """
    es = d2["es"]
    delta = d2["delta"]
    
    if figax is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    else:
        fig, axes = figax

    # Calculate kappa components
    kappa = calc_kappa_all(G_p[:, :, 0], G_p[:, 0, 0])
    kappa_A = calc_kappa_all(G_p[:, :, 1], G_p[:, 0, 0])
    kappa_N = calc_kappa_all((G_p[:, :, 0] - 2 * G_p[:, :, 1]), G_p[:, 0, 0])
    
    results = {
        "kappa": kappa,
        "kappa_A": kappa_A,
        "kappa_N": kappa_N,
    }
    
    n_half = kappa.shape[1] // 2

    # Color gradient
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0.1, 1, len(G_p)))

    for color, kap_T, kap_A, kap_N, _G in zip(colors, kappa, kappa_A, kappa_N, G_p):
        ro = 2 * _G[:, 1] / _G[:, 0]
        comb = kap_A * ro + (1 - ro) * kap_N

        axes[0].plot(es / delta, kap_T, linewidth=1.5, color=color)
        axes[0].plot((es / delta)[n_half:], comb[n_half:], '--', 
                     color='red', linewidth=1.5, alpha=0.5)
        axes[1].plot(es / delta, kap_A, linewidth=1.5, color=color)
        axes[2].plot(es / delta, kap_N, linewidth=1.5, color=color)

    axes[0].set_xlabel(r"Energy ($\Delta$)", labelpad=3.0)
    axes[0].set_ylabel(r"$\kappa$", labelpad=3.0)
    axes[1].set_xlabel(r"Energy ($\Delta$)", labelpad=3.0)
    axes[1].set_ylabel(r"$\kappa_{A}$", labelpad=3.0)
    axes[2].set_xlabel(r"Energy ($\Delta$)", labelpad=3.0)
    axes[2].set_ylabel(r"$\kappa_{e}$", labelpad=3.0)

    plt.tight_layout()
    return [fig, results]
