"""
tweakfig.tar_model_func
=======================

Model functions and analysis utilities for Tunneling Andreev Reflection (TAR).
These are specialized research functions that may require domain-specific dependencies.

Note: Some functions require external libraries like pyqula for band structure calculations.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter as savgol
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter


# =============================================================================
# Band Structure Utilities (requires pyqula)
# =============================================================================

def get_sc(h, nk=100):
    """
    Extract the superconducting order parameter from a Hamiltonian.
    
    Parameters
    ----------
    h : Hamiltonian object
        Hamiltonian from pyqula library.
    nk : int
        Number of k-points for reciprocal space sampling.
    
    Returns
    -------
    ndarray
        Superconducting order parameter in reciprocal space.
    
    Note
    ----
    Requires the pyqula library.
    """
    fk = h.get_hk_gen()
    dref = fk(np.random.random(3))[0, 2]
    dref = dref / np.abs(dref)
    
    def f(k):
        m = fk(k)
        return (m[0, 2] / dref).real
    
    from pyqula import spectrum
    (kxy, sc) = spectrum.reciprocal_map(h, f, nk=nk)
    return sc


def get_band_structure(h, nk=100, en=0, delta=0.1):
    """
    Compute band structure / Fermi surface.
    
    Parameters
    ----------
    h : Hamiltonian object
        Hamiltonian from pyqula library.
    nk : int
        Number of k-points.
    en : float
        Energy level.
    delta : float
        Smearing parameter.
    
    Returns
    -------
    ndarray
        Band structure data reshaped to (nk, nk) grid.
    """
    (x, y, d) = h.get_fermi_surface(e=en, delta=delta, nk=nk)
    return d.reshape((nk, nk))


# =============================================================================
# Conductance Calculations
# =============================================================================

def calc_conductance(G, **kwargs):
    """
    Calculate conductance components from raw data.
    
    Parameters
    ----------
    G : ndarray
        Conductance array with shape (..., 2).
    
    Returns
    -------
    dict
        Components: 'total', 'electron', 'Andreev', 'GA_GT'.
    """
    return {
        "total": G[:, :, 0] / 2,
        "electron": (G[:, :, 0] - 2 * G[:, :, 1]) / 2,
        "Andreev": G[:, :, 1],
        "GA_GT": 2 * G[:, :, 1] / G[:, :, 0],
    }


# =============================================================================
# Kappa Calculations
# =============================================================================

def calc_kappa_all(G_T, x, thresh=1e-4):
    """
    Calculate kappa (logarithmic derivative) for all energies.
    
    Parameters
    ----------
    G_T : ndarray
        Conductance array.
    x : ndarray
        Reference values for normalization.
    thresh : float
        Threshold below which values are masked.
    
    Returns
    -------
    ndarray
        Kappa values with NaN for low-conductance regions.
    """
    dlogx = savgol(np.log(x), axis=0, window_length=5, polyorder=2, deriv=1)
    dlogy = savgol(np.log(G_T), axis=0, window_length=5, polyorder=2, deriv=1)
    kappa = np.apply_along_axis(lambda arr: arr / dlogx, 0, dlogy)
    return np.where(G_T > thresh, kappa, np.nan)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_conductance(G_plot, d2, figax=None, noxlabel=False):
    """
    Plot conductance vs energy.
    
    Parameters
    ----------
    G_plot : ndarray
        Conductance data.
    d2 : dict
        Dictionary with 'es' (energies) and 'delta' (gap).
    figax : tuple, optional
        (fig, axes) tuple. Creates new figure if None.
    noxlabel : bool
        Suppress x-axis labels if True.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    es = d2["es"]
    delta = d2["delta"]

    if figax is None:
        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(10, 3))
    else:
        fig, axes = figax

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
    Plot kappa components.
    
    Parameters
    ----------
    G_p : ndarray
        Conductance data.
    d2 : dict
        Dictionary with 'es' and 'delta'.
    figax : tuple, optional
        (fig, axes) tuple.
    refwidth : float
        Reference width for figure.
    
    Returns
    -------
    list
        [fig, results_dict]
    """
    es = d2["es"]
    delta = d2["delta"]

    if figax is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    else:
        fig, axes = figax

    kappa = calc_kappa_all(G_p[:, :, 0], G_p[:, 0, 0])
    kappa_A = calc_kappa_all(G_p[:, :, 1], G_p[:, 0, 0])
    kappa_N = calc_kappa_all((G_p[:, :, 0] - 2 * G_p[:, :, 1]), G_p[:, 0, 0])

    results = {"kappa": kappa, "kappa_A": kappa_A, "kappa_N": kappa_N}
    n_half = kappa.shape[1] // 2

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

