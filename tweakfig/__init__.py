"""
tweakfig - Quick matplotlib figure enhancement for publication-ready plots
===========================================================================

A lightweight library to make matplotlib figures more readable, prettier, 
and publication-ready with minimal effort.

Quick Start:
    >>> import tweakfig as tfp
    >>> tfp.Pretty()  # Apply global styling (alias for PrettyMatplotlib)
    >>> 
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> tfp.num_ticks(ax, xaxis=(5, 2), yaxis=(4, 5))  # Customize ticks

Main Functions:
    - Pretty / PrettyMatplotlib: Apply publication-ready rcParams globally
    - set_preset: Apply predefined style presets ('nature', 'presentation', etc.)
    - num_ticks: Control the number of major/minor ticks
    - set_spine_style: Control axis spine visibility and style
    - annotate_subplots: Add (a), (b), (c) labels to subplots
    - colorFader: Interpolate between two colors
    - plot_array: Plot multiple arrays with automatic color gradients
    - savefig_tight: Save with autocropping
    - autocrop_png: Remove transparent borders from saved figures
"""

__version__ = "0.2.0"

# Core plotting utilities
from .plots import (
    # Main styling
    PrettyMatplotlib,
    Pretty,
    set_preset,
    FIGURE_SIZES,
    
    # Tick control
    num_ticks,
    x_num_ticks,
    y_num_ticks,
    
    # Axes styling
    set_spine_style,
    set_aspect_ratio,
    annotate_subplots,
    
    # Color utilities
    colorFader,
    plot_array,
    colorbar,
    
    # Save utilities
    savefig_tight,
    autocrop_png,
    autocrop_png_whitespace,
    save_veusz,
    crop_whitespace,
    
    # Helpers
    normalize,
    
    # Figure adjustment
    adjust,
    scale_fonts,
)

__all__ = [
    # Styling
    "PrettyMatplotlib",
    "Pretty",
    "set_preset",
    "FIGURE_SIZES",
    
    # Ticks
    "num_ticks",
    "x_num_ticks", 
    "y_num_ticks",
    
    # Axes
    "set_spine_style",
    "set_aspect_ratio",
    "annotate_subplots",
    
    # Colors
    "colorFader",
    "plot_array",
    "colorbar",
    
    # Saving
    "savefig_tight",
    "autocrop_png",
    "autocrop_png_whitespace",
    "save_veusz",
    "crop_whitespace",
    
    # Helpers
    "normalize",
    
    # Figure adjustment
    "adjust",
    "scale_fonts",
]
