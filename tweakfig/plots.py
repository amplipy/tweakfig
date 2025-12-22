"""
tweakfig.plots
==============

A lightweight library for creating publication-ready matplotlib figures.
Provides simple functions to style plots with consistent fonts, ticks, and colors.

Main Functions:
    - PrettyMatplotlib(): Apply publication-ready rcParams globally
    - num_ticks(): Control the number of major/minor ticks on axes
    - colorFader(): Interpolate between two colors
    - plot_array(): Plot multiple arrays with automatic color gradients
    - autocrop_png(): Remove transparent/white borders from images
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator
from PIL import Image


# =============================================================================
# Global Style Configuration
# =============================================================================

def PrettyMatplotlib(
    fig_font_scale=1.0,
    minor_tick_size=0.5,
    major_tick_size=1.0,
    font_family='Arial',
    **kwargs
):
    """
    Apply publication-ready matplotlib rcParams globally.
    
    Call this once at the start of your script/notebook to set consistent
    styling for all subsequent figures.
    
    Parameters
    ----------
    fig_font_scale : float, default=1.0
        Scale factor for all font sizes. Use >1 for larger text.
    minor_tick_size : float, default=0.5
        Scale factor for minor tick length and width.
    major_tick_size : float, default=1.0
        Scale factor for major tick length and width.
    font_family : str, default='Arial'
        Font family for all text elements.
    **kwargs : dict
        Additional matplotlib rcParams to set (e.g., 'figure.dpi': 150).
    
    Examples
    --------
    >>> import tweakfig.plots as tfp
    >>> tfp.PrettyMatplotlib(fig_font_scale=1.2)
    >>> # All subsequent plots will use the new style
    """
    # Font sizes (base sizes scaled by fig_font_scale)
    mpl.rcParams["axes.labelsize"] = fig_font_scale * 14
    mpl.rcParams["axes.titlesize"] = fig_font_scale * 16
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['xtick.labelsize'] = fig_font_scale * 12
    mpl.rcParams['ytick.labelsize'] = fig_font_scale * 12
    mpl.rcParams["axes.labelpad"] = 0.7
    mpl.rcParams["legend.fontsize"] = fig_font_scale * 12

    # Tick sizes
    mpl.rcParams['xtick.major.size'] = major_tick_size * 6
    mpl.rcParams['xtick.minor.size'] = minor_tick_size * 6
    mpl.rcParams['ytick.major.size'] = major_tick_size * 6
    mpl.rcParams['ytick.minor.size'] = minor_tick_size * 6

    # Enable minor ticks on all sides
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.minor.top'] = True
    mpl.rcParams['ytick.minor.right'] = True
    mpl.rcParams['xtick.minor.bottom'] = True
    mpl.rcParams['ytick.minor.left'] = True

    # Tick widths
    mpl.rcParams['xtick.minor.width'] = minor_tick_size
    mpl.rcParams['ytick.minor.width'] = minor_tick_size
    mpl.rcParams['xtick.major.width'] = major_tick_size
    mpl.rcParams['ytick.major.width'] = major_tick_size

    # Apply any additional user-specified rcParams
    for key, value in kwargs.items():
        mpl.rcParams[key] = value


# Convenient alias
Pretty = PrettyMatplotlib


# =============================================================================
# Style Presets
# =============================================================================

# Common journal figure sizes (width in inches)
FIGURE_SIZES = {
    'single_column': (3.5, 2.5),    # Single column width
    'double_column': (7.0, 4.0),    # Double column width  
    'presentation': (10, 6),         # For slides
    'square': (4, 4),
    'golden': (6, 3.7),              # Golden ratio
}


def set_preset(preset='default'):
    """
    Apply a predefined style preset.
    
    Parameters
    ----------
    preset : str
        Preset name. Options:
        - 'default': Standard publication style
        - 'nature': Nature journal style
        - 'presentation': Larger fonts for slides
        - 'minimal': Clean, minimal style
    
    Examples
    --------
    >>> tfp.set_preset('nature')
    """
    presets = {
        'default': {
            'fig_font_scale': 1.0,
            'font_family': 'Arial',
        },
        'nature': {
            'fig_font_scale': 0.9,
            'font_family': 'Arial',
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
        },
        'presentation': {
            'fig_font_scale': 1.5,
            'font_family': 'Arial',
            'major_tick_size': 1.3,
        },
        'minimal': {
            'fig_font_scale': 1.0,
            'font_family': 'Helvetica',
            'axes.spines.top': False,
            'axes.spines.right': False,
        },
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Options: {list(presets.keys())}")
    
    PrettyMatplotlib(**presets[preset])


# =============================================================================
# Tick Control Functions
# =============================================================================

# =============================================================================
# Tick Control Functions
# =============================================================================

def num_ticks(ax, xaxis=(4, 4), yaxis=(4, 4), log_scale=False):
    """
    Set the number of major and minor ticks on both axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    xaxis : tuple of (int, int), default=(4, 4)
        (n_major, n_minor) for the x-axis. n_minor is the number of 
        minor ticks between each pair of major ticks.
    yaxis : tuple of (int, int), default=(4, 4)
        (n_major, n_minor) for the y-axis.
    log_scale : bool, default=False
        If True, use LogLocator for the y-axis (for log-scale plots).
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> num_ticks(ax, xaxis=(5, 2), yaxis=(4, 5))
    """
    # X-axis (always linear locator)
    ax.xaxis.set_major_locator(MaxNLocator(xaxis[0]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xaxis[1]))
    
    # Y-axis
    if log_scale:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=yaxis[0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=yaxis[1]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(yaxis[0]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(yaxis[1]))


def x_num_ticks(ax, xaxis=(4, 4)):
    """
    Set the number of major and minor ticks on the x-axis only.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    xaxis : tuple of (int, int), default=(4, 4)
        (n_major, n_minor) for the x-axis.
    """
    ax.xaxis.set_major_locator(MaxNLocator(xaxis[0]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xaxis[1]))


def y_num_ticks(ax, yaxis=(4, 4), log_scale=False):
    """
    Set the number of major and minor ticks on the y-axis only.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    yaxis : tuple of (int, int), default=(4, 4)
        (n_major, n_minor) for the y-axis.
    log_scale : bool, default=False
        If True, use LogLocator (for log-scale plots).
    """
    if log_scale:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=yaxis[0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=yaxis[1]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(yaxis[0]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(yaxis[1]))


# =============================================================================
# Color Utilities
# =============================================================================

def colorFader(c1, c2, mix=0):
    """
    Linearly interpolate between two colors.
    
    Parameters
    ----------
    c1 : color
        Starting color (at mix=0). Any matplotlib-compatible color format.
    c2 : color
        Ending color (at mix=1). Any matplotlib-compatible color format.
    mix : float, default=0
        Interpolation factor between 0 and 1.
    
    Returns
    -------
    str
        Hex color string representing the interpolated color.
    
    Examples
    --------
    >>> colorFader('red', 'blue', 0.5)
    '#800080'
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_array(arr, ax=None, x=None, cmap='viridis', color=None, 
               alpha=1.0, lw=1.5, linestyle='-'):
    """
    Plot multiple arrays with automatic color gradient from a colormap.
    
    Useful for visualizing hyperspectral data or parameter sweeps.
    
    Parameters
    ----------
    arr : list of array-like
        List of 1D arrays to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    x : array-like, optional
        X-values for all arrays. If None, uses integer indices.
    cmap : str or Colormap, default='viridis'
        Colormap name or Colormap object for automatic coloring.
    color : color, optional
        If provided, use this single color for all lines (overrides cmap).
    alpha : float, default=1.0
        Line transparency.
    lw : float, default=1.5
        Line width.
    linestyle : str, default='-'
        Line style.
    
    Returns
    -------
    list or None
        [fig, ax] if a new figure was created, otherwise None.
    
    Examples
    --------
    >>> data = [np.sin(np.linspace(0, 2*np.pi, 100) + i) for i in range(5)]
    >>> fig, ax = plot_array(data, cmap='plasma')
    """
    import seaborn as sns
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False
    
    # Set up x values
    if x is None:
        x = np.arange(arr[0].shape[0])
    
    # Set up colors
    if color is not None:
        colors = [color] * len(arr)
    elif isinstance(cmap, str):
        cmap_obj = sns.color_palette(cmap, as_cmap=True)
        colors = cmap_obj(np.linspace(0.1, 1, len(arr)))
    elif isinstance(cmap, mpl.colors.Colormap):
        colors = cmap(np.linspace(0.1, 1, len(arr)))
    else:
        colors = ['C0'] * len(arr)
    
    # Plot each array
    for c, y in zip(colors, arr):
        ax.plot(x, y, linewidth=lw, color=c, alpha=alpha, linestyle=linestyle)
    
    if return_fig:
        return [fig, ax]


def colorbar(arr, fig=None, cmap='viridis', orientation='vertical'):
    """
    Add a standalone colorbar to a figure.
    
    Parameters
    ----------
    arr : array-like
        Array to determine colorbar range (uses min/max).
    fig : matplotlib.figure.Figure, optional
        Figure to add colorbar to. If None, creates a new figure.
    cmap : str, default='viridis'
        Colormap name.
    orientation : str, default='vertical'
        'vertical' or 'horizontal'.
    
    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    if fig is None:
        fig, _ = plt.subplots(1, 1)
    
    norm = Normalize(vmin=np.min(arr), vmax=np.max(arr))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, orientation=orientation)
    
    plt.tight_layout()
    return cbar


# =============================================================================
# Image Processing Utilities
# =============================================================================

def autocrop_png(input_path, output_path, border=0):
    """
    Autocrop a PNG image by removing transparent borders.
    
    Parameters
    ----------
    input_path : str
        Path to the input PNG image.
    output_path : str
        Path to save the cropped output image.
    border : int, default=0
        Optional padding to add around the cropped area (in pixels).
    
    Returns
    -------
    str
        Message indicating the result of the operation.
    
    Examples
    --------
    >>> autocrop_png('figure.png', 'figure_cropped.png', border=10)
    """
    try:
        image = Image.open(input_path)
        
        # Ensure RGBA mode
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get bounding box of non-transparent pixels
        bbox = image.getbbox()
        
        if bbox:
            image = image.crop(bbox)
            width, height = image.size
            
            # Add border if specified
            if border > 0:
                new_width = width + border * 2
                new_height = height + border * 2
                bordered = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
                bordered.paste(image, (border, border))
                image = bordered
            
            image.save(output_path, format="PNG")
            return f"Cropped image saved to {output_path}"
        else:
            image.save(output_path, format="PNG")
            return "No visible content found. Original image saved."
    except Exception as e:
        return f"Error processing image: {str(e)}"


def autocrop_png_whitespace(input_path, output_path, border=0, white_thresh=10):
    """
    Autocrop a PNG image by removing both transparent and white borders.
    
    Parameters
    ----------
    input_path : str
        Path to the input PNG image.
    output_path : str
        Path to save the cropped output image.
    border : int, default=0
        Optional padding to add around the cropped area (in pixels).
    white_thresh : int, default=10
        Threshold for what is considered "white" (0-255). 
        Pixels with RGB values > (255 - white_thresh) are treated as white.
    
    Returns
    -------
    str
        Message indicating the result of the operation.
    """
    from PIL import ImageChops
    
    image = Image.open(input_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Step 1: Crop transparent borders
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Step 2: Crop white borders
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    diff = ImageChops.difference(image, bg)
    diff = diff.convert("L")
    
    bbox = diff.point(lambda x: 255 if x > white_thresh else 0).getbbox()
    if bbox:
        left = max(bbox[0] - border, 0)
        upper = max(bbox[1] - border, 0)
        right = min(bbox[2] + border, image.width)
        lower = min(bbox[3] + border, image.height)
        image = image.crop((left, upper, right, lower))
    
    image.save(output_path, format="PNG")
    return f"Cropped image saved to {output_path}"


# =============================================================================
# Convenience Helpers
# =============================================================================

def normalize(x):
    """Normalize array by its first element."""
    return x / x[0]


# =============================================================================
# Spine and Axes Styling
# =============================================================================

def set_spine_style(ax, spines='all', linewidth=1.0, color='black', visible=True):
    """
    Control axis spine visibility, thickness, and color.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    spines : str or list, default='all'
        Which spines to modify. Options:
        - 'all': All four spines
        - 'box': All four spines (same as 'all')
        - 'L': Left and bottom only (common for plots)
        - list: e.g., ['left', 'bottom', 'top', 'right']
    linewidth : float, default=1.0
        Spine line width.
    color : str, default='black'
        Spine color.
    visible : bool, default=True
        Whether spines should be visible.
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> set_spine_style(ax, spines='L', linewidth=1.5)  # Only left and bottom
    """
    all_spines = ['top', 'bottom', 'left', 'right']
    
    if spines == 'all' or spines == 'box':
        target_spines = all_spines
    elif spines == 'L':
        target_spines = ['left', 'bottom']
        # Hide the others
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    elif isinstance(spines, list):
        target_spines = spines
        # Hide spines not in the list
        for spine in all_spines:
            if spine not in target_spines:
                ax.spines[spine].set_visible(False)
    else:
        target_spines = all_spines
    
    for spine in target_spines:
        ax.spines[spine].set_visible(visible)
        ax.spines[spine].set_linewidth(linewidth)
        ax.spines[spine].set_color(color)


def set_aspect_ratio(fig, ratio='golden', width=None):
    """
    Set figure to a specific aspect ratio.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to resize.
    ratio : str or float, default='golden'
        Aspect ratio. Options:
        - 'golden': Golden ratio (1.618)
        - 'square': 1:1
        - 'wide': 16:9
        - 'paper': 4:3
        - float: Custom width/height ratio
    width : float, optional
        Figure width in inches. If None, keeps current width.
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> set_aspect_ratio(fig, 'golden', width=6)
    """
    ratios = {
        'golden': 1.618,
        'square': 1.0,
        'wide': 16/9,
        'paper': 4/3,
    }
    
    if isinstance(ratio, str):
        if ratio not in ratios:
            raise ValueError(f"Unknown ratio '{ratio}'. Options: {list(ratios.keys())}")
        r = ratios[ratio]
    else:
        r = float(ratio)
    
    current_width, _ = fig.get_size_inches()
    w = width if width is not None else current_width
    h = w / r
    
    fig.set_size_inches(w, h)


def annotate_subplots(axes, labels=None, loc='upper left', fontsize=14, 
                      fontweight='bold', prefix='', suffix=')'):
    """
    Add (a), (b), (c), ... labels to subplots.
    
    Parameters
    ----------
    axes : array-like of Axes
        List or array of axes to label.
    labels : list, optional
        Custom labels. If None, uses a, b, c, ...
    loc : str, default='upper left'
        Label location. Options: 'upper left', 'upper right', 
        'lower left', 'lower right', or tuple (x, y) in axes coords.
    fontsize : int, default=14
        Label font size.
    fontweight : str, default='bold'
        Font weight ('normal', 'bold', etc.).
    prefix : str, default=''
        Text before the letter (e.g., '(' for '(a)').
    suffix : str, default=')'
        Text after the letter (e.g., ')' for '(a)').
    
    Examples
    --------
    >>> fig, axes = plt.subplots(2, 2)
    >>> annotate_subplots(axes.flat)  # Adds (a), (b), (c), (d)
    >>> annotate_subplots(axes.flat, prefix='(', suffix=')')  # Same
    """
    import string
    
    # Flatten axes if needed
    if hasattr(axes, 'flat'):
        axes = list(axes.flat)
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    
    # Generate labels
    if labels is None:
        labels = [f"{prefix}{letter}{suffix}" for letter in string.ascii_lowercase[:len(axes)]]
    
    # Position mapping
    positions = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
    }
    
    if isinstance(loc, str):
        x, y = positions.get(loc, (0.05, 0.95))
        ha = 'left' if 'left' in loc else 'right'
        va = 'top' if 'upper' in loc else 'bottom'
    else:
        x, y = loc
        ha, va = 'left', 'top'
    
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                fontweight=fontweight, ha=ha, va=va)


# =============================================================================
# Save Utilities
# =============================================================================

def savefig_tight(fig, path, dpi=300, transparent=True, autocrop=True, 
                  border=10, **kwargs):
    """
    Save figure with tight layout and optional autocropping.
    
    Convenience function that combines savefig with bbox_inches='tight'
    and optional transparent background + autocropping.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Output file path (should end in .png for autocrop).
    dpi : int, default=300
        Resolution in dots per inch.
    transparent : bool, default=True
        Whether to use transparent background.
    autocrop : bool, default=True
        Whether to autocrop the saved image (PNG only).
    border : int, default=10
        Border padding for autocrop (in pixels).
    **kwargs : dict
        Additional arguments passed to fig.savefig().
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> savefig_tight(fig, 'plot.png')
    """
    import os
    
    fig.savefig(path, dpi=dpi, transparent=transparent, 
                bbox_inches='tight', **kwargs)
    
    # Autocrop if PNG
    if autocrop and path.lower().endswith('.png'):
        autocrop_png(path, path, border=border)