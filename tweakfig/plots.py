import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

from PIL import Image
import os

from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, FixedFormatter
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def PrettyMatplotlib(fig_font_scale=1.0, minor_tick_size=0.5, major_tick_size=1.0, num_minor_ticks=4, **kwargs):
    import matplotlib as mpl

    # Font sizes
    mpl.rcParams["axes.labelsize"] = fig_font_scale * 14
    mpl.rcParams["axes.titlesize"] = fig_font_scale * 16
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['xtick.labelsize'] = fig_font_scale * 12
    mpl.rcParams['ytick.labelsize'] = fig_font_scale * 12
    mpl.rcParams["axes.labelpad"] = 0.7
    mpl.rcParams["legend.fontsize"] = fig_font_scale * 12  # Legend font size

    # Tick sizes
    mpl.rcParams['xtick.major.size'] = major_tick_size * 6
    mpl.rcParams['xtick.minor.size'] = minor_tick_size * 6
    mpl.rcParams['ytick.major.size'] = major_tick_size * 6
    mpl.rcParams['ytick.minor.size'] = minor_tick_size * 6

    # Number of minor ticks
    mpl.rcParams['xtick.minor.visible'] = True
    #just a comment
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.minor.top'] = True
    mpl.rcParams['ytick.minor.right'] = True
    mpl.rcParams['xtick.minor.bottom'] = True
    mpl.rcParams['ytick.minor.left'] = True
    mpl.rcParams['xtick.minor.width'] = minor_tick_size
    mpl.rcParams['ytick.minor.width'] = minor_tick_size
    mpl.rcParams['xtick.major.width'] = major_tick_size
    mpl.rcParams['ytick.major.width'] = major_tick_size

    # Apply additional customizations from kwargs
    for key, value in kwargs.items():
        mpl.rcParams[key] = value

_nred = lambda x: x/x[0]

def x_num_ticks(ax, xaxis=(4, 4)):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator

    ax.xaxis.set_major_locator(MaxNLocator(xaxis[0]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xaxis[1]))
    # Example usage

def y_num_ticks(ax, yaxis=(4, 4), log_scale=False):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator

    if log_scale:
        # Use LogLocator for log-scale axes
        #ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=major_ticks[0]))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=yaxis[0]))
        #ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=minor_ticks[0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=yaxis[1]))
    else:
        # Use MaxNLocator and AutoMinorLocator for linear-scale axes
        ax.yaxis.set_major_locator(MaxNLocator(yaxis[0]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(yaxis[1]))


def num_ticks(ax, xaxis=(4, 4), yaxis=(4, 4), log_scale=False):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator

    if log_scale:
        # Use LogLocator for log-scale axes
        #ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=major_ticks[0]))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=yaxis[0]))
        #ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=minor_ticks[0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=yaxis[1]))
    else:
        # Use MaxNLocator and AutoMinorLocator for linear-scale axes
        ax.yaxis.set_major_locator(MaxNLocator(yaxis[0]))
        ax.yaxis.set_minor_locator(AutoMinorLocator(yaxis[1]))

    ax.xaxis.set_major_locator(MaxNLocator(xaxis[0]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xaxis[1]))
    # Example usage

def colorFader(c1, c2, mix=0):
    # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def colorbar(arr, **kwargs):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    params = {
        'fig': None,
        'cmap': 'viridis'
    }

    params.update(kwargs)

    if params['fig'] is None:
        f2, a2 = plt.subplots(1, 1)
    else:
        f2 = params['fig']
    

    norm = Normalize(vmin=np.min(arr), vmax=np.max(arr))  # Set your desired range here
    sm = ScalarMappable(norm=norm, cmap='viridis')

    # Remove the axis
    #a2.axis('off')

    # Add the colorbar
    cbar = f2.colorbar(sm, orientation='vertical')

    plt.tight_layout()
    plt.show()

def plot_array(arr, **kwargs):
    
    import seaborn as sns

    params = {
        'alpha': 1.0,
        'ax': None,
        'x': np.arange(arr[0].shape[0]),
        'lw': 1.5,
        'linestyle': '-',
        'color': None,
        'cmap': 'viridis'
    }
    
    params.update(kwargs)

    if params['ax'] is None:
        f2, a2 = plt.subplots(1, 1)
    else:
        a2 = params['ax']
    
    if params['color'] is None:
        if isinstance(params['cmap'], str):
            viridis = sns.color_palette(params['cmap'], as_cmap=True)
            colors = viridis(np.linspace(0.1, 1, len(arr)))
        elif isinstance(params['cmap'], mpl.colors.Colormap):
            colors = params['cmap']
    else:
        colors = [params['color']] * len(arr)

    for _c, _arr in zip(colors, arr):
        a2.plot(params['x'], _arr, linewidth=params['lw'], color=_c, alpha=params['alpha'], linestyle=params['linestyle'])   

    if params['ax'] is None:
        return [f2, a2]


def autocrop_png(input_path, output_path, border=0):
    """
    Autocrop a PNG image to its visible area by removing transparent borders.
    
    Parameters:
    - input_path (str): Path to the input PNG image.
    - output_path (str): Path to save the cropped output image.
    - border (int): Optional border to add around the cropped area (in pixels).
    
    Returns:
    - str: Message indicating the result of the operation.
    """
    try:
        # Open the input image
        image = Image.open(input_path)
        
        # Ensure the image has an alpha channel (transparency)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get the bounding box of non-transparent pixels
        bbox = image.getbbox()
        
        if bbox:
            # Crop the image to the bounding box
            image = image.crop(bbox)
            
            # Get dimensions of the cropped image
            width, height = image.size
            
            # Add border if specified
            if border > 0:
                width += border * 2
                height += border * 2
                # Create a new image with border, filled with transparent background
                cropped_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                # Paste the cropped image in the center with border offset
                cropped_image.paste(image, (border, border))
                image = cropped_image
            
            # Save the cropped image
            image.save(output_path, format="PNG")
            return f"Cropped image saved to {output_path}"
        else:
            # Save the original image if no content is detected
            image.save(output_path, format="PNG")
            return "No visible content found in the image to crop. Original image saved."
    except Exception as e:
        return f"Error processing image: {str(e)}"


def autocrop_png2(input_path, output_path, border=0):
    from PIL import Image, ImageChops
    
    """
    Autocrop a PNG image by removing both transparent and white borders.
    
    Parameters:
    - input_path (str): Path to the input PNG image.
    - output_path (str): Path to save the cropped output image.
    - border (int): Optional border to add around the cropped area (in pixels).
    - white_thresh (int): Threshold for what is considered "white" (0-255).
    """
    image = Image.open(input_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 1. Crop transparent borders
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)

    # 2. Crop white borders
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    diff = ImageChops.difference(image, bg)
    diff = diff.convert("L")
    # Everything below threshold is considered white
    bbox = diff.point(lambda x: 255 if x > (255 - white_thresh) else 0).getbbox()
    if bbox:
        # Expand bbox by border, but keep within image bounds
        left = max(bbox[0] - border, 0)
        upper = max(bbox[1] - border, 0)
        right = min(bbox[2] + border, image.width)
        lower = min(bbox[3] + border, image.height)
        image = image.crop((left, upper, right, lower))

    image.save(output_path, format="PNG")
    return f"Cropped image saved to {output_path}"