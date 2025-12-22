# tweakfig

A lightweight Python library for creating publication-ready matplotlib figures with minimal effort.

## Features

- **One-line styling**: Apply consistent, beautiful styling to all your plots
- **Tick control**: Fine-grained control over major and minor tick counts
- **Color utilities**: Easy color interpolation and colormap-based line colors
- **Image cropping**: Remove transparent/white borders from saved figures
- **Minimal & transparent**: Simple functions that work with standard matplotlib

## Installation

```bash
pip install tweakfig
```

Or install from source:
```bash
git clone https://github.com/amplipy/tweakfig.git
cd tweakfig
pip install -e .
```

## Quick Start

```python
import tweakfig as tfp
import matplotlib.pyplot as plt
import numpy as np

# Apply publication-ready styling (call once at the start)
tfp.PrettyMatplotlib()

# Create your plot as usual
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# Customize tick counts: (n_major, n_minor)
tfp.num_ticks(ax, xaxis=(5, 2), yaxis=(4, 5))

plt.savefig('figure.png', dpi=300, transparent=True)

# Optionally crop the saved figure
tfp.autocrop_png('figure.png', 'figure_cropped.png', border=10)
```

## API Reference

### Global Styling

#### `PrettyMatplotlib(fig_font_scale=1.0, minor_tick_size=0.5, major_tick_size=1.0, font_family='Arial', **kwargs)`

Apply publication-ready matplotlib rcParams globally. Call once at the start of your script.

```python
# Larger fonts for presentations
tfp.PrettyMatplotlib(fig_font_scale=1.5)

# Custom settings
tfp.PrettyMatplotlib(font_family='Times New Roman', **{'figure.dpi': 150})
```

### Tick Control

#### `num_ticks(ax, xaxis=(4, 4), yaxis=(4, 4), log_scale=False)`

Set the number of major and minor ticks on both axes.

```python
tfp.num_ticks(ax, xaxis=(5, 2), yaxis=(4, 5))  # 5 major, 2 minor on x; 4 major, 5 minor on y
```

#### `x_num_ticks(ax, xaxis=(4, 4))` / `y_num_ticks(ax, yaxis=(4, 4), log_scale=False)`

Set ticks on individual axes.

### Color Utilities

#### `colorFader(c1, c2, mix=0)`

Linearly interpolate between two colors.

```python
mid_color = tfp.colorFader('red', 'blue', 0.5)  # Purple
```

#### `plot_array(arr, ax=None, cmap='viridis', ...)`

Plot multiple arrays with automatic color gradient from a colormap.

```python
data = [np.sin(np.linspace(0, 2*np.pi, 100) + i) for i in range(5)]
fig, ax = tfp.plot_array(data, cmap='plasma')
```

### Image Processing

#### `autocrop_png(input_path, output_path, border=0)`

Remove transparent borders from a PNG image.

#### `autocrop_png_whitespace(input_path, output_path, border=0, white_thresh=10)`

Remove both transparent and white borders.

## License

MIT
