import json
from math import radians

from pathlib import Path

import cmocean
import matplotlib.colors as colors

# import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mhkit.tidal as tidal
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from matplotlib.collections import PolyCollection
from matplotlib import colormaps
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# from matplotlib.projections import PolarAxes
# from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from utide import solve, reconstruct
from windrose import WindroseAxes
from sklearn.metrics.pairwise import haversine_distances
from haversine import haversine


sns.set_theme()


# INPUT_PATH = Path("./data/b4_parquet_partition/cook_inlet")
# INPUT_PATH = Path("./data/WA_puget_sound/b4_by_point_parquet/")
# INPUT_PATH = Path("./data/AK_cook_inlet_nikiski/b4_partition/")
INPUT_PATH = Path("./data/WA_puget_sound/b4_partition/")
OUTPUT_PATH = Path("./viz/WA_puget_sound/rosario_strait")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

JPD_WIDTH_DIRECTION_BIN_WIDTH_DEGREES = 1
JPD_VELOCITY_BIN_WIDTH_METERS_PER_SECOND = 0.1

parquet_files = sorted(list(INPUT_PATH.glob("*.parquet")))

plt.rcParams["font.family"] = "Public Sans, Arial, sans-serif"
plt.rcParams["axes.titleweight"] = "bold"  # Bold titles
plt.rcParams["axes.labelweight"] = "bold"  # Bold axis labels

colormaps.register(name="cmocean_thermal", cmap=cmocean.cm.thermal)
plt.set_cmap("cmocean_thermal")


def standardize_metadata(schema_metadata, standardize_values=True):
    """
    Standardize PyArrow schema metadata keys and optionally values.

    Args:
        schema_metadata (dict): The original metadata dictionary with byte keys
        standardize_values (bool): Whether to also standardize values

    Returns:
        dict: A new dictionary with standardized keys and optionally values
    """
    standardized = {}

    # Process each key-value pair
    for key, value in schema_metadata.items():
        # print(f"Original key is: {key}")
        # print(f"Original value is: {value}")
        # Decode byte keys to strings
        if isinstance(key, bytes):
            std_key = key.decode("utf-8")
        else:
            std_key = str(key)

        # Standardize values if requested
        if standardize_values:
            if isinstance(value, bytes):
                std_value = value.decode("utf-8")
            elif isinstance(value, dict):
                # Recursively standardize nested dictionaries
                std_value = standardize_metadata(value, standardize_values=True)
            elif isinstance(value, list):
                # Standardize items in lists
                std_value = [
                    item.decode("utf-8") if isinstance(item, bytes) else item
                    for item in value
                ]
            else:
                std_value = value
        else:
            std_value = value

        # print(f"Standardized key is: {std_key}")
        # print(f"Standardized value is: {std_value}")
        standardized[std_key] = std_value

    return standardized


def generate_tidal_joint_probability(df, sigma_layer):
    to_direction = df[f"vap_sea_water_to_direction_layer_{sigma_layer}"]
    speed = df[f"vap_sea_water_speed_layer_{sigma_layer}"]
    depth = df[f"vap_sigma_depth_layer_{sigma_layer}"]

    title = f"{df['dataset_name'].iloc[0]}"
    time_str = f"Time Range: {df.index[0]} - {df.index[-1]} [UTC]"
    depth_str = f"Depth Range: {depth.min():.2f} - {depth.max():.2f} [m]"
    speed_str = f"Speed Range: {speed.min():.2f} - {speed.max():.2f} [m/s]"

    # direction1, direction2 = tidal.resource.principal_flow_directions(
    #     to_direction, JPD_WIDTH_DIRECTION_BIN_WIDTH_DEGREES
    # )
    #
    # direction2 = np.mod(direction1 - 180, 360)

    ax = tidal.graphics.plot_joint_probability_distribution(
        to_direction,
        speed,
        JPD_WIDTH_DIRECTION_BIN_WIDTH_DEGREES,
        JPD_VELOCITY_BIN_WIDTH_METERS_PER_SECOND,
        metadata={
            "name": f"{time_str}\n{speed_str}\n{depth_str}",
            "lat": df["lat_center"].iloc[0],
            "lon": df["lon_center"].iloc[0],
        },
        # flood=float(direction1),
        # ebb=float(direction2),
    )
    ax.figure.set_size_inches(8, 8)
    plt.suptitle(title)
    plt.tight_layout()

    return ax.figure


def plot_velocity_profile_with_histograms(
    df: pd.DataFrame,
    # Sigma layer filtering options
    filter_dry_points: bool = True,
    dry_threshold: float = 0.0,  # Surface layer depth threshold for "dry" detection
    filter_smushed_layers: bool = True,
    min_layer_thickness: float = 0.1,  # Minimum thickness for a layer to be considered valid
    min_total_depth: float = 1.0,  # Minimum total water depth to include point
    # Visualization options
    show_filtered_stats: bool = True,
    invert_depth_axis: bool = True,  # True for oceanographic convention (surface at top)
    # Debug options
    verbose: bool = False,
    # ) -> Tuple[plt.Figure, Dict[str, Any]]:
):
    """
    Plot velocity profiles with handling for FVCOM sigma coordinate edge cases.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing FVCOM velocity and depth data
    filter_dry_points : bool
        Remove time points where the surface layer indicates "dry" conditions
    dry_threshold : float
        Depth threshold below which surface layer is considered "dry" (usually 0.0)
    filter_smushed_layers : bool
        Remove individual layers that are too thin (smushed sigma layers)
    min_layer_thickness : float
        Minimum thickness (m) for a layer to be included
    min_total_depth : float
        Minimum total water depth (m) to include a time point
    show_filtered_stats : bool
        Display statistics about filtered data
    invert_depth_axis : bool
        Invert y-axis so surface (depth=0) is at top
    verbose : bool
        Print detailed filtering information

    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    stats : dict
        Dictionary containing filtering statistics and data quality metrics
    """

    # Initialize statistics dictionary
    stats = {
        "original_points": len(df),
        "filtered_points": 0,
        "dry_points_removed": 0,
        "shallow_points_removed": 0,
        "smushed_layers_removed": 0,
        "total_layers_original": 0,
        "total_layers_after_filtering": 0,
    }

    # Extract layer information
    n_layers = 10  # Assuming 10 sigma layers
    velocity_cols = [f"vap_sea_water_speed_layer_{i}" for i in range(n_layers)]
    direction_cols = [f"vap_sea_water_to_direction_layer_{i}" for i in range(n_layers)]
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(n_layers)]

    # Check if all required columns exist
    missing_cols = []
    for cols in [velocity_cols, direction_cols, depth_cols]:
        missing_cols.extend([col for col in cols if col not in df.columns])

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if verbose:
        print(f"Starting with {len(df)} data points")

    # Create copies for filtering
    df_filtered = df.copy()

    # 1. Filter dry points (where surface layer is above/at dry threshold)
    if filter_dry_points:
        surface_depths = df_filtered[depth_cols[0]]  # Assuming layer 0 is surface
        dry_mask = surface_depths <= dry_threshold
        stats["dry_points_removed"] = dry_mask.sum()
        df_filtered = df_filtered[~dry_mask]

        if verbose:
            print(
                f"Removed {stats['dry_points_removed']} dry points (surface depth <= {dry_threshold}m)"
            )

    # 2. Filter points with insufficient total depth
    if min_total_depth > 0:
        # Calculate total depth as maximum depth across all layers
        max_depths = df_filtered[depth_cols].max(axis=1)
        shallow_mask = max_depths < min_total_depth
        stats["shallow_points_removed"] = shallow_mask.sum()
        df_filtered = df_filtered[~shallow_mask]

        if verbose:
            print(
                f"Removed {stats['shallow_points_removed']} shallow points (max depth < {min_total_depth}m)"
            )

    stats["filtered_points"] = len(df_filtered)

    if len(df_filtered) == 0:
        raise ValueError("No data points remain after filtering!")

    # Extract filtered data
    n_locations = len(df_filtered)
    all_velocities = np.zeros((n_locations, n_layers))
    all_depths = np.zeros((n_locations, n_layers))
    all_directions = np.zeros((n_locations, n_layers))
    layer_valid_mask = np.ones((n_locations, n_layers), dtype=bool)

    # Fill arrays and apply layer-wise filtering
    for loc in range(n_locations):
        for layer in range(n_layers):
            all_velocities[loc, layer] = df_filtered.iloc[loc][velocity_cols[layer]]
            all_directions[loc, layer] = df_filtered.iloc[loc][direction_cols[layer]]
            all_depths[loc, layer] = df_filtered.iloc[loc][depth_cols[layer]]

    stats["total_layers_original"] = n_locations * n_layers

    # 3. Filter smushed layers
    if filter_smushed_layers:
        for loc in range(n_locations):
            # Calculate layer thicknesses
            loc_depths = all_depths[loc, :]

            # Sort depths to calculate thicknesses properly
            depth_order = np.argsort(loc_depths)
            sorted_depths = loc_depths[depth_order]

            # Calculate thicknesses between consecutive layers
            thicknesses = np.diff(sorted_depths)

            # Find layers that are too thin
            for i, layer_idx in enumerate(depth_order[:-1]):
                if thicknesses[i] < min_layer_thickness:
                    layer_valid_mask[loc, layer_idx] = False
                    stats["smushed_layers_removed"] += 1

    stats["total_layers_after_filtering"] = layer_valid_mask.sum()

    if verbose:
        print(
            f"Removed {stats['smushed_layers_removed']} smushed layers (thickness < {min_layer_thickness}m)"
        )
        print(
            f"Retained {stats['total_layers_after_filtering']} / {stats['total_layers_original']} total layer-points"
        )

    # Apply layer mask to data
    all_velocities = np.where(layer_valid_mask, all_velocities, np.nan)
    all_directions = np.where(layer_valid_mask, all_directions, np.nan)
    all_depths = np.where(layer_valid_mask, all_depths, np.nan)

    # Calculate statistics using valid data only
    with np.errstate(invalid="ignore"):  # Suppress NaN warnings
        mean_depths = np.nanmean(all_depths, axis=0)
        max_depths = np.nanmax(all_depths, axis=0)
        min_depths = np.nanmin(all_depths, axis=0)
        mean_velocities = np.nanmean(all_velocities, axis=0)
        mean_directions = np.nanmean(all_directions, axis=0)

    # Colors (same as original)
    colors = sns.color_palette()
    boxplot_color = colors[0]
    boxplot_alpha = 0.3
    boxplot_edge_color = colors[0]
    median_line_color = colors[1]
    whisker_color = "#333333"
    whisker_width = 0.75

    profile_line_color = colors[0]
    profile_marker_color = colors[0]

    histogram_color = colors[0]
    histogram_edge_color = colors[0]
    histogram_alpha = 0.8

    direction_hist_color = colors[2]
    direction_hist_edge_color = colors[2]
    direction_hist_alpha = 0.8

    depth_hist_color = colors[3]
    depth_hist_edge_color = colors[3]
    depth_hist_alpha = 0.8

    # Create main figure
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(1, 1)

    # Create top row grid
    top_row = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 0.75, 0.75, 0.75])

    # Create axes for top row
    ax_profile = fig.add_subplot(top_row[0])
    ax_scatter = fig.add_subplot(top_row[1])
    ax_hist = fig.add_subplot(top_row[2])
    ax_dir = fig.add_subplot(top_row[3])
    ax_depth = fig.add_subplot(top_row[4])

    # Set custom tick formatters for all axes to use 2 decimal places
    for ax in [ax_profile, ax_scatter]:
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # Calculate boxplot statistics manually (handling NaN values)
    boxplot_data = []
    for i in range(n_layers):
        layer_velocities = all_velocities[:, i]
        valid_velocities = layer_velocities[~np.isnan(layer_velocities)]

        if len(valid_velocities) > 0:
            q1 = np.percentile(valid_velocities, 25)
            median = np.percentile(valid_velocities, 50)
            q3 = np.percentile(valid_velocities, 75)
            iqr = q3 - q1
            whisker_low = max(np.min(valid_velocities), q1 - 1.5 * iqr)
            whisker_high = min(np.max(valid_velocities), q3 + 1.5 * iqr)

            boxplot_data.append(
                {
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "whisker_low": whisker_low,
                    "whisker_high": whisker_high,
                    "mean": np.nanmean(valid_velocities),
                    "n_valid": len(valid_velocities),
                }
            )
        else:
            boxplot_data.append(None)  # No valid data for this layer

    # Create the boxplots with custom colors (only for layers with valid data)
    for i in range(n_layers):
        if boxplot_data[i] is not None and not np.isnan(mean_depths[i]):
            layer_velocities = all_velocities[:, i]
            valid_velocities = layer_velocities[~np.isnan(layer_velocities)]

            box = ax_profile.boxplot(
                valid_velocities,
                positions=[mean_depths[i]],
                vert=False,
                widths=0.2,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=boxplot_color,
                    alpha=boxplot_alpha,
                    edgecolor=boxplot_edge_color,
                ),
                medianprops=dict(color=median_line_color, linewidth=1.5),
                whiskerprops=dict(color=whisker_color, linewidth=whisker_width),
                capprops=dict(color=whisker_color),
            )

    # Plot the velocity profile with markers using the mean values (only valid layers)
    valid_layers = ~np.isnan(mean_depths) & ~np.isnan(mean_velocities)
    if np.any(valid_layers):
        ax_profile.plot(
            mean_velocities[valid_layers],
            mean_depths[valid_layers],
            "-",
            linewidth=2,
            label="Mean Velocity Profile",
            marker=".",
            markersize=10,
            color=profile_line_color,
            markerfacecolor=profile_marker_color,
            zorder=10,
        )

    # Create custom legend entries for box plot elements
    ax_profile.plot(
        [], [], "-", color=whisker_color, label="Whiskers: Min/Max (no outliers)"
    )
    ax_profile.plot(
        [],
        [],
        "s",
        color=boxplot_edge_color,
        markerfacecolor=boxplot_color,
        alpha=boxplot_alpha,
        markersize=8,
        label="Quartiles: 25th/75th percentile",
    )
    ax_profile.plot([], [], "-", color=median_line_color, linewidth=1.5, label="Median")

    # Move legend to the bottom of the plot, outside the axes
    ax_profile.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Set labels and title for velocity profile
    ax_profile.set_xlabel("Sea Water Speed [m/s]")
    ax_profile.set_ylabel("Depth [m]")

    # Create title with filtering info
    title = "Velocity Profile with Box Plots"
    if show_filtered_stats:
        title += f" (n={stats['filtered_points']}/{stats['original_points']} points)"
    ax_profile.set_title(title)

    # Create custom y-tick labels with depth ranges (only for valid layers)
    valid_depth_layers = ~np.isnan(mean_depths)
    if np.any(valid_depth_layers):
        y_ticks = mean_depths[valid_depth_layers]
        y_tick_labels = [
            f"{min_depths[i]:.1f} - {max_depths[i]:.1f}"
            for i in range(n_layers)
            if valid_depth_layers[i]
        ]

        ax_profile.set_yticks(y_ticks)
        ax_profile.set_yticklabels(y_tick_labels)

    # Invert y-axis so that depth increases downward (oceanographic convention)
    if invert_depth_axis:
        ax_profile.invert_yaxis()
    ax_profile.grid(True)

    # Create the scatter plot (similar updates for NaN handling)
    # Flatten arrays for scatter plot, removing NaN values
    flat_depths = all_depths.flatten()
    flat_velocities = all_velocities.flatten()
    flat_directions = all_directions.flatten()

    # Remove NaN values
    valid_data = ~(
        np.isnan(flat_depths) | np.isnan(flat_velocities) | np.isnan(flat_directions)
    )
    flat_depths = flat_depths[valid_data]
    flat_velocities = flat_velocities[valid_data]
    flat_directions = flat_directions[valid_data]

    if len(flat_depths) > 0:
        # Create scatter plot
        scatter = ax_scatter.scatter(
            flat_velocities,
            flat_depths,
            alpha=1.0,
            c=flat_directions,
            # cmap="twilight",
            cmap=cmocean.cm.phase,
            s=3,
            edgecolor="none",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label("Direction")

        # Calculate mean and max velocities per layer (using valid data)
        layer_mean_speeds = np.nanmean(all_velocities, axis=0)
        layer_max_speeds = np.nanmax(all_velocities, axis=0)
        layer_depths = mean_depths

        # Only use layers with valid data for fitting
        fit_valid = ~(
            np.isnan(layer_depths)
            | np.isnan(layer_mean_speeds)
            | np.isnan(layer_max_speeds)
        )

        if np.sum(fit_valid) >= 3:  # Need at least 3 points for quadratic fit
            # Fit quadratic models based on the layer data
            mean_coefs = np.polyfit(
                layer_depths[fit_valid], layer_mean_speeds[fit_valid], 2
            )
            a_mean, b_mean, c_mean = mean_coefs

            max_coefs = np.polyfit(
                layer_depths[fit_valid], layer_max_speeds[fit_valid], 2
            )
            a_max, b_max, c_max = max_coefs

            # Generate x values for plotting smooth curves
            line_x = np.linspace(np.nanmin(flat_depths), np.nanmax(flat_depths), 100)

            # Generate quadratic model predictions
            mean_speed_pred = a_mean * line_x**2 + b_mean * line_x + c_mean
            max_speed_pred = a_max * line_x**2 + b_max * line_x + c_max

            # Plot the fit lines
            ax_scatter.plot(
                mean_speed_pred,
                line_x,
                color=sns.color_palette()[0],
                linewidth=2,
                label="Mean Speed (Quadratic Fit)",
            )

            ax_scatter.plot(
                max_speed_pred,
                line_x,
                color=sns.color_palette()[3],
                linewidth=2,
                linestyle="--",
                label="Max Speed (Quadratic Fit)",
            )

            # Add equation text
            mean_eq = f"Mean Speed = {a_mean:.4f} × Depth² + {b_mean:.4f} × Depth + {c_mean:.4f}"
            max_eq = (
                f"Max Speed = {a_max:.4f} × Depth² + {b_max:.4f} × Depth + {c_max:.4f}"
            )

            # Add regression equations below the plot
            fig.text(
                0.5,
                0.02,
                f"{mean_eq}\n{max_eq}",
                ha="center",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Plot the layer mean and max points (only valid ones)
        valid_for_scatter = (
            fit_valid if "fit_valid" in locals() else ~np.isnan(layer_depths)
        )
        if np.any(valid_for_scatter):
            ax_scatter.scatter(
                layer_mean_speeds[valid_for_scatter],
                layer_depths[valid_for_scatter],
                color=sns.color_palette()[0],
                s=50,
                zorder=5,
                label="Layer Mean Speeds",
            )

            ax_scatter.scatter(
                layer_max_speeds[valid_for_scatter],
                layer_depths[valid_for_scatter],
                color=sns.color_palette()[3],
                s=50,
                zorder=5,
                marker="s",
                label="Layer Max Speeds",
            )

    # Set labels and title
    ax_scatter.set_xlabel("Sea Water Speed [m/s]")
    ax_scatter.set_ylabel("Depth [m]")
    ax_scatter.set_title("Depth vs. Speed Scatter Plot")

    # Move legend to the bottom of the plot, outside the axes
    ax_scatter.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Invert y-axis to match profile plot
    if invert_depth_axis:
        ax_scatter.invert_yaxis()

    # Add gridlines
    ax_scatter.grid(True, linestyle="--", alpha=0.7)

    # Continue with histogram creation (with NaN handling)...
    # [Rest of histogram code would follow similar pattern with NaN handling]

    # Remove the placeholder axes and create subplot grids
    ax_hist.remove()
    hist_grid = top_row[2].subgridspec(n_layers, 1, hspace=0.3)

    ax_dir.remove()
    dir_grid = top_row[3].subgridspec(n_layers, 1, hspace=0.3)

    ax_depth.remove()
    depth_grid = top_row[4].subgridspec(n_layers, 1, hspace=0.3)

    # Create histograms with NaN handling
    hist_axes = []
    dir_axes = []
    depth_axes = []
    n_bins = 50

    # Calculate bin edges using valid data only
    all_valid_velocities = all_velocities[~np.isnan(all_velocities)]
    all_valid_directions = all_directions[~np.isnan(all_directions)]
    all_valid_depths_flat = all_depths[~np.isnan(all_depths)]

    if len(all_valid_velocities) > 0:
        vel_bin_edges = np.linspace(
            np.min(all_valid_velocities) * 0.9,
            np.max(all_valid_velocities) * 1.1,
            n_bins + 1,
        )
    else:
        vel_bin_edges = np.linspace(0, 1, n_bins + 1)

    dir_bin_edges = np.linspace(0, 360, n_bins + 1)

    if len(all_valid_depths_flat) > 0:
        depth_bin_edges = np.linspace(
            np.min(all_valid_depths_flat) * 0.95,
            np.max(all_valid_depths_flat) * 1.05,
            n_bins + 1,
        )
    else:
        depth_bin_edges = np.linspace(0, 10, n_bins + 1)

    for i in range(n_layers):
        # Create subplot for this velocity histogram
        ax_vel = fig.add_subplot(hist_grid[i])
        hist_axes.append(ax_vel)

        # Create subplot for this direction histogram
        ax_dir = fig.add_subplot(dir_grid[i])
        dir_axes.append(ax_dir)

        # Create subplot for depth histogram
        ax_dep = fig.add_subplot(depth_grid[i])
        depth_axes.append(ax_dep)

        # Extract data for this layer, removing NaN values
        layer_velocities = all_velocities[:, i]
        layer_directions = all_directions[:, i]
        layer_depths_col = all_depths[:, i]

        # Remove NaN values
        valid_vel = layer_velocities[~np.isnan(layer_velocities)]
        valid_dir = layer_directions[~np.isnan(layer_directions)]
        valid_dep = layer_depths_col[~np.isnan(layer_depths_col)]

        # Only create histograms if we have valid data
        if len(valid_vel) > 0:
            ax_vel.hist(
                valid_vel,
                bins=vel_bin_edges,
                color=histogram_color,
                edgecolor=histogram_edge_color,
                alpha=histogram_alpha,
            )

        if len(valid_dir) > 0:
            ax_dir.hist(
                valid_dir,
                bins=dir_bin_edges,
                color=direction_hist_color,
                edgecolor=direction_hist_edge_color,
                alpha=direction_hist_alpha,
            )

        if len(valid_dep) > 0:
            ax_dep.hist(
                valid_dep,
                bins=depth_bin_edges,
                color=depth_hist_color,
                edgecolor=depth_hist_edge_color,
                alpha=depth_hist_alpha,
            )

        # Add layer info with valid data count
        if not np.isnan(min_depths[i]) and not np.isnan(max_depths[i]):
            depth_label = f"Layer {i}: {min_depths[i]:.1f} - {max_depths[i]:.1f} m (n={np.sum(~np.isnan(layer_velocities))})"
        else:
            depth_label = f"Layer {i}: No valid data"

        # Add depth range label to all histograms
        for ax in [ax_vel, ax_dir, ax_dep]:
            ax.text(
                0.02,
                0.8,
                depth_label,
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        # Set y-label only for middle histograms
        if i == n_layers // 2:
            ax_vel.set_ylabel("Count")
            ax_dir.set_ylabel("Count")
            ax_dep.set_ylabel("Count")

        # Remove x labels for all but the bottom histogram
        if i < n_layers - 1:
            ax_vel.set_xticklabels([])
            ax_dir.set_xticklabels([])
            ax_dep.set_xticklabels([])
        else:
            ax_vel.set_xlabel("Sea Water Speed [m/s]")
            ax_dir.set_xlabel("Sea Water Direction [degrees]")
            ax_dep.set_xlabel("Depth [m]")

        # Set consistent x limits
        ax_vel.set_xlim(vel_bin_edges[0], vel_bin_edges[-1])
        ax_dir.set_xlim(0, 360)
        ax_dep.set_xlim(depth_bin_edges[0], depth_bin_edges[-1])

        # Add gridlines
        ax_vel.grid(True, linestyle="--", alpha=0.6)
        ax_dir.grid(True, linestyle="--", alpha=0.6)
        ax_dep.grid(True, linestyle="--", alpha=0.6)

    # Add titles to the top histograms
    hist_axes[0].set_title("Velocity Distributions by Depth")
    dir_axes[0].set_title("Direction Distributions by Depth")
    depth_axes[0].set_title("Depth Distributions by Layer")

    # Add filtering statistics as text
    if show_filtered_stats:
        stats_text = (
            f"Data Filtering Summary:\n"
            f"• Original points: {stats['original_points']}\n"
            f"• Dry points removed: {stats['dry_points_removed']}\n"
            f"• Shallow points removed: {stats['shallow_points_removed']}\n"
            f"• Smushed layers removed: {stats['smushed_layers_removed']}\n"
            f"• Final points: {stats['filtered_points']}\n"
            f"• Valid layer-points: {stats['total_layers_after_filtering']}/{stats['total_layers_original']}"
        )

        fig.text(
            0.02,
            0.98,
            stats_text,
            transform=fig.transFigure,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    # Adjust figure to make room for legends and equations
    fig.subplots_adjust(bottom=0.20, left=0.15)

    # return fig, stats
    return fig


# def plot_velocity_profile_with_histograms(df):
#     # Colors
#     colors = sns.color_palette()
#     boxplot_color = colors[0]
#     boxplot_alpha = 0.3
#     boxplot_edge_color = colors[0]
#     median_line_color = colors[1]
#     whisker_color = "#333333"
#     whisker_width = 0.75
#
#     profile_line_color = colors[0]
#     profile_marker_color = colors[0]
#
#     histogram_color = colors[0]
#     histogram_edge_color = colors[0]
#     histogram_alpha = 0.8
#
#     direction_hist_color = colors[2]
#     direction_hist_edge_color = colors[2]
#     direction_hist_alpha = 0.8
#
#     depth_hist_color = colors[3]
#     depth_hist_edge_color = colors[3]
#     depth_hist_alpha = 0.8
#
#     # Number of locations in the dataframe
#     n_locations = len(df)
#     # Number of depth layers
#     n_layers = 10
#
#     # Extract all velocity data and depths
#     all_velocities = np.zeros((n_locations, n_layers))
#     all_depths = np.zeros((n_locations, n_layers))
#     all_directions = np.zeros((n_locations, n_layers))
#
#     # Fill the arrays with data from each location and layer
#     for loc in range(n_locations):
#         for layer in range(n_layers):
#             all_velocities[loc, layer] = df.iloc[loc][
#                 f"vap_sea_water_speed_layer_{layer}"
#             ]
#             all_directions[loc, layer] = df.iloc[loc][
#                 f"vap_sea_water_to_direction_layer_{layer}"
#             ]
#             all_depths[loc, layer] = df.iloc[loc][f"vap_sigma_depth_layer_{layer}"]
#
#     # Calculate mean depths and mean velocities
#     mean_depths = np.mean(all_depths, axis=0)
#     max_depths = np.max(all_depths, axis=0)
#     min_depths = np.min(all_depths, axis=0)
#     mean_velocities = np.mean(all_velocities, axis=0)
#     mean_directions = np.mean(all_directions, axis=0)
#
#     # Create main figure
#     fig = plt.figure(figsize=(24, 12))
#     gs = fig.add_gridspec(1, 1)
#
#     # Create top row grid
#     top_row = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 0.75, 0.75, 0.75])
#
#     # Create axes for top row
#     ax_profile = fig.add_subplot(top_row[0])
#     ax_scatter = fig.add_subplot(top_row[1])
#     ax_hist = fig.add_subplot(top_row[2])
#     ax_dir = fig.add_subplot(top_row[3])
#     ax_depth = fig.add_subplot(top_row[4])
#
#     # Set custom tick formatters for all axes to use 2 decimal places
#     for ax in [ax_profile, ax_scatter]:
#         ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
#         ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
#
#     # Calculate boxplot statistics manually
#     boxplot_data = []
#     for i in range(n_layers):
#         # Calculate statistical values for boxplots
#         q1 = np.percentile(all_velocities[:, i], 25)
#         median = np.percentile(all_velocities[:, i], 50)
#         q3 = np.percentile(all_velocities[:, i], 75)
#         iqr = q3 - q1
#         whisker_low = max(np.min(all_velocities[:, i]), q1 - 1.5 * iqr)
#         whisker_high = min(np.max(all_velocities[:, i]), q3 + 1.5 * iqr)
#
#         boxplot_data.append(
#             {
#                 "q1": q1,
#                 "median": median,
#                 "q3": q3,
#                 "whisker_low": whisker_low,
#                 "whisker_high": whisker_high,
#                 "mean": mean_velocities[i],
#             }
#         )
#
#     # Create the boxplots with custom colors
#     for i in range(n_layers):
#         # Create the box plot with custom styling
#         box = ax_profile.boxplot(
#             all_velocities[:, i],
#             positions=[mean_depths[i]],
#             vert=False,
#             widths=0.2,
#             patch_artist=True,
#             showfliers=False,
#             boxprops=dict(
#                 facecolor=boxplot_color,
#                 alpha=boxplot_alpha,
#                 edgecolor=boxplot_edge_color,
#             ),
#             medianprops=dict(color=median_line_color, linewidth=1.5),
#             whiskerprops=dict(color=whisker_color, linewidth=whisker_width),
#             capprops=dict(color=whisker_color),
#         )
#
#     # Plot the velocity profile with markers using the mean values
#     ax_profile.plot(
#         mean_velocities,
#         mean_depths,
#         "-",
#         linewidth=2,
#         label="Mean Velocity Profile",
#         marker=".",
#         markersize=10,
#         color=profile_line_color,
#         markerfacecolor=profile_marker_color,
#         zorder=10,
#     )
#
#     # Create custom legend entries for box plot elements
#     ax_profile.plot(
#         [], [], "-", color=whisker_color, label="Whiskers: Min/Max (no outliers)"
#     )
#     ax_profile.plot(
#         [],
#         [],
#         "s",
#         color=boxplot_edge_color,
#         markerfacecolor=boxplot_color,
#         alpha=boxplot_alpha,
#         markersize=8,
#         label="Quartiles: 25th/75th percentile",
#     )
#     ax_profile.plot([], [], "-", color=median_line_color, linewidth=1.5, label="Median")
#
#     # Move legend to the bottom of the plot, outside the axes
#     ax_profile.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
#
#     # Set labels and title for velocity profile
#     ax_profile.set_xlabel("Sea Water Speed [m/s]")
#     ax_profile.set_ylabel("Depth [m]")
#     ax_profile.set_title("Velocity Profile with Box Plots")
#
#     # Create custom y-tick labels with depth ranges
#     y_ticks = mean_depths
#     y_tick_labels = [
#         f"{min_depths[i]:.1f} - {max_depths[i]:.1f}" for i in range(n_layers)
#     ]
#
#     ax_profile.set_yticks(y_ticks)
#     ax_profile.set_yticklabels(y_tick_labels)
#
#     # Invert y-axis so that depth increases downward
#     ax_profile.invert_yaxis()
#     ax_profile.grid(True)
#
#     # Create the scatter plot with quadratic fit lines for mean and max speed
#     # Flatten arrays for scatter plot
#     flat_depths = all_depths.flatten()
#     flat_velocities = all_velocities.flatten()
#     flat_directions = all_directions.flatten()
#
#     # Create scatter plot
#     scatter = ax_scatter.scatter(
#         flat_velocities,
#         flat_depths,
#         alpha=1.0,
#         c=flat_directions,
#         # cmap="viridis",
#         cmap="twilight",
#         s=3,
#         edgecolor="none",
#     )
#
#     # Add colorbar
#     cbar = plt.colorbar(scatter, ax=ax_scatter)
#     cbar.set_label("Direction")
#
#     # Calculate mean and max velocities per layer
#     layer_mean_speeds = np.mean(all_velocities, axis=0)  # Mean speed at each layer
#     layer_max_speeds = np.max(all_velocities, axis=0)  # Max speed at each layer
#
#     # Use mean depths for each layer
#     layer_depths = mean_depths
#
#     # Fit quadratic models based on the layer data
#     mean_coefs = np.polyfit(layer_depths, layer_mean_speeds, 2)
#     a_mean, b_mean, c_mean = mean_coefs
#
#     max_coefs = np.polyfit(layer_depths, layer_max_speeds, 2)
#     a_max, b_max, c_max = max_coefs
#
#     # Generate x values for plotting smooth curves
#     line_x = np.linspace(min(flat_depths), max(flat_depths), 100)
#
#     # Generate quadratic model predictions
#     mean_speed_pred = a_mean * line_x**2 + b_mean * line_x + c_mean
#     max_speed_pred = a_max * line_x**2 + b_max * line_x + c_max
#
#     # Plot the fit lines
#     ax_scatter.plot(
#         mean_speed_pred,
#         line_x,
#         color=sns.color_palette()[0],
#         linewidth=2,
#         label="Mean Speed (Quadratic Fit)",
#     )
#
#     ax_scatter.plot(
#         max_speed_pred,
#         line_x,
#         color=sns.color_palette()[3],
#         linewidth=2,
#         linestyle="--",
#         label="Max Speed (Quadratic Fit)",
#     )
#
#     # Plot the layer mean and max points
#     ax_scatter.scatter(
#         layer_mean_speeds,
#         layer_depths,
#         color=sns.color_palette()[0],
#         s=50,
#         zorder=5,
#         label="Layer Mean Speeds",
#     )
#
#     ax_scatter.scatter(
#         layer_max_speeds,
#         layer_depths,
#         color=sns.color_palette()[3],
#         s=50,
#         zorder=5,
#         marker="s",
#         label="Layer Max Speeds",
#     )
#
#     # Add equation text to place below the plot
#     mean_eq = (
#         f"Mean Speed = {a_mean:.4f} × Depth² + {b_mean:.4f} × Depth + {c_mean:.4f}"
#     )
#     max_eq = f"Max Speed = {a_max:.4f} × Depth² + {b_max:.4f} × Depth + {c_max:.4f}"
#
#     # Set labels and title
#     ax_scatter.set_xlabel("Sea Water Speed [m/s]")
#     ax_scatter.set_ylabel("Depth [m]")
#     ax_scatter.set_title("Depth vs. Speed Scatter Plot")
#
#     # Move legend to the bottom of the plot, outside the axes
#     ax_scatter.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
#
#     # Add regression equations below the plot
#     fig.text(
#         0.5,
#         0.02,
#         f"{mean_eq}\n{max_eq}",
#         ha="center",
#         fontsize=10,
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
#     )
#
#     # Invert y-axis to match profile plot
#     ax_scatter.invert_yaxis()
#
#     # Add gridlines
#     ax_scatter.grid(True, linestyle="--", alpha=0.7)
#
#     # Remove the placeholder axes and create subplot grids
#     ax_hist.remove()
#     hist_grid = top_row[2].subgridspec(n_layers, 1, hspace=0.3)
#
#     ax_dir.remove()
#     dir_grid = top_row[3].subgridspec(n_layers, 1, hspace=0.3)
#
#     ax_depth.remove()
#     depth_grid = top_row[4].subgridspec(n_layers, 1, hspace=0.3)
#
#     # Create proper histograms for each depth layer
#     hist_axes = []
#     dir_axes = []
#     depth_axes = []
#     n_bins = 50
#     vel_bin_edges = np.linspace(
#         np.min(all_velocities) * 0.9, np.max(all_velocities) * 1.1, n_bins + 1
#     )
#     dir_bin_edges = np.linspace(0, 360, n_bins + 1)
#
#     # Calculate depth bin edges
#     depth_range = np.max(all_depths) - np.min(all_depths)
#     depth_bin_edges = np.linspace(
#         np.min(all_depths) * 0.95, np.max(all_depths) * 1.05, n_bins + 1
#     )
#
#     for i in range(n_layers):
#         # Create subplot for this velocity histogram
#         ax_vel = fig.add_subplot(hist_grid[i])
#         hist_axes.append(ax_vel)
#
#         # Create subplot for this direction histogram
#         ax_dir = fig.add_subplot(dir_grid[i])
#         dir_axes.append(ax_dir)
#
#         # Create subplot for depth histogram
#         ax_dep = fig.add_subplot(depth_grid[i])
#         depth_axes.append(ax_dep)
#
#         # Extract data for this layer
#         layer_velocities = all_velocities[:, i]
#         layer_directions = all_directions[:, i]
#         layer_depths = all_depths[:, i]
#         layer_min_depth = min_depths[i]
#         layer_max_depth = max_depths[i]
#
#         # Create velocity histogram with custom colors
#         ax_vel.hist(
#             layer_velocities,
#             bins=vel_bin_edges,
#             color=histogram_color,
#             edgecolor=histogram_edge_color,
#             alpha=histogram_alpha,
#         )
#
#         # Create direction histogram with custom colors
#         ax_dir.hist(
#             layer_directions,
#             bins=dir_bin_edges,
#             color=direction_hist_color,
#             edgecolor=direction_hist_edge_color,
#             alpha=direction_hist_alpha,
#         )
#
#         # Create depth histogram with custom colors
#         ax_dep.hist(
#             layer_depths,
#             bins=depth_bin_edges,
#             color=depth_hist_color,
#             edgecolor=depth_hist_edge_color,
#             alpha=depth_hist_alpha,
#         )
#
#         # Add depth range label to all histograms
#         depth_label = f"Depth: {layer_min_depth:.1f} - {layer_max_depth:.1f} m"
#         ax_vel.text(
#             0.02,
#             0.8,
#             depth_label,
#             transform=ax_vel.transAxes,
#             fontsize=8,
#             bbox=dict(facecolor="white", alpha=0.7),
#         )
#
#         ax_dir.text(
#             0.02,
#             0.8,
#             depth_label,
#             transform=ax_dir.transAxes,
#             fontsize=8,
#             bbox=dict(facecolor="white", alpha=0.7),
#         )
#
#         ax_dep.text(
#             0.02,
#             0.8,
#             depth_label,
#             transform=ax_dep.transAxes,
#             fontsize=8,
#             bbox=dict(facecolor="white", alpha=0.7),
#         )
#
#         # Set y-label only for middle histograms
#         if i == n_layers // 2:
#             ax_vel.set_ylabel("Count")
#             ax_dir.set_ylabel("Count")
#             ax_dep.set_ylabel("Count")
#
#         # Remove x labels for all but the bottom histogram
#         if i < n_layers - 1:
#             ax_vel.set_xticklabels([])
#             ax_dir.set_xticklabels([])
#             ax_dep.set_xticklabels([])
#         else:
#             ax_vel.set_xlabel("Sea Water Speed [m/s]")
#             ax_dir.set_xlabel("Sea Water Direction [degrees]")
#             ax_dep.set_xlabel("Depth [m]")
#
#         # Set consistent x limits
#         ax_vel.set_xlim(vel_bin_edges[0], vel_bin_edges[-1])
#         ax_dir.set_xlim(0, 360)
#         ax_dep.set_xlim(depth_bin_edges[0], depth_bin_edges[-1])
#
#         # Add gridlines
#         ax_vel.grid(True, linestyle="--", alpha=0.6)
#         ax_dir.grid(True, linestyle="--", alpha=0.6)
#         ax_dep.grid(True, linestyle="--", alpha=0.6)
#
#     # Add titles to the top histograms
#     hist_axes[0].set_title("Velocity Distributions by Depth")
#     dir_axes[0].set_title("Direction Distributions by Depth")
#     depth_axes[0].set_title("Depth Distributions by Layer")
#
#     # Adjust figure to make room for legends and equations
#     fig.subplots_adjust(bottom=0.20)
#
#     return fig


def plot_tidal_harmonic_analysis(df, layer=4, n_components=5):
    """
    Perform tidal harmonic analysis with visualization.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing tidal data with DatetimeIndex
    layer : int
        Depth layer to analyze (0-9, default is middle layer 4)
    n_components : int
        Number of harmonic components to extract

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """

    # Extract data
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    u = df[f"u_layer_{layer}"].values
    v = df[f"v_layer_{layer}"].values
    t = df.index.to_numpy()  # Get timestamps as numpy array

    # Calculate harmonic constituents
    coef = solve(t, u, v, lat=df["lat_center"].iloc[0], conf_int="linear")

    # Extract the major constituents
    major_names = coef["name"][:n_components]
    major_freqs = coef["aux"]["frq"][:n_components]

    # Convert frequencies to periods in hours
    major_periods = []
    for freq in major_freqs:
        # Convert from radians per second to hours
        period_hours = 2 * np.pi / (freq * 3600)
        major_periods.append(period_hours)

    major_amps = np.sqrt(
        coef["Lsmaj"][:n_components] ** 2 + coef["Lsmin"][:n_components] ** 2
    )

    # Create a figure with 3 rows
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # First row: Tidal current speed time series
    ax_speed = fig.add_subplot(gs[0])

    # Second row: FFT analysis
    ax_fft = fig.add_subplot(gs[1])

    # Third row: Tidal constituents
    ax_harm = fig.add_subplot(gs[2])

    # Plot limit for time series
    plot_limit = min(1000, len(df))
    t_plot = df.index[:plot_limit]
    # t_plot = df.index

    # Reconstruct tidal signal
    recon = reconstruct(t_plot, coef)

    # Get the reconstructed components
    if hasattr(recon, "h"):
        # For non-vector case
        speed_recon = recon.h
    else:
        # For vector case
        u_recon = recon.u
        v_recon = recon.v
        speed_recon = np.sqrt(u_recon**2 + v_recon**2)

    # 1. Plot speed time series
    ax_speed.plot(t_plot, speeds[:plot_limit], "k-", alpha=0.7, label="Original")
    ax_speed.plot(t_plot, speed_recon, "r-", label="Reconstructed")
    ax_speed.set_ylabel("Current Speed (m/s)")
    ax_speed.set_title("Tidal Current Speed")
    ax_speed.legend()
    ax_speed.grid(True)

    ax_speed.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Add annotation to show fit quality for speed
    rmse_speed = np.sqrt(np.mean((speeds[:plot_limit] - speed_recon) ** 2))
    r2_speed = 1 - np.sum((speeds[:plot_limit] - speed_recon) ** 2) / np.sum(
        (speeds[:plot_limit] - np.mean(speeds[:plot_limit])) ** 2
    )
    ax_speed.text(
        0.02,
        0.95,
        f"RMSE: {rmse_speed:.3f} m/s\nR²: {r2_speed:.3f}",
        transform=ax_speed.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # 2. Plot FFT analysis for speed with linear scale
    # Determine sampling interval
    if len(t) > 1:
        # Calculate time differences and get the average
        time_diffs = np.diff(t.astype("datetime64[s]").astype(float))
        dt = np.mean(time_diffs) / 86400  # Convert seconds to days
    else:
        dt = 1 / 48  # Default to half-hourly data (48 samples per day)

    # Perform FFT
    N = len(speeds)
    yf = fft(speeds)
    xf = fftfreq(N, dt)[: N // 2]  # Frequency in cycles per day

    # Convert to periods in hours and plot only positive frequencies
    periods = 24 / xf  # Convert frequency (cycles/day) to period (hours)
    power = 2.0 / N * np.abs(yf[: N // 2])

    # Filter to periods between 1 and 30 hours for clarity with linear scale
    period_mask = (periods >= 1) & (periods <= 30)
    filtered_periods = periods[period_mask]
    filtered_power = power[period_mask]

    # Plot the FFT with linear scale
    ax_fft.plot(filtered_periods, filtered_power, "-", linewidth=1.5, color="blue")
    ax_fft.set_xlabel("Period (hours)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("FFT Frequency Spectrum (Linear Scale)")
    ax_fft.grid(True, linestyle="--", alpha=0.7)

    # Add vertical lines for principal tidal periods
    principal_periods = {
        "M2": 12.42,  # Principal lunar semidiurnal
        "S2": 12.00,  # Principal solar semidiurnal
        "N2": 12.66,  # Larger lunar elliptic semidiurnal
        "K1": 23.93,  # Lunar diurnal
        "O1": 25.82,  # Lunar diurnal
        "M4": 6.21,  # Shallow water overtides of M2
        "M6": 4.14,  # Shallow water overtides of M2
    }

    colors = plt.cm.tab10(np.linspace(0, 1, len(principal_periods)))
    for i, (name, period) in enumerate(principal_periods.items()):
        if period >= 1 and period <= 30:  # Only show lines within our x-axis range
            ax_fft.axvline(
                x=period,
                color=colors[i],
                linestyle="--",
                alpha=0.7,
                label=f"{name} ({period:.2f}h)",
            )

    ax_fft.legend(loc="upper right")

    # 3. Constituent bar chart
    # Dictionary of constituent descriptions
    constituent_info = {
        "M2": "Principal lunar semidiurnal (12.42h)",
        "S2": "Principal solar semidiurnal (12.00h)",
        "N2": "Larger lunar elliptic semidiurnal (12.66h)",
        "K1": "Lunar diurnal (23.93h)",
        "O1": "Lunar diurnal (25.82h)",
        "M4": "Shallow water overtides of M2 (6.21h)",
        "M6": "Shallow water overtides of M2 (4.14h)",
        "K2": "Solar semidiurnal (11.97h)",
        "L2": "Smaller lunar elliptic semidiurnal (12.19h)",
        "P1": "Solar diurnal (24.07h)",
        "Q1": "Larger lunar elliptic diurnal (26.87h)",
        "MK3": "Shallow water terdiurnal (8.18h)",
        "MN4": "Shallow water quarter diurnal (6.27h)",
        "MS4": "Shallow water quarter diurnal (6.10h)",
    }

    # Plot the harmonic components with enhanced colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(major_names)))
    bars = ax_harm.bar(range(len(major_names)), major_amps, alpha=0.8, color=colors)

    # Improved label approach
    # First just set simple tick labels with constituent names
    ax_harm.set_xticks(range(len(major_names)))
    ax_harm.set_xticklabels(major_names)

    # Then add descriptive text below the x-axis
    for i, name in enumerate(major_names):
        if name in constituent_info:
            # Get period
            period = major_periods[i]
            # Extract just the description part (remove the period)
            description = constituent_info[name].split(" (")[0]

            # Add description as text below the axis with appropriate positioning
            ax_harm.text(
                i,
                -0.08,
                description,
                ha="center",
                va="top",
                transform=ax_harm.get_xaxis_transform(),
                fontsize=9,
                rotation=0,
            )

            # Add period on top of the bars
            ax_harm.text(
                i,
                major_amps[i] + 0.02,
                f"Period: {period:.2f}h",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    # Add more space at the bottom for labels
    ax_harm.set_ylabel("Amplitude (m/s)")
    ax_harm.set_title("Major Tidal Constituents")

    # Add subtitle about strongest constituents
    ax_harm.text(
        0.5,
        -0.22,
        "The strongest constituents indicate primary tidal forces affecting this location.",
        ha="center",
        va="center",
        transform=ax_harm.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.7),
    )

    ax_harm.grid(True, axis="y")

    # Get the average depth value for this layer
    depth_value = df[f"vap_sigma_depth_layer_{layer}"].mean()

    # Add layer depth information to the title
    plt.suptitle(
        f"Tidal Harmonic Analysis at {depth_value:.1f}m Depth", fontsize=14, y=0.98
    )

    # Adjust layout to make room for the descriptive labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Slight bottom margin for subtitle

    return fig


def plot_velocity_profile(df, timestamp_idx=0):
    """
    Plot a simple velocity profile for a specific timestamp.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    timestamp_idx : int
        Index of the timestamp to plot

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract depth and velocity data for the specified timestamp
    depths = []
    velocities = []

    # Number of layers (assuming 10 layers as in your data)
    n_layers = 10

    for layer in range(n_layers):
        depths.append(df.iloc[timestamp_idx][f"vap_sigma_depth_layer_{layer}"])
        velocities.append(df.iloc[timestamp_idx][f"vap_sea_water_speed_layer_{layer}"])

    # Create the plot
    ax.plot(velocities, depths, "o-", linewidth=2, markersize=8)

    # Customize the plot
    ax.set_xlabel("Sea Water Speed [m/s]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Velocity Profile at {df.index[timestamp_idx]}")

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_current_rose(df, layer=0, bins=16, vmax=None):
    """
    Create a current rose plot to visualize the distribution of current speeds and directions.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int
        The depth layer to visualize (0-9)
    bins : int
        Number of direction bins to use
    vmax : float or None
        Maximum velocity for color scaling, if None will use data max

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    """
    # Extract the data for the specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    directions = df[f"vap_sea_water_to_direction_layer_{layer}"].values
    depth = df[f"vap_sigma_depth_layer_{layer}"].iloc[0]  # Get depth from first row

    # Convert directions to radians for polar plot
    directions_rad = np.deg2rad(directions)

    # Set up bins for directions
    bin_width = 2 * np.pi / bins
    direction_bins = np.linspace(0, 2 * np.pi, bins + 1)

    # Find the maximum velocity if not provided
    if vmax is None:
        vmax = np.ceil(speeds.max() * 10) / 10  # Round up to nearest 0.1

    # Create velocity bins (5 bins from 0 to vmax)
    vel_bins = np.linspace(0, vmax, 6)

    # Initialize arrays to hold frequency counts
    freq = np.zeros((len(vel_bins) - 1, bins))

    # Count occurrences in each bin
    for i in range(len(vel_bins) - 1):
        mask = (speeds >= vel_bins[i]) & (speeds < vel_bins[i + 1])
        for j in range(bins):
            dir_mask = (directions_rad >= direction_bins[j]) & (
                directions_rad < direction_bins[j + 1]
            )
            freq[i, j] = np.sum(mask & dir_mask)

    # Normalize to get percentages
    freq = freq / len(speeds) * 100

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    # Plot each velocity bin as a separate set of bars
    width = bin_width * 0.8  # Slightly narrow bars for better visibility

    # Use a colormap suitable for velocity
    colors = plt.cm.viridis(np.linspace(0, 1, len(vel_bins) - 1))

    # Plot each velocity bin
    for i in range(len(vel_bins) - 1):
        bars = ax.bar(
            direction_bins[:-1],
            freq[i],
            width=width,
            bottom=0.0 if i == 0 else np.sum(freq[:i], axis=0),
            color=colors[i],
            alpha=0.8,
            label=f"{vel_bins[i]:.1f}-{vel_bins[i + 1]:.1f} m/s",
        )

    # Customize the plot
    ax.set_theta_zero_location("N")  # 0 degrees at the top (North)
    ax.set_theta_direction(-1)  # Clockwise

    # Set labels at cardinal and intercardinal directions
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    # Add title and legend
    plt.title(f"Current Rose at {depth:.1f}m Depth (Layer {layer})")
    plt.legend(loc="lower right", bbox_to_anchor=(1.1, -0.1))

    return fig


def plot_tidal_time_series(df, start_date=None, end_date=None, layers=None):
    """
    Create time series plots for tidal current data.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    start_date, end_date : datetime or str, optional
        Date range to plot (if None, uses full range)
    layers : list of int, optional
        List of depth layers to plot (if None, plots layers 0, 4, and 9)

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plots
    """
    # Filter by date range if specified
    if start_date is not None and end_date is not None:
        plot_df = df.loc[start_date:end_date].copy()
    else:
        plot_df = df.copy()

    # Default layers if not specified (surface, middle, bottom)
    if layers is None:
        layers = [0, 4, 9]

    # Set up a 3-panel figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    # Panel 1: Current Speed
    for i, layer in enumerate(layers):
        axs[0].plot(
            plot_df.index,
            plot_df[f"vap_sea_water_speed_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer} (~{plot_df[f'vap_sigma_depth_layer_{layer}'].iloc[0]:.1f}m)",
        )

    axs[0].set_ylabel("Current Speed (m/s)")
    axs[0].set_title("Tidal Current Speed Time Series")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(loc="upper right")

    # Panel 2: Current Direction
    for i, layer in enumerate(layers):
        axs[1].plot(
            plot_df.index,
            plot_df[f"vap_sea_water_to_direction_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer}",
        )

    axs[1].set_ylabel("Direction (degrees)")
    axs[1].set_title("Tidal Current Direction Time Series")
    axs[1].set_yticks(np.arange(0, 361, 45))
    axs[1].set_yticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    axs[1].set_ylim(0, 360)
    axs[1].grid(True, linestyle="--", alpha=0.7)

    # Panel 3: Power Density
    for i, layer in enumerate(layers):
        axs[2].plot(
            plot_df.index,
            plot_df[f"vap_sea_water_power_density_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer}",
        )

    axs[2].set_ylabel("Power Density (W/m²)")
    axs[2].set_title("Tidal Current Power Density Time Series")
    axs[2].grid(True, linestyle="--", alpha=0.7)

    # X-axis formatting
    axs[2].set_xlabel("Date/Time")
    plt.gcf().autofmt_xdate()
    date_format = mdates.DateFormatter("%Y-%m-%d %H:%M")
    axs[2].xaxis.set_major_formatter(date_format)

    # Add major ticks for each day
    axs[2].xaxis.set_major_locator(mdates.DayLocator())

    # Add minor ticks every 6 hours
    axs[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axs[2].grid(True, which="minor", linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


def old_plot_velocity_exceedance(df, layers=None, key_percentiles=None):
    """
    Create velocity exceedance curves for tidal current data.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layers : list of int, optional
        List of depth layers to plot (if None, uses layers 0, 2, 5, and 9)
    key_percentiles : list of float, optional
        List of percentiles to highlight (if None, uses [50, 75, 90, 95])

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plots
    stats : dict
        Dictionary containing the exceedance statistics for each layer
    """
    # Default layers if not specified
    if layers is None:
        # layers = [0, 2, 5, 9]
        layers = range(10)

    # Default key percentiles
    if key_percentiles is None:
        key_percentiles = [
            50,
            75,
            90,
            95,
        ]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Color palette
    # colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    colors = cmocean.cm.deep(np.linspace(0, 1, len(layers)))

    # Store statistics
    stats = {}

    # Plot each layer
    for i, layer in enumerate(layers):
        # Extract velocity data for this layer
        velocities = df[f"vap_sea_water_speed_layer_{layer}"].values
        depth = df[f"vap_sigma_depth_layer_{layer}"].iloc[0]  # Get depth from first row

        # Sort velocities in descending order
        sorted_velocities = np.sort(velocities)[::-1]

        # Calculate exceedance probabilities
        p = np.arange(1, len(sorted_velocities) + 1) / len(sorted_velocities) * 100

        # Plot the exceedance curve
        ax.plot(
            p,
            sorted_velocities,
            "-",
            color=colors[i],
            label=f"Layer {layer} (~{depth:.1f}m)",
        )

        # Calculate statistics for key percentiles
        layer_stats = {}
        for percentile in key_percentiles:
            # Find the velocity that is exceeded x% of the time
            exceeded_velocity = np.percentile(velocities, 100 - percentile)
            layer_stats[f"{percentile}%"] = exceeded_velocity

            # Add markers for key percentiles
            ax.plot(
                percentile,
                exceeded_velocity,
                "o",
                color=colors[i],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

            # Add text annotation for the first layer only to avoid clutter
            if i == 0:
                ax.annotate(
                    f"{percentile}%: {exceeded_velocity:.2f} m/s",
                    xy=(percentile, exceeded_velocity),
                    xytext=(percentile + 2, exceeded_velocity),
                    arrowprops=dict(arrowstyle="->"),
                )

        # Calculate additional statistics
        layer_stats["mean"] = np.mean(velocities)
        layer_stats["max"] = np.max(velocities)

        # Store layer statistics
        stats[f"Layer {layer}"] = layer_stats

    # Customize the plot
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Current Speed (m/s)")
    ax.set_title("Velocity Exceedance Curves")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    # Add a vertical line at 50% exceedance (median)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig, stats


def plot_velocity_exceedance(df, layers=None, key_percentiles=None):
    """
    Create velocity exceedance curves for tidal current data following IEC 62600-201.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layers : list of int, optional
        List of depth layers to plot (if None, uses layers 0, 2, 5, and 9)
    key_percentiles : list of float, optional
        List of percentiles to highlight (if None, uses [50, 75, 90, 95])

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plots
    stats : dict
        Dictionary containing the exceedance statistics for each layer
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cmocean

    # Default layers if not specified
    if layers is None:
        # layers = [0, 2, 5, 9]  # Using subset for cleaner visualization
        layers = range(10)

    # Default key percentiles
    if key_percentiles is None:
        # key_percentiles = [50, 75, 95]
        max_key_percentiles = [50, 25, 10, 5, 1, 0.001]
        min_key_percentiles = [50, 25, 10]
        key_percentiles = list(set(max_key_percentiles + min_key_percentiles))

    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Color palette
    colors = cmocean.cm.deep(np.linspace(0, 1, len(layers)))

    # Store statistics
    stats = {}

    # Track min/max values for annotations
    percentile_values = {
        p: {
            "min": float("inf"),
            "max": -float("inf"),
            "min_layer": None,
            "max_layer": None,
        }
        for p in key_percentiles
    }

    # Plot each layer
    for i, layer in enumerate(layers):
        # Extract velocity data for this layer
        velocities = df[f"vap_sea_water_speed_layer_{layer}"].values
        depth = df[f"vap_sigma_depth_layer_{layer}"].iloc[0]  # Get depth from first row

        # Sort velocities in descending order
        sorted_velocities = np.sort(velocities)[::-1]

        # Calculate exceedance probabilities (0-100%)
        p = np.arange(1, len(sorted_velocities) + 1) / len(sorted_velocities) * 100

        # Plot the exceedance curve - FIXED: x=velocity, y=probability
        ax.plot(
            sorted_velocities,
            p,
            "-",
            color=colors[i],
            label=f"Layer {layer} (~{depth:.1f}m)",
            linewidth=2,
        )

        # Calculate statistics for key percentiles
        layer_stats = {}
        for percentile in key_percentiles:
            # Find the velocity that is exceeded x% of the time
            exceeded_velocity = np.percentile(velocities, 100 - percentile)
            layer_stats[f"{percentile}%"] = exceeded_velocity

            # Track min/max values across all layers for this percentile
            if exceeded_velocity < percentile_values[percentile]["min"]:
                percentile_values[percentile]["min"] = exceeded_velocity
                percentile_values[percentile]["min_layer"] = layer
            if exceeded_velocity > percentile_values[percentile]["max"]:
                percentile_values[percentile]["max"] = exceeded_velocity
                percentile_values[percentile]["max_layer"] = layer

        # Calculate additional statistics
        layer_stats["mean"] = np.mean(velocities)
        layer_stats["max"] = np.max(velocities)

        # Store layer statistics
        stats[f"Layer {layer}"] = layer_stats

    # Add markers and annotations for highest values at max percentiles only
    for percentile in max_key_percentiles:
        max_val = percentile_values[percentile]["max"]
        max_layer = percentile_values[percentile]["max_layer"]

        # Add marker for max value at this percentile
        ax.plot(
            max_val,
            percentile,
            "o",
            color=colors[max_layer],
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Annotate maximum value
        ax.annotate(
            f"{percentile}% Max: {max_val:.2f} m/s\n(Layer {max_layer})",
            xy=(max_val, percentile),
            xytext=(max_val + 0.1, percentile + 8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9,
            ha="center",
        )

    # Add markers and annotations for lowest values at min percentiles only
    for percentile in min_key_percentiles:
        min_val = percentile_values[percentile]["min"]
        min_layer = percentile_values[percentile]["min_layer"]

        # Add marker for min value at this percentile
        ax.plot(
            min_val,
            percentile,
            "o",
            color=colors[min_layer],
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Annotate minimum value
        ax.annotate(
            f"{percentile}% Min: {min_val:.2f} m/s\n(Layer {min_layer})",
            xy=(min_val, percentile),
            xytext=(min_val - 0.1, percentile - 8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="blue"),
            fontsize=9,
            ha="center",
        )

    ax.set_xlabel("Sea Water Speed [m/s]", fontsize=14)
    ax.set_ylabel("Probability of Exceedance [%]", fontsize=14)
    # ax.set_title("Velocity Exceedance Prob (IEC 62600-201)", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=12)

    # Add horizontal lines at key percentiles
    # for percentile in max_key_p:
    #     ax.axhline(y=percentile, color="gray", linestyle="--", alpha=0.3)

    # Set reasonable axis limits
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    plt.tight_layout()

    return fig, stats


def analyze_power_density(df, layer=None, rho=1025, cutout_speed=0.5, rated_speed=1.5):
    """
    Analyze tidal current power density and create visualizations for tidal energy assessment.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int or None
        Depth layer to analyze (if None, uses the layer with maximum mean power)
    rho : float
        Water density in kg/m³ (default: 1025 kg/m³ for seawater)
    cutout_speed : float
        Cut-in speed for turbine operation in m/s (default: 0.5 m/s)
    rated_speed : float
        Rated speed for turbine operation in m/s (default: 2.0 m/s)

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the analysis plots
    summary : dict
        Dictionary containing the power analysis summary
    """
    # If layer is not specified, find the layer with maximum mean power density
    if layer is None:
        mean_powers = []
        for i in range(10):  # Assuming 10 layers as in your data
            mean_powers.append(df[f"vap_sea_water_power_density_layer_{i}"].mean())
        layer = np.argmax(mean_powers)
        print(f"Selected layer {layer} with highest mean power density")

    # Extract data for the specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    power_densities = df[f"vap_sea_water_power_density_layer_{layer}"].values
    depth = df[f"vap_sigma_depth_layer_{layer}"].iloc[0]  # Get depth from first row

    # Create a 2x2 figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Histogram of current speeds
    bins = np.linspace(0, np.max(speeds) * 1.05, 30)
    axs[0, 0].hist(speeds, bins=bins, alpha=0.7, color="royalblue", edgecolor="black")
    axs[0, 0].set_xlabel("Current Speed (m/s)")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].set_title(
        f"Distribution of Current Speeds at {depth:.1f}m (Layer {layer})"
    )
    axs[0, 0].grid(True, linestyle="--", alpha=0.7)

    # Add vertical lines for cutout and rated speeds
    axs[0, 0].axvline(
        x=cutout_speed, color="r", linestyle="--", label=f"Cut-in: {cutout_speed} m/s"
    )
    axs[0, 0].axvline(
        x=rated_speed, color="g", linestyle="--", label=f"Rated: {rated_speed} m/s"
    )
    axs[0, 0].legend()

    # Plot 2: Power density vs current speed (P = 0.5 * rho * V³)
    v_range = np.linspace(0, np.max(speeds) * 1.1, 100)
    p_range = 0.5 * rho * v_range**3

    axs[0, 1].plot(v_range, p_range, "k-", label="P = 0.5 * ρ * V³")
    axs[0, 1].scatter(
        speeds, power_densities, alpha=0.3, color="royalblue", label="Observed data"
    )

    axs[0, 1].set_xlabel("Current Speed (m/s)")
    axs[0, 1].set_ylabel("Power Density (W/m²)")
    axs[0, 1].set_title("Power Density vs Current Speed")
    axs[0, 1].grid(True, linestyle="--", alpha=0.7)
    axs[0, 1].legend()

    # Plot 3: Histogram of power density
    bins = np.linspace(0, np.max(power_densities) * 1.05, 30)
    axs[1, 0].hist(
        power_densities, bins=bins, alpha=0.7, color="green", edgecolor="black"
    )
    axs[1, 0].set_xlabel("Power Density (W/m²)")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Distribution of Power Density")
    axs[1, 0].grid(True, linestyle="--", alpha=0.7)

    # Plot 4: Calculate theoretical turbine power output
    # Define a simple turbine power curve
    def turbine_power(v, cut_in=cutout_speed, rated=rated_speed):
        p = np.zeros_like(v)
        # Below cut-in: no power
        # Between cut-in and rated: cubic scaling
        # Above rated: constant power
        mask1 = (v >= cut_in) & (v < rated)
        mask2 = v >= rated

        # Normalize to 1.0 at rated speed
        p[mask1] = ((v[mask1] - cut_in) / (rated - cut_in)) ** 3
        p[mask2] = 1.0

        return p

    # Calculate capacity factor
    power_curve = turbine_power(speeds)
    capacity_factor = np.mean(power_curve) * 100  # as percentage

    # Calculate theoretical power duration curve
    sorted_power = np.sort(power_curve)[::-1]
    p = np.arange(1, len(sorted_power) + 1) / len(sorted_power) * 100

    axs[1, 1].plot(p, sorted_power, "g-")
    axs[1, 1].set_xlabel("Exceedance Probability (%)")
    axs[1, 1].set_ylabel("Normalized Power Output")
    axs[1, 1].set_title(f"Power Duration Curve (CF: {capacity_factor:.1f}%)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.7)

    # Calculate additional statistics
    usable_time = 100 * np.mean(speeds >= cutout_speed)  # % of time above cut-in
    rated_time = 100 * np.mean(speeds >= rated_speed)  # % of time at rated power

    # Mean and max power density
    mean_pd = np.mean(power_densities)
    max_pd = np.max(power_densities)

    # Compile summary statistics
    summary = {
        "layer": layer,
        "depth": depth,
        "mean_speed": np.mean(speeds),
        "max_speed": np.max(speeds),
        "mean_power_density": mean_pd,
        "max_power_density": max_pd,
        "usable_time_pct": usable_time,
        "rated_time_pct": rated_time,
        "capacity_factor": capacity_factor,
    }

    # Add summary text to the figure
    summary_text = (
        f"Layer {layer} at {depth:.1f}m depth\n"
        f"Mean Speed: {summary['mean_speed']:.2f} m/s\n"
        f"Max Speed: {summary['max_speed']:.2f} m/s\n"
        f"Mean Power Density: {summary['mean_power_density']:.2f} W/m²\n"
        f"Max Power Density: {summary['max_power_density']:.2f} W/m²\n"
        f"Time above cut-in speed: {summary['usable_time_pct']:.1f}%\n"
        f"Time at rated power: {summary['rated_time_pct']:.1f}%\n"
        f"Capacity Factor: {summary['capacity_factor']:.1f}%"
    )

    fig.text(
        0.5,
        0.01,
        summary_text,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Tidal Energy Resource Assessment", fontsize=16, y=0.98)

    return fig, summary


def plot_tidal_velocity_profile(df, timestamp_index=None):
    """
    Plot a simple vertical profile of tidal currents at a specific time.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    timestamp_index : int, optional
        Index of the timestamp to plot, defaults to the first timestamp

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if timestamp_index is None:
        timestamp_index = 0

    # Extract data for the specified timestamp
    data = df.iloc[timestamp_index]

    # Extract depths and velocities for all layers
    depths = [data[f"vap_sigma_depth_layer_{i}"] for i in range(10)]
    speeds = [data[f"vap_sea_water_speed_layer_{i}"] for i in range(10)]
    directions = [data[f"vap_sea_water_to_direction_layer_{i}"] for i in range(10)]

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Plot speed profile
    ax1.plot(speeds, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Current Speed (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()  # Deeper depths at bottom of plot

    # Plot direction profile
    ax2.plot(directions, depths, "o-", linewidth=2, markersize=8, color="orange")
    ax2.set_xlabel("Current Direction (degrees)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Direction Profile")
    ax2.grid(True)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 360)

    # Add timestamp as suptitle
    if hasattr(df.index, "values"):
        plt.suptitle(f"Profiles at {df.index[timestamp_index]}")

    plt.tight_layout()
    return fig


def plot_power_density_profile(df, timestamp_index=None):
    """
    Plot the power density profile for tidal energy assessment.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    timestamp_index : int, optional
        Index of the timestamp to plot, defaults to the first timestamp

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if timestamp_index is None:
        timestamp_index = 0

    # Extract data for the specified timestamp
    data = df.iloc[timestamp_index]

    # Extract depths and power density for all layers
    depths = [data[f"vap_sigma_depth_layer_{i}"] for i in range(10)]
    power_density = [data[f"vap_sea_water_power_density_layer_{i}"] for i in range(10)]
    speeds = [data[f"vap_sea_water_speed_layer_{i}"] for i in range(10)]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a twin axis for speed
    ax2 = ax.twiny()

    # Plot power density
    ax.plot(
        power_density,
        depths,
        "o-",
        linewidth=2,
        markersize=8,
        color="red",
        label="Power Density",
    )
    ax.set_xlabel("Power Density (W/m²)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Tidal Power Density Profile")
    ax.grid(True)
    ax.invert_yaxis()

    # Plot speed on secondary x-axis
    ax2.plot(
        speeds,
        depths,
        "o--",
        linewidth=1.5,
        markersize=6,
        color="blue",
        label="Current Speed",
    )
    ax2.set_xlabel("Current Speed (m/s)")

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Add timestamp as suptitle
    if hasattr(df.index, "values"):
        plt.suptitle(f"Power Density at {df.index[timestamp_index]}")

    plt.tight_layout()
    return fig


def plot_tidal_time_series(df, start_index=0, end_index=None, layer=4):
    """
    Plot time series of tidal velocity and power density for a specific layer.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data with DatetimeIndex
    start_index, end_index : int
        Start and end indices for the time series
    layer : int
        Depth layer to plot (0-9, default is middle layer 4)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if end_index is None:
        end_index = min(start_index + 500, len(df) - 1)  # Default to 500 timesteps

    # Slice the dataframe for the specified time range
    df_slice = df.iloc[start_index:end_index]
    timestamps = df_slice.index

    # Extract data for the specified layer
    speeds = df_slice[f"vap_sea_water_speed_layer_{layer}"]
    directions = df_slice[f"vap_sea_water_to_direction_layer_{layer}"]
    power = df_slice[f"vap_sea_water_power_density_layer_{layer}"]
    depth = df_slice[f"vap_sigma_depth_layer_{layer}"].mean()  # Average depth

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot speed and direction
    ax1.plot(timestamps, speeds, linewidth=1.5, color="blue", label="Speed")

    # Create a second y-axis for direction
    ax1_dir = ax1.twinx()
    ax1_dir.plot(
        timestamps, directions, linewidth=1, color="green", alpha=0.7, label="Direction"
    )
    ax1_dir.set_ylabel("Direction (degrees)")
    ax1_dir.set_ylim(0, 360)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_dir.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Plot power density
    ax2.plot(timestamps, power, linewidth=1.5, color="red")
    ax2.set_ylabel("Power Density (W/m²)")
    ax2.set_xlabel("Time")

    # Format x-axis
    ax1.set_ylabel("Current Speed (m/s)")
    ax1.set_title(f"Tidal Current Speed and Direction at {depth:.1f}m Depth")
    ax2.set_title(f"Tidal Power Density at {depth:.1f}m Depth")

    # Format datetime x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    # Add gridlines
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def plot_tidal_rose(df, layer=4):
    """
    Create a current rose diagram showing the distribution of tidal current
    directions and speeds.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int
        Depth layer to analyze (0-9, default is middle layer 4)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """

    # Extract data for the specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    directions = df[f"vap_sea_water_to_direction_layer_{layer}"].values
    depth = df[f"vap_sigma_depth_layer_{layer}"].mean()  # Average depth

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect)
    fig.add_axes(ax)

    # Create the rose plot
    ax.bar(
        directions,
        speeds,
        normed=True,
        opening=0.8,
        edgecolor="white",
        cmap=plt.cm.viridis,  # Set colormap to viridis
    )

    # Set legend and labels
    ax.set_legend(title="Speed [m/s]")
    ax.set_title(f"Tidal Current Rose at {depth:.1f}m Depth")

    return fig


def plot_tidal_exceedance(df, layers=[0, 4, 9]):
    """
    Plot velocity and power exceedance curves for tidal energy development.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layers : list of int
        Depth layers to analyze

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    for i, layer in enumerate(layers):
        # Extract data for this layer
        speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
        power = df[f"vap_sea_water_power_density_layer_{layer}"].values
        depth = df[f"vap_sigma_depth_layer_{layer}"].mean()  # Average depth

        # Sort data in descending order
        speeds_sorted = np.sort(speeds)[::-1]
        power_sorted = np.sort(power)[::-1]

        # Calculate exceedance probabilities
        exceedance = np.arange(1, len(speeds) + 1) / len(speeds) * 100

        # Plot speed exceedance
        ax1.plot(
            exceedance,
            speeds_sorted,
            label=f"Depth: {depth:.1f}m",
            color=colors[i],
            linewidth=2,
        )

        # Plot power exceedance
        ax2.plot(
            exceedance,
            power_sorted,
            label=f"Depth: {depth:.1f}m",
            color=colors[i],
            linewidth=2,
        )

    # Add reference lines
    ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.7)
    ax1.axhline(y=2.0, color="r", linestyle="--", alpha=0.7)
    ax1.text(95, 1.0, "1.0 m/s", va="bottom", ha="right", color="r")
    ax1.text(95, 2.0, "2.0 m/s", va="bottom", ha="right", color="r")

    # Format axes
    ax1.set_xlabel("Exceedance Probability (%)")
    ax1.set_ylabel("Current Speed (m/s)")
    ax1.set_title("Tidal Current Speed Exceedance")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="best")

    ax2.set_xlabel("Exceedance Probability (%)")
    ax2.set_ylabel("Power Density (W/m²)")
    ax2.set_title("Tidal Power Density Exceedance")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="best")

    # Set log scale for power density
    ax2.set_yscale("log")

    plt.tight_layout()
    return fig


def create_tidal_resource_dashboard(df, timestamp_index=None):
    """
    Create a comprehensive dashboard for tidal resource characterization.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    timestamp_index : int, optional
        Index of the timestamp to analyze, defaults to the first timestamp

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if timestamp_index is None:
        timestamp_index = 0

    # Create the figure
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])

    # Extract data for the specified timestamp
    data = df.iloc[timestamp_index]

    # 1. Velocity Profile
    ax1 = fig.add_subplot(gs[0, 0])
    depths = [data[f"vap_sigma_depth_layer_{i}"] for i in range(10)]
    speeds = [data[f"vap_sea_water_speed_layer_{i}"] for i in range(10)]
    ax1.plot(speeds, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Current Speed (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()

    # 2. Power Density Profile
    ax2 = fig.add_subplot(gs[0, 1])
    power_density = [data[f"vap_sea_water_power_density_layer_{i}"] for i in range(10)]
    ax2.plot(power_density, depths, "o-", linewidth=2, markersize=8, color="red")
    ax2.set_xlabel("Power Density (W/m²)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Power Density Profile")
    ax2.grid(True)
    ax2.invert_yaxis()

    # 3. Velocity vs Depth Scatterplot
    ax3 = fig.add_subplot(gs[0, 2])
    # Flattened arrays for all days
    all_depths = np.array(
        [df[f"vap_sigma_depth_layer_{i}"] for i in range(10)]
    ).T.flatten()
    all_speeds = np.array(
        [df[f"vap_sea_water_speed_layer_{i}"] for i in range(10)]
    ).T.flatten()

    ax3.scatter(all_speeds, all_depths, s=10, alpha=0.3)
    ax3.set_xlabel("Current Speed (m/s)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Speed vs Depth Distribution")
    ax3.grid(True)
    ax3.invert_yaxis()

    # 4. Time Series Plot (speed at multiple depths)
    ax4 = fig.add_subplot(gs[1, :2])
    # Get a slice of the time series (around the selected timestamp)
    window = 100
    start_idx = max(0, timestamp_index - window)
    end_idx = min(len(df), timestamp_index + window)
    df_slice = df.iloc[start_idx:end_idx]

    for layer in [0, 4, 9]:  # Top, middle, bottom layers
        ax4.plot(
            df_slice.index,
            df_slice[f"vap_sea_water_speed_layer_{layer}"],
            label=f"Depth: {depths[layer]:.1f}m",
        )

    # Add vertical line at selected timestamp
    ax4.axvline(x=df.index[timestamp_index], color="r", linestyle="--")

    ax4.set_xlabel("Time")
    ax4.set_ylabel("Current Speed (m/s)")
    ax4.set_title("Tidal Current Speed Time Series")
    ax4.legend()
    ax4.grid(True)

    # 5. Direction Distribution (rose plot in cylindrical projection)
    ax5 = fig.add_subplot(gs[1, 2], polar=True)
    directions = np.array(
        [df[f"vap_sea_water_to_direction_layer_{4}"] for i in range(1)]
    ).flatten()
    speeds_for_dir = np.array(
        [df[f"vap_sea_water_speed_layer_{4}"] for i in range(1)]
    ).flatten()

    # Convert to radians
    dir_rad = np.radians(directions)

    # Create histogram
    bins = np.linspace(0, 2 * np.pi, 16 + 1)
    n, bins = np.histogram(dir_rad, bins=bins)
    width = bins[1] - bins[0]

    # Plot rose
    ax5.bar(bins[:-1], n, width=width, bottom=0.0, alpha=0.7)
    ax5.set_theta_zero_location("N")
    ax5.set_theta_direction(-1)  # Clockwise
    ax5.set_title("Current Direction Distribution")

    # 6. Exceedance Probability Curves
    ax6 = fig.add_subplot(gs[2, 0])
    # Calculate exceedance for the middle layer
    speeds_middle = df[f"vap_sea_water_speed_layer_{4}"].values
    speeds_sorted = np.sort(speeds_middle)[::-1]
    exceedance = np.arange(1, len(speeds_middle) + 1) / len(speeds_middle) * 100

    ax6.plot(exceedance, speeds_sorted, linewidth=2)
    ax6.set_xlabel("Exceedance Probability (%)")
    ax6.set_ylabel("Current Speed (m/s)")
    ax6.set_title("Speed Exceedance Curve")
    ax6.grid(True)

    # Add reference lines
    cut_in_speed = 0.5  # Example cut-in speed for a turbine
    rated_speed = 2.0  # Example rated speed

    ax6.axhline(y=cut_in_speed, color="g", linestyle="--")
    ax6.axhline(y=rated_speed, color="r", linestyle="--")
    ax6.text(95, cut_in_speed, f"Cut-in: {cut_in_speed} m/s", ha="right", va="bottom")
    ax6.text(95, rated_speed, f"Rated: {rated_speed} m/s", ha="right", va="bottom")

    # 7. Power Exceedance Curve (log scale)
    ax7 = fig.add_subplot(gs[2, 1])
    power_middle = df[f"vap_sea_water_power_density_layer_{4}"].values
    power_sorted = np.sort(power_middle)[::-1]

    ax7.plot(exceedance, power_sorted, linewidth=2, color="red")
    ax7.set_xlabel("Exceedance Probability (%)")
    ax7.set_ylabel("Power Density (W/m²)")
    ax7.set_title("Power Density Exceedance Curve")
    ax7.set_yscale("log")
    ax7.grid(True)

    # 8. Site Statistics and Resource Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")

    # Calculate statistics
    mean_speed = df["vap_water_column_mean_sea_water_speed"].mean()
    max_speed = df["vap_water_column_max_sea_water_speed"].max()
    p95_speed = df["vap_water_column_95th_percentile_sea_water_speed"].mean()

    mean_power = df["vap_water_column_mean_sea_water_power_density"].mean()
    max_power = df["vap_water_column_max_sea_water_power_density"].max()
    p95_power = df["vap_water_column_95th_percentile_sea_water_power_density"].mean()

    # Calculate speed at middle layer exceeding thresholds
    speeds_middle = df[f"vap_sea_water_speed_layer_{4}"].values
    pct_above_1ms = (speeds_middle >= 1.0).sum() / len(speeds_middle) * 100
    pct_above_2ms = (speeds_middle >= 2.0).sum() / len(speeds_middle) * 100

    # Add text info
    info_text = (
        "SITE RESOURCE SUMMARY\n"
        "=====================\n\n"
        f"Water Depth: {data['vap_sea_floor_depth']:.1f} m\n\n"
        f"Mean Speed: {mean_speed:.2f} m/s\n"
        f"Max Speed: {max_speed:.2f} m/s\n"
        f"95th Percentile Speed: {p95_speed:.2f} m/s\n\n"
        f"Mean Power Density: {mean_power:.2f} W/m²\n"
        f"Max Power Density: {max_power:.2f} W/m²\n"
        f"95th Percentile Power: {p95_power:.2f} W/m²\n\n"
        f"Time Above 1.0 m/s: {pct_above_1ms:.1f}%\n"
        f"Time Above 2.0 m/s: {pct_above_2ms:.1f}%\n\n"
        "Location: Piscataqua River, NH\n"
        f"Lat: {data['lat_center']:.4f}, Lon: {data['lon_center']:.4f}"
    )

    ax8.text(0, 1.0, info_text, va="top", fontfamily="monospace")

    # Add main title
    plt.suptitle(
        "Tidal Energy Resource Characterization Dashboard",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def plot_tidal_asymmetry(df, layer=4):
    """
    Analyze and visualize tidal asymmetry (flood vs ebb currents).

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int
        Depth layer to analyze

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    # Extract data for specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    directions = df[f"vap_sea_water_to_direction_layer_{layer}"].values
    u = df[f"u_layer_{layer}"].values
    v = df[f"v_layer_{layer}"].values

    # Create principal flow axis by finding the dominant direction
    # Convert to u,v components
    u_dir = -speeds * np.sin(
        np.radians(directions)
    )  # negative because oceanographic convention
    v_dir = -speeds * np.cos(np.radians(directions))

    # Find principal direction using PCA

    pca = PCA(n_components=2)
    pca.fit(np.column_stack([u_dir, v_dir]))
    principal_axis = (
        np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])) % 360
    )

    # Project speeds onto principal axis
    principal_comp = u_dir * np.cos(np.radians(principal_axis)) + v_dir * np.sin(
        np.radians(principal_axis)
    )

    # Separate flood and ebb (positive and negative components)
    flood_speeds = principal_comp[principal_comp > 0]
    ebb_speeds = -principal_comp[principal_comp < 0]  # Make positive for comparison

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Histogram of flood vs ebb speeds
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(np.max(flood_speeds), np.max(ebb_speeds)) * 1.1, 30)
    ax1.hist(flood_speeds, bins=bins, alpha=0.7, label="Flood", color="blue")
    ax1.hist(ebb_speeds, bins=bins, alpha=0.7, label="Ebb", color="red")
    ax1.set_xlabel("Current Speed (m/s)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Flood vs Ebb Speed Distribution")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 2. Cumulative distribution of speeds
    ax2 = axes[0, 1]
    flood_sorted = np.sort(flood_speeds)
    ebb_sorted = np.sort(ebb_speeds)
    flood_cdf = np.arange(1, len(flood_speeds) + 1) / len(flood_speeds)
    ebb_cdf = np.arange(1, len(ebb_speeds) + 1) / len(ebb_speeds)

    ax2.plot(flood_sorted, flood_cdf, "-", linewidth=2, label="Flood", color="blue")
    ax2.plot(ebb_sorted, ebb_cdf, "-", linewidth=2, label="Ebb", color="red")
    ax2.set_xlabel("Current Speed (m/s)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Flood vs Ebb Speed CDF")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # 3. Polar plot of current direction
    ax3 = axes[1, 0]
    ax3.remove()  # Remove to replace with polar axis
    ax3 = fig.add_subplot(223, projection="polar")

    # Convert directions to radians and plot
    dir_rad = np.radians(directions)
    ax3.scatter(dir_rad, speeds, s=5, alpha=0.3, c=speeds, cmap="viridis")

    # Add markers for flood and ebb directions
    flood_dir = principal_axis
    ebb_dir = (principal_axis + 180) % 360
    ax3.scatter(
        [np.radians(flood_dir)],
        [np.median(flood_speeds)],
        s=100,
        color="blue",
        marker="^",
        label="Flood Dir",
    )
    ax3.scatter(
        [np.radians(ebb_dir)],
        [np.median(ebb_speeds)],
        s=100,
        color="red",
        marker="v",
        label="Ebb Dir",
    )

    # Format polar axis
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)  # Clockwise
    ax3.set_title("Current Direction Distribution")
    ax3.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

    # 4. Asymmetry metrics and stats
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate asymmetry metrics
    flood_mean = np.mean(flood_speeds)
    ebb_mean = np.mean(ebb_speeds)
    flood_max = np.max(flood_speeds)
    ebb_max = np.max(ebb_speeds)
    flood_p95 = np.percentile(flood_speeds, 95)
    ebb_p95 = np.percentile(ebb_speeds, 95)

    # Calculate ratio metrics
    speed_ratio = flood_mean / ebb_mean if ebb_mean > 0 else float("inf")
    max_ratio = flood_max / ebb_max if ebb_max > 0 else float("inf")
    p95_ratio = flood_p95 / ebb_p95 if ebb_p95 > 0 else float("inf")

    # Calculate power density (proportional to speed^3)
    flood_power = np.mean(flood_speeds**3)
    ebb_power = np.mean(ebb_speeds**3)
    power_ratio = flood_power / ebb_power if ebb_power > 0 else float("inf")

    # Flood duration vs ebb duration
    flood_duration = len(flood_speeds) / len(speeds) * 100
    ebb_duration = len(ebb_speeds) / len(speeds) * 100

    # Create text for stats
    stats_text = (
        "TIDAL ASYMMETRY METRICS\n"
        "======================\n\n"
        f"Principal Flow Axis: {principal_axis:.1f}° / {(principal_axis + 180) % 360:.1f}°\n\n"
        f"SPEEDS:\n"
        f"  Flood Mean: {flood_mean:.2f} m/s\n"
        f"  Ebb Mean: {ebb_mean:.2f} m/s\n"
        f"  Ratio (Flood/Ebb): {speed_ratio:.2f}\n\n"
        f"  Flood Max: {flood_max:.2f} m/s\n"
        f"  Ebb Max: {ebb_max:.2f} m/s\n"
        f"  Max Ratio: {max_ratio:.2f}\n\n"
        f"POWER:\n"
        f"  Flood Power Density: {flood_power:.2f} (m/s)³\n"
        f"  Ebb Power Density: {ebb_power:.2f} (m/s)³\n"
        f"  Power Ratio: {power_ratio:.2f}\n\n"
        f"DURATION:\n"
        f"  Flood: {flood_duration:.1f}%\n"
        f"  Ebb: {ebb_duration:.1f}%\n"
    )

    ax4.text(0, 1.0, stats_text, va="top", fontfamily="monospace")

    plt.tight_layout()
    depth_value = df[f"vap_sigma_depth_layer_{layer}"].mean()
    plt.suptitle(
        f"Tidal Asymmetry Analysis at {depth_value:.1f}m Depth", fontsize=16, y=1.02
    )

    return fig


def plot_tidal_energy_yield(
    df,
    layer=4,
    turbine_diameter=10.0,
    # RM1 Specs: https://www.mdpi.com/1996-1073/13/19/5145
    # turbine_diameter=20,
    rated_speed=2.0,
    efficiency=0.45,
    # Cut in and cut out speed not explicitly documented anywhere
    # Typical for cut in speed to be ~30% of rated speed
    cut_in_speed=0.5,
    cut_out_speed=5.0,
    check_for_collisions=True,
):
    """
    Estimate and visualize potential energy yield from a tidal turbine with collision detection.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int
        Depth layer for turbine placement
    turbine_diameter : float
        Diameter of the turbine in meters
    cut_in_speed : float
        Minimum speed at which turbine starts generating
    rated_speed : float
        Speed at which turbine reaches rated power
    cut_out_speed : float
        Maximum speed at which turbine operates
    efficiency : float
        Turbine efficiency (Cp value)
    check_for_collisions: bool
        Whether to check for potential collisions with water surface or seafloor
    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """

    # Get turbine depth from the data
    turbine_depth = df[f"vap_sigma_depth_layer_{layer}"].mean()

    # Collision detection
    collision_detected = False
    collision_messages = []

    if check_for_collisions is True:
        depth_min = 0
        depth_max = df["vap_sea_floor_depth"].mean()  # Average seafloor depth
        turbine_radius = turbine_diameter / 2
        blade_top = turbine_depth - turbine_radius
        blade_bottom = turbine_depth + turbine_radius

        if depth_min is not None and blade_top < depth_min:
            collision_detected = True
            collision_messages.append(
                f"COLLISION: Turbine blades extend {depth_min - blade_top:.1f}m above water surface!"
            )

        if depth_max is not None and blade_bottom > depth_max:
            collision_detected = True
            collision_messages.append(
                f"COLLISION: Turbine blades extend {blade_bottom - depth_max:.1f}m below seafloor!"
            )

    # Extract speeds for the specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    timestamps = df.index

    # Water density (kg/m³)
    rho = 1025

    # Calculate turbine swept area (m²)
    area = np.pi * (turbine_diameter / 2) ** 2

    # Create power curve function
    def turbine_power(speed):
        # If collision detected, return 0 power (turbine can't operate)
        if collision_detected:
            return 0

        if speed < cut_in_speed or speed > cut_out_speed:
            return 0
        elif speed < rated_speed:
            # Linear ramp-up between cut-in and rated speed
            power_coef = (
                efficiency
                * (speed**3 - cut_in_speed**3)
                / (rated_speed**3 - cut_in_speed**3)
            )
            return 0.5 * rho * area * power_coef * speed**3
        else:
            # Constant rated power
            return 0.5 * rho * area * efficiency * rated_speed**3

    # Calculate power for all speeds
    power_output = np.array([turbine_power(s) for s in speeds])

    # Convert to kW
    power_output_kw = power_output / 1000

    # Calculate energy yield
    # Assuming timestamps are evenly spaced, get time delta in hours
    if len(timestamps) > 1:
        time_delta_seconds = (timestamps[1] - timestamps[0]).total_seconds()
        time_delta_hours = time_delta_seconds / 3600
    else:
        time_delta_hours = 1  # Default if only one timestamp

    energy_yield_kwh = power_output_kw * time_delta_hours
    total_energy_kwh = np.sum(energy_yield_kwh)
    total_energy_mwh = total_energy_kwh / 1000

    # Calculate capacity factor
    # Max theoretical energy if running at rated power all the time
    max_power_kw = 0.5 * rho * area * efficiency * rated_speed**3 / 1000
    max_energy_kwh = max_power_kw * time_delta_hours * len(speeds)
    capacity_factor = total_energy_kwh / max_energy_kwh if max_energy_kwh > 0 else 0

    # Create figure with collision warning if needed
    fig_height = 10 if collision_detected else 8
    fig = plt.figure(figsize=(18, fig_height))

    # Adjust grid layout for collision warning
    if collision_detected:
        gs = fig.add_gridspec(
            3, 2, width_ratios=[7, 1], height_ratios=[1, 2, 2], hspace=0.3, wspace=0.1
        )
        # Add collision warning at the top
        ax_warning = fig.add_subplot(gs[0, :])
        ax_warning.axis("off")

        warning_text = "TURBINE COLLISION DETECTED!\n" + "\n".join(collision_messages)
        ax_warning.text(
            0.5,
            0.5,
            warning_text,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            transform=ax_warning.transAxes,
        )

        ax1 = fig.add_subplot(gs[1, 0])  # Power curve
        ax2 = fig.add_subplot(gs[2, 0])  # Time series
        ax3 = fig.add_subplot(gs[1, 1])  # Turbine specs
        ax4 = fig.add_subplot(gs[2, 1])  # Key results
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[7, 1], hspace=0.3, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, 0])  # Top left - Power curve
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left - Time series
        ax3 = fig.add_subplot(gs[0, 1])  # Top right - Turbine specs
        ax4 = fig.add_subplot(gs[1, 1])  # Bottom right - Key results

    # 1. Power curve
    speed_range = np.linspace(0, max(cut_out_speed * 1.2, np.max(speeds) * 1.1), 100)
    power_curve = (
        np.array([turbine_power(s) for s in speed_range]) / 1000
    )  # Convert to kW

    curve_color = "red" if collision_detected else sns.color_palette()[0]
    curve_label = "Power Curve (COLLISION!)" if collision_detected else "Power Curve"

    ax1.plot(
        speed_range, power_curve, color=curve_color, linewidth=2, label=curve_label
    )
    ax1.set_xlabel("Sea Water Speed [m/s]")
    ax1.set_ylabel("Estimated Turbine Power [kW]")
    ax1.set_title("Turbine Power Curve")

    # Add vertical lines for cut-in, rated and cut-out speeds
    ax1.axvline(
        x=cut_in_speed, color="g", linestyle="--", label=f"Cut-in: {cut_in_speed} m/s"
    )
    ax1.axvline(
        x=rated_speed, color="r", linestyle="--", label=f"Rated: {rated_speed} m/s"
    )
    ax1.axvline(
        x=cut_out_speed,
        color="k",
        linestyle="--",
        label=f"Cut-out: {cut_out_speed} m/s",
    )

    # Add histogram of actual speeds as background
    ax1_twin = ax1.twinx()
    ax1_twin.hist(speeds, bins=30, alpha=0.3, color="gray")
    ax1_twin.set_ylabel("Frequency")
    ax1_twin.set_zorder(0)
    ax1_twin.set_axisbelow(True)
    ax1_twin.grid(False)

    ax1.grid(True, linestyle="--", alpha=0.7)
    legend = ax1.legend(loc="upper right", facecolor="white")

    # 2. Time series of power output
    power_color = "red" if collision_detected else sns.color_palette()[0]

    ax2.plot(
        timestamps,
        power_output_kw,
        color=power_color,
        linewidth=0.5,
        label="Estimated Turbine Power [kW]",
    )
    if not collision_detected:
        ax2.plot(
            timestamps,
            pd.Series(power_output_kw).rolling(window=48).mean(),
            color=sns.color_palette()[1],
            linewidth=1.5,
            alpha=0.9,
            label="Estimated Turbine Power - Daily Average [kW]",
        )
    ax2.set_xlabel("Time [UTC]")
    ax2.set_ylabel("Estimated Turbine Power [kW]")
    ax2.set_title("Estimated Power Generation Over Time")

    # Add horizontal line for mean power
    mean_power_kw = np.mean(power_output_kw)
    max_power_kw = np.max(power_output_kw)
    min_power_kw = np.min(power_output_kw)
    ax2.axhline(
        y=mean_power_kw, color="k", linestyle="-", label=f"Mean: {mean_power_kw:.1f} kW"
    )

    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right", facecolor="white")

    # Calculate additional metrics
    operating_hours = np.sum(speeds >= cut_in_speed) * time_delta_hours
    operating_percentage = operating_hours / (len(speeds) * time_delta_hours) * 100

    hours_at_rated = np.sum(speeds >= rated_speed) * time_delta_hours
    rated_percentage = hours_at_rated / (len(speeds) * time_delta_hours) * 100

    # Time period covered
    if len(timestamps) > 1:
        days_covered = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    else:
        days_covered = 1

    # Extrapolate to annual yield if data is less than a year
    if days_covered < 365:
        annual_factor = 365 / days_covered
        annual_yield_mwh = total_energy_mwh * annual_factor
    else:
        annual_yield_mwh = total_energy_mwh

    # 3. Turbine specifications (top right) - enhanced with collision info
    ax3.axis("off")

    # Build collision status text
    collision_status = ""
    if depth_min is not None or depth_max is not None:
        turbine_radius = turbine_diameter / 2
        blade_top = turbine_depth - turbine_radius
        blade_bottom = turbine_depth + turbine_radius

        surface_clearance = blade_top - depth_min if depth_min is not None else "N/A"
        seafloor_clearance = (
            depth_max - blade_bottom if depth_max is not None else "N/A"
        )

        collision_status = (
            f"COLLISION CHECK\n"
            # f"  • Turbine Depth: {turbine_depth:.1f} m\n"
            # f"  • Blade Top: {blade_top:.1f} m\n"
            # f"  • Blade Bottom: {blade_bottom:.1f} m\n"
            f"  • Surface Clearance: {surface_clearance:.1f} m\n"
            f"  • Sea Floor Clearance: {seafloor_clearance:.1f} m\n"
            if depth_min is not None
            else f"  • Seafloor Clearance: {seafloor_clearance:.1f} m\n"
            if depth_max is not None
            else f"  • Status: {'COLLISION!' if collision_detected else 'OK'}\n"
        )

    turbine_specs_text = (
        f"TURBINE SETUP\n"
        f"  • Diameter: {turbine_diameter} m\n"
        f"  • Swept Area: {area:.1f} m²\n"
        f"  • Efficiency: {efficiency:.1%}\n"
        f"OPERATING SPEEDS\n"
        f"  • Cut-in: {cut_in_speed} m/s\n"
        f"  • Rated: {rated_speed} m/s\n"
        f"  • Cut-out: {cut_out_speed} m/s\n"
        f"CURRENT CONDITIONS\n"
        f"  • Mean Speed: {np.mean(speeds):.2f} m/s\n"
        f"  • Max Speed: {np.max(speeds):.2f} m/s\n"
        f"  • Data Period: {days_covered:.0f} days\n"
        f"{collision_status}"
        f"POWER EQUATION\n"
        f"P = ½ρAC_p v³\n"
        f"({rho} kg/m³ × {area:.0f} m² × {efficiency} × v³)"
    )

    text_color = "red" if collision_detected else "black"
    ax3.text(
        0.05,
        0.95,
        turbine_specs_text,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="normal",
        color=text_color,
        transform=ax3.transAxes,
        linespacing=1.4,
    )

    # 4. Key results summary (bottom right)
    ax4.axis("off")

    status_text = (
        "INOPERABLE DUE TO COLLISION"
        if collision_detected
        else f"{annual_yield_mwh:.2f} MWh/year"
    )

    key_results_text = (
        f"ANNUAL ENERGY YIELD\n"
        f"  • {status_text}\n"
        f"CAPACITY FACTOR\n"
        f"  • {capacity_factor * 100:.2f}%\n"
        # f"  • Efficiency vs. theoretical max\n"
        f"POWER OUTPUT\n"
        f"  • MEAN: {mean_power_kw:.2f} kW\n"
        f"  • MAX: {max_power_kw:.2f} kW\n"
        f"  • MIN: {min_power_kw:.2f} kW\n"
        f"OPERATIONAL TIME\n"
        f"  • {operating_hours:.0f} of {len(speeds) * time_delta_hours:.0f} hours\n"
        f"  • {operating_percentage:.2f}% of time generating\n"
        f"ENERGY PRODUCTION PERIOD\n"
        f"  • {days_covered:.0f} days of data\n"
        f"  • {hours_at_rated:.2f} hours at rated power"
    )

    ax4.text(
        0.05,
        0.95,
        key_results_text,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="normal",
        color=text_color,
        transform=ax4.transAxes,
        linespacing=1.4,
    )

    plt.subplots_adjust(
        hspace=0.0, top=0.85
    )  # hspace for vertical spacing between subplots, top to adjust suptitle position

    plt.tight_layout()

    return fig


def extra_plot_tidal_energy_yield(
    df,
    layer=0,
    turbine_diameter=10.0,
    cut_in_speed=0.5,
    rated_speed=2.0,
    cut_out_speed=4.0,
    efficiency=0.35,
    depth_min=None,
):
    """
    Estimate and visualize potential energy yield from a tidal turbine.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data
    layer : int
        Depth layer for turbine placement
    turbine_diameter : float
        Diameter of the turbine in meters
    cut_in_speed : float
        Minimum speed at which turbine starts generating
    rated_speed : float
        Speed at which turbine reaches rated power
    cut_out_speed : float
        Maximum speed at which turbine operates
    efficiency : float
        Turbine efficiency (Cp value)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    # Extract speeds for the specified layer
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    timestamps = df.index

    # Water density (kg/m³)
    rho = 1025

    # Calculate turbine swept area (m²)
    area = np.pi * (turbine_diameter / 2) ** 2

    # Create power curve function
    def turbine_power(speed):
        if speed < cut_in_speed or speed > cut_out_speed:
            return 0
        elif speed < rated_speed:
            # Linear ramp-up between cut-in and rated speed
            power_coef = (
                efficiency
                * (speed**3 - cut_in_speed**3)
                / (rated_speed**3 - cut_in_speed**3)
            )
            return 0.5 * rho * area * power_coef * speed**3
        else:
            # Constant rated power
            return 0.5 * rho * area * efficiency * rated_speed**3

    # Calculate power for all speeds
    power_output = np.array([turbine_power(s) for s in speeds])

    # Convert to kW
    power_output_kw = power_output / 1000

    # Calculate energy yield
    # Assuming timestamps are evenly spaced, get time delta in hours
    if len(timestamps) > 1:
        time_delta_seconds = (timestamps[1] - timestamps[0]).total_seconds()
        time_delta_hours = time_delta_seconds / 3600
    else:
        time_delta_hours = 1  # Default if only one timestamp

    energy_yield_kwh = power_output_kw * time_delta_hours
    total_energy_kwh = np.sum(energy_yield_kwh)
    total_energy_mwh = total_energy_kwh / 1000

    # Calculate capacity factor
    # Max theoretical energy if running at rated power all the time
    max_power_kw = 0.5 * rho * area * efficiency * rated_speed**3 / 1000
    max_energy_kwh = max_power_kw * time_delta_hours * len(speeds)
    capacity_factor = total_energy_kwh / max_energy_kwh if max_energy_kwh > 0 else 0

    # Create figure with 2x2 grid layout - wider text panels for better readability
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[7, 1], hspace=0.3, wspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])  # Top left - Power curve
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left - Time series
    ax3 = fig.add_subplot(gs[0, 1])  # Top right - Turbine specs
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right - Key results

    # 1. Power curve
    speed_range = np.linspace(0, max(cut_out_speed * 1.2, np.max(speeds) * 1.1), 100)
    power_curve = (
        np.array([turbine_power(s) for s in speed_range]) / 1000
    )  # Convert to kW

    ax1.plot(speed_range, power_curve, "b-", linewidth=2, label="Power Curve")
    ax1.set_xlabel("Sea Water Speed [m/s]")
    ax1.set_ylabel("Estimated Turbine Power [kW]")
    ax1.set_title("Turbine Power Curve")

    # Add vertical lines for cut-in, rated and cut-out speeds
    ax1.axvline(
        x=cut_in_speed, color="g", linestyle="--", label=f"Cut-in: {cut_in_speed} m/s"
    )
    ax1.axvline(
        x=rated_speed, color="r", linestyle="--", label=f"Rated: {rated_speed} m/s"
    )
    ax1.axvline(
        x=cut_out_speed,
        color="k",
        linestyle="--",
        label=f"Cut-out: {cut_out_speed} m/s",
    )

    # Add histogram of actual speeds as background
    ax1_twin = ax1.twinx()
    ax1_twin.hist(speeds, bins=30, alpha=0.3, color="gray")
    ax1_twin.set_ylabel("Frequency")
    ax1_twin.set_zorder(0)
    ax1_twin.set_axisbelow(True)
    ax1_twin.grid(False)

    ax1.grid(True, linestyle="--", alpha=0.7)
    legend = ax1.legend(loc="upper right", facecolor="white")

    # 2. Time series of power output
    ax2.plot(
        timestamps,
        power_output_kw,
        # "r-",
        linewidth=0.5,
        # alpha=0.8,
        label="Estimated Turbine Power [kW]",
    )
    ax2.plot(
        timestamps,
        pd.Series(power_output_kw).rolling(window=48).mean(),
        # "r-",
        linewidth=0.5,
        # alpha=0.8,
        label="Turbine Power - Daily Average [kW]",
    )
    ax2.set_xlabel("Time [UTC]")
    ax2.set_ylabel("Turbine Output Power [kW]")
    ax2.set_title("Estimated Power Generation Over Time")

    # Add horizontal line for mean power
    mean_power_kw = np.mean(power_output_kw)
    ax2.axhline(
        y=mean_power_kw, color="k", linestyle="-", label=f"Mean: {mean_power_kw:.1f} kW"
    )

    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right", facecolor="white")

    # Calculate additional metrics
    operating_hours = np.sum(speeds >= cut_in_speed) * time_delta_hours
    operating_percentage = operating_hours / (len(speeds) * time_delta_hours) * 100

    hours_at_rated = np.sum(speeds >= rated_speed) * time_delta_hours
    rated_percentage = hours_at_rated / (len(speeds) * time_delta_hours) * 100

    # Time period covered
    if len(timestamps) > 1:
        days_covered = (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600)
    else:
        days_covered = 1

    # Extrapolate to annual yield if data is less than a year
    if days_covered < 365:
        annual_factor = 365 / days_covered
        annual_yield_mwh = total_energy_mwh * annual_factor
    else:
        annual_yield_mwh = total_energy_mwh

    # 3. Turbine specifications (top right) - simplified and cleaner
    ax3.axis("off")

    turbine_specs_text = (
        f"TURBINE SETUP\n"
        f"  • Diameter: {turbine_diameter} m\n"
        f"  • Swept Area: {area:.1f} m²\n"
        f"  • Efficiency: {efficiency:.1%}\n"
        f"OPERATING SPEEDS\n"
        f"  • Cut-in: {cut_in_speed} m/s\n"
        f"  • Rated: {rated_speed} m/s\n"
        f"  • Cut-out: {cut_out_speed} m/s\n"
        f"CURRENT CONDITIONS\n"
        f"  • Mean Speed: {np.mean(speeds):.2f} m/s\n"
        f"  • Max Speed: {np.max(speeds):.2f} m/s\n"
        f"  • Data Period: {days_covered:.0f} days\n"
        f"POWER EQUATION\n"
        f"P = ½ρAC_p v³\n"
        f"({rho} kg/m³ × {area:.0f} m² × {efficiency} × v³)"
    )

    ax3.text(
        0.05,
        0.95,
        turbine_specs_text,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="normal",
        transform=ax3.transAxes,
        linespacing=1.4,
    )

    # 4. Key results summary (bottom right) - consistent with top panel
    ax4.axis("off")

    key_results_text = (
        f"ANNUAL ENERGY YIELD\n"
        f"  • {annual_yield_mwh:.2f} MWh/year\n"
        f"CAPACITY FACTOR\n"
        f"  • {capacity_factor * 100:.2f}%\n"
        f"  • Efficiency vs. theoretical max\n"
        f"MEAN POWER OUTPUT\n"
        f"  • {mean_power_kw:.2f} kW\n"
        f"OPERATIONAL TIME\n"
        f"  • {operating_hours:.0f} of {len(speeds) * time_delta_hours:.0f} hours\n"
        f"  • {operating_percentage:.2f}% of time generating\n"
        f"ENERGY PRODUCTION PERIOD\n"
        f"  • {days_covered:.0f} days of data\n"
        f"  • {hours_at_rated:.2f} hours at rated power"
    )

    ax4.text(
        0.05,
        0.95,
        key_results_text,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="normal",
        transform=ax4.transAxes,
        linespacing=1.4,
    )

    legend.set_zorder(1000)  # Ensure legend is above other elements

    plt.tight_layout()

    plt.subplots_adjust(
        hspace=0.0, top=0.85
    )  # hspace for vertical spacing between subplots, top to adjust suptitle position

    # depth_value = df[f"vap_sigma_depth_layer_{layer}"].mean()
    # plt.suptitle(
    #     f"Tidal Energy Yield Estimation at {depth_value:.1f}m Depth",
    #     fontsize=16,
    #     y=1.02,
    # )

    return fig


def plot_velocity_shear_profile(df):
    """
    Analyze velocity shear and turbulence across the water column.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    # Calculate average depths for each layer
    depths = []
    for i in range(10):
        layer_depths = df[f"vap_sigma_depth_layer_{i}"].values
        depths.append(np.mean(layer_depths))

    # Calculate velocity differences between adjacent layers
    velocity_diffs = []
    depth_diffs = []

    for i in range(9):  # 10 layers means 9 differences
        v1 = df[f"vap_sea_water_speed_layer_{i}"].values
        v2 = df[f"vap_sea_water_speed_layer_{i + 1}"].values
        velocity_diff = v1 - v2

        d1 = df[f"vap_sigma_depth_layer_{i}"].values
        d2 = df[f"vap_sigma_depth_layer_{i + 1}"].values
        depth_diff = d2 - d1  # Should be positive as depth increases

        velocity_diffs.append(velocity_diff)
        depth_diffs.append(depth_diff)

    # Calculate shear (du/dz) for each layer interface
    shear = []
    mean_interface_depths = []

    for i in range(9):
        # Average depth difference between layers
        mean_depth_diff = np.mean(depth_diffs[i])
        # Calculate shear as velocity gradient
        layer_shear = velocity_diffs[i] / depth_diffs[i]
        shear.append(layer_shear)
        # Calculate mean depth at each interface
        mean_interface_depths.append((depths[i] + depths[i + 1]) / 2)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))

    # 1. Vertical profile of mean velocity
    mean_velocities = [df[f"vap_sea_water_speed_layer_{i}"].mean() for i in range(10)]
    ax1.plot(mean_velocities, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Mean Current Speed (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Vertical Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()

    # 2. Mean shear profile
    mean_shear = [np.mean(s) for s in shear]
    std_shear = [np.std(s) for s in shear]

    ax2.plot(
        mean_shear,
        mean_interface_depths,
        "o-",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    # Add error bars for shear variability
    ax2.errorbar(
        mean_shear,
        mean_interface_depths,
        xerr=std_shear,
        fmt="none",
        ecolor="orange",
        alpha=0.5,
        capsize=5,
    )

    ax2.set_xlabel("Velocity Shear (1/s)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Vertical Shear Profile")
    ax2.grid(True)
    ax2.invert_yaxis()

    # 3. Shear statistics as box plots
    boxplot_positions = mean_interface_depths
    boxplot_data = shear

    # Create boxplots
    bp = ax3.boxplot(
        boxplot_data,
        positions=boxplot_positions,
        vert=False,
        patch_artist=True,
        widths=0.5,
    )

    # Customize boxplot colors
    for box in bp["boxes"]:
        box.set(facecolor="lightblue", alpha=0.8)
    for median in bp["medians"]:
        median.set(color="darkred", linewidth=2)

    ax3.set_xlabel("Velocity Shear (1/s)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Shear Variability")
    ax3.grid(True, axis="x")
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.suptitle("Vertical Velocity Shear Analysis", fontsize=16, y=1.02)

    # Add a calculated metric for overall shear intensity
    total_depth = depths[-1] - depths[0]
    surface_speed = df["vap_sea_water_speed_layer_0"].mean()
    bottom_speed = df["vap_sea_water_speed_layer_9"].mean()
    overall_shear = (surface_speed - bottom_speed) / total_depth

    fig.text(
        0.5,
        0.01,
        f"Overall Water Column Shear: {overall_shear:.5f} (1/s)",
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    return fig


def plot_tidal_phase_analysis(df, layer=4):
    """
    Analyze and visualize tidal phases and their associated current velocities.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data with DatetimeIndex
    layer : int
        Depth layer to analyze

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    # Extract data
    speeds = df[f"vap_sea_water_speed_layer_{layer}"].values
    timestamps = df.index
    water_level = df["vap_surface_elevation"].values  # Water elevation

    # Create a time series of hours since start of dataset
    hours_elapsed = np.zeros(len(timestamps))
    for i in range(1, len(timestamps)):
        hours_elapsed[i] = (
            hours_elapsed[i - 1]
            + (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        )

    # Identify local maxima and minima in water level (high and low tides)

    high_tide_idx, _ = find_peaks(
        water_level, distance=5
    )  # Adjust distance based on data resolution
    low_tide_idx, _ = find_peaks(-water_level, distance=5)

    # Calculate tidal range
    tidal_range = []
    for i in range(len(high_tide_idx) - 1):
        # Find closest low tide between consecutive high tides
        low_between = [
            j for j in low_tide_idx if high_tide_idx[i] < j < high_tide_idx[i + 1]
        ]
        if low_between:
            low_idx = low_between[0]
            tide_range = water_level[high_tide_idx[i]] - water_level[low_idx]
            tidal_range.append(tide_range)

    mean_tidal_range = np.mean(tidal_range) if tidal_range else 0

    # Create figures for tidal phase analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    # 1. Water level time series with high/low tide markers
    ax1.plot(timestamps, water_level, "b-", linewidth=1.5)
    ax1.plot(
        timestamps[high_tide_idx], water_level[high_tide_idx], "ro", label="High Tide"
    )
    ax1.plot(
        timestamps[low_tide_idx], water_level[low_tide_idx], "go", label="Low Tide"
    )

    ax1.set_ylabel("Water Level (m)")
    ax1.set_title("Tidal Elevation")
    ax1.legend()
    ax1.grid(True)

    # 2. Current speed time series
    ax2.plot(timestamps, speeds, "r-", linewidth=1.5)

    # Add vertical lines for high and low tides
    for idx in high_tide_idx:
        ax2.axvline(x=timestamps[idx], color="r", linestyle="--", alpha=0.3)
    for idx in low_tide_idx:
        ax2.axvline(x=timestamps[idx], color="g", linestyle="--", alpha=0.3)

    ax2.set_ylabel("Current Speed (m/s)")
    ax2.set_title("Tidal Current Speed")
    ax2.grid(True)

    # 3. Phase relationship (speed vs. water level)
    ax3.scatter(water_level, speeds, s=10, alpha=0.5)
    ax3.set_xlabel("Water Level (m)")
    ax3.set_ylabel("Current Speed (m/s)")
    ax3.set_title("Current Speed vs. Water Level (Phase Relationship)")
    ax3.grid(True)

    # Add best fit curve

    # Try a quadratic fit (typical for tidal currents vs. water level)
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    try:
        popt, _ = curve_fit(quadratic, water_level, speeds)
        a, b, c = popt

        x_fit = np.linspace(min(water_level), max(water_level), 100)
        y_fit = quadratic(x_fit, a, b, c)

        ax3.plot(
            x_fit,
            y_fit,
            "r-",
            linewidth=2,
            label=f"Fit: {a:.4f}x² + {b:.4f}x + {c:.4f}",
        )
        ax3.legend()
    except:
        pass  # If fitting fails, just skip it

    # Add another subplot for phase lag analysis
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax3.set_position(gs[2, 0].get_position(fig))
    ax4 = fig.add_subplot(gs[2, 1])

    # Calculate phase lag between max water level and max current
    phase_lags = []

    for i in range(len(high_tide_idx)):
        high_tide_time = timestamps[high_tide_idx[i]]

        # Find max current within +/- 3 hours of high tide
        window_size = 3 * 3600  # 3 hours in seconds

        # Create window boundaries
        start_window = high_tide_time - pd.Timedelta(seconds=window_size)
        end_window = high_tide_time + pd.Timedelta(seconds=window_size)

        # Find indices within window
        window_mask = (timestamps >= start_window) & (timestamps <= end_window)
        window_speeds = speeds[window_mask]
        window_times = timestamps[window_mask]

        if len(window_speeds) > 0:
            max_speed_idx = np.argmax(window_speeds)
            max_speed_time = window_times[max_speed_idx]

            # Calculate lag in hours
            lag_hours = (max_speed_time - high_tide_time).total_seconds() / 3600
            phase_lags.append(lag_hours)

    # Plot histogram of phase lags
    if phase_lags:
        ax4.hist(phase_lags, bins=15, alpha=0.7)
        ax4.set_xlabel("Phase Lag (hours)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Phase Lag Distribution")
        ax4.grid(True)

        # Add mean lag line
        mean_lag = np.mean(phase_lags)
        ax4.axvline(
            x=mean_lag, color="r", linestyle="-", label=f"Mean: {mean_lag:.2f} hrs"
        )
        ax4.legend()

    plt.tight_layout()
    depth_value = df[f"vap_sigma_depth_layer_{layer}"].mean()
    plt.suptitle(
        f"Tidal Phase Analysis at {depth_value:.1f}m Depth", fontsize=16, y=1.02
    )

    # Add summary statistics
    if phase_lags:
        fig.text(
            0.5,
            0.01,
            f"Mean Tidal Range: {mean_tidal_range:.2f} m | "
            f"Mean Phase Lag: {mean_lag:.2f} hours | "
            f"Type: {'Flood Dominant' if mean_lag < 0 else 'Ebb Dominant'}",
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    return fig


def generate_tidal_site_assessment(df, site_name="Tidal Site"):
    """
    Generate a comprehensive tidal site assessment report with multiple visualizations.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the tidal data with DatetimeIndex
    site_name : str
        Name of the tidal site

    Returns:
    --------
    figs : list of matplotlib Figure
        List of created figures
    """

    # Create a list to store all figures
    figs = []

    # 1. Create a site overview dashboard
    fig1 = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig1)

    # Extract key site metrics
    mean_depth = df["vap_sea_floor_depth"].mean()
    max_speed = df["vap_water_column_max_sea_water_speed"].max()
    mean_speed = df["vap_water_column_mean_sea_water_speed"].mean()
    p95_speed = df["vap_water_column_95th_percentile_sea_water_speed"].mean()
    max_power = df["vap_water_column_max_sea_water_power_density"].max()
    mean_power = df["vap_water_column_mean_sea_water_power_density"].mean()

    # Calculate velocity exceedance statistics
    speeds = df["vap_water_column_mean_sea_water_speed"].values
    speeds_sorted = np.sort(speeds)[::-1]
    exceedance = np.arange(1, len(speeds) + 1) / len(speeds) * 100

    # Find exceedance percentages for key thresholds
    exc_1ms = np.sum(speeds >= 1.0) / len(speeds) * 100
    exc_1_5ms = np.sum(speeds >= 1.5) / len(speeds) * 100
    exc_2ms = np.sum(speeds >= 2.0) / len(speeds) * 100

    # Create summary text box
    ax_summary = fig1.add_subplot(gs[0, 0])
    ax_summary.axis("off")
    summary_text = (
        f"SITE ASSESSMENT SUMMARY: {site_name}\n"
        "====================================\n\n"
        f"LOCATION\n"
        f"  Latitude: {df['lat_center'].iloc[0]:.4f}°\n"
        f"  Longitude: {df['lon_center'].iloc[0]:.4f}°\n"
        f"  Mean Water Depth: {mean_depth:.1f} m\n\n"
        f"CURRENT VELOCITY\n"
        f"  Mean Speed: {mean_speed:.2f} m/s\n"
        f"  Maximum Speed: {max_speed:.2f} m/s\n"
        f"  95th Percentile: {p95_speed:.2f} m/s\n\n"
        f"POWER RESOURCE\n"
        f"  Mean Power Density: {mean_power:.2f} W/m²\n"
        f"  Maximum Power Density: {max_power:.2f} W/m²\n\n"
        f"EXCEEDANCE METRICS\n"
        f"  > 1.0 m/s: {exc_1ms:.1f}% of time\n"
        f"  > 1.5 m/s: {exc_1_5ms:.1f}% of time\n"
        f"  > 2.0 m/s: {exc_2ms:.1f}% of time\n"
    )
    ax_summary.text(0, 1.0, summary_text, va="top", fontfamily="monospace", fontsize=12)

    # 2. Velocity Exceedance Curve
    ax_exc = fig1.add_subplot(gs[0, 1:])
    ax_exc.plot(exceedance, speeds_sorted, "b-", linewidth=2)

    # Add reference lines
    ax_exc.axhline(y=1.0, color="g", linestyle="--", label="1.0 m/s")
    ax_exc.axhline(y=1.5, color="orange", linestyle="--", label="1.5 m/s")
    ax_exc.axhline(y=2.0, color="r", linestyle="--", label="2.0 m/s")

    ax_exc.set_xlabel("Exceedance Probability (%)")
    ax_exc.set_ylabel("Current Speed (m/s)")
    ax_exc.set_title("Current Speed Exceedance Curve")
    ax_exc.grid(True)
    ax_exc.legend()

    # 3. Time series of speed and power
    ax_ts = fig1.add_subplot(gs[1, :])

    # Limit to first 1000 points for clarity if dataset is large
    n_points = min(1000, len(df))
    timestamps = df.index[:n_points]

    # Plot speed time series on left y-axis
    ax_ts.plot(
        timestamps,
        df["vap_water_column_mean_sea_water_speed"][:n_points],
        "b-",
        linewidth=1,
        label="Mean Speed",
    )
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Current Speed (m/s)", color="b")
    ax_ts.tick_params(axis="y", labelcolor="b")

    # Plot power density on right y-axis
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(
        timestamps,
        df["vap_water_column_mean_sea_water_power_density"][:n_points],
        "r-",
        linewidth=1,
        label="Power Density",
    )
    ax_ts2.set_ylabel("Power Density (W/m²)", color="r")
    ax_ts2.tick_params(axis="y", labelcolor="r")

    # Add legend
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax_ts.set_title("Current Speed and Power Density Time Series")
    ax_ts.grid(True)

    # 4. Speed distribution histogram
    ax_hist = fig1.add_subplot(gs[2, 0])
    ax_hist.hist(speeds, bins=30, color="skyblue", edgecolor="black")

    # Add vertical lines for mean and percentiles
    ax_hist.axvline(
        x=mean_speed, color="r", linestyle="-", label=f"Mean: {mean_speed:.2f} m/s"
    )
    ax_hist.axvline(
        x=p95_speed, color="g", linestyle="--", label=f"95th: {p95_speed:.2f} m/s"
    )

    ax_hist.set_xlabel("Current Speed (m/s)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Current Speed Distribution")
    ax_hist.grid(True)
    ax_hist.legend()

    ax_rose = WindroseAxes(fig1, fig1.add_subplot(gs[2, 1]).get_position())
    fig1.add_axes(ax_rose)

    directions = df["vap_water_column_mean_sea_water_to_direction"].values
    speeds = df["vap_water_column_mean_sea_water_speed"].values

    ax_rose.bar(directions, speeds, normed=True, opening=0.8, edgecolor="white")
    ax_rose.set_legend(title="Speed (m/s)")
    ax_rose.set_title("Current Rose")

    # 6. Vertical profile and power curve
    ax_prof = fig1.add_subplot(gs[2, 2])

    # Calculate mean values for each depth layer
    mean_speeds = []
    mean_depths = []
    for i in range(10):
        mean_speeds.append(df[f"vap_sea_water_speed_layer_{i}"].mean())
        mean_depths.append(df[f"vap_sigma_depth_layer_{i}"].mean())

    # Plot vertical profile
    ax_prof.plot(mean_speeds, mean_depths, "o-", linewidth=2, markersize=8)
    ax_prof.set_xlabel("Mean Current Speed (m/s)")
    ax_prof.set_ylabel("Depth (m)")
    ax_prof.set_title("Vertical Velocity Profile")
    ax_prof.grid(True)
    ax_prof.invert_yaxis()

    # Add info about optimal depth
    optimal_layer = np.argmax(mean_speeds)
    ax_prof.annotate(
        f"Optimal Layer: {optimal_layer} ({mean_depths[optimal_layer]:.1f}m)",
        xy=(mean_speeds[optimal_layer], mean_depths[optimal_layer]),
        xytext=(mean_speeds[optimal_layer] - 0.2, mean_depths[optimal_layer]),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    # Add suptitle
    plt.suptitle(f"Tidal Energy Resource Assessment: {site_name}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    figs.append(fig1)

    # Create additional specialized plots if needed
    # These would be other functions we defined earlier

    return figs


def plot_fft(df, sample_rate=None):
    """
    Generate FFT plots for each depth layer of tidal current speed.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing tidal current data with columns:
        - vap_sea_water_speed_layer_{i} : Current speed at layer i
        - vap_sigma_depth_layer_{i} : Depth of layer i
    sample_rate : float, optional
        Sampling rate in hours. If None, it will be calculated from the DataFrame index.

    Returns:
    --------
    fig : matplotlib Figure
        Figure containing FFT plots for each depth layer
    """

    # Style parameters
    colors = sns.color_palette()

    # Get the number of layers (assuming 10 layers)
    n_layers = 10

    # Extract timestamps
    timestamps = df.index

    # Calculate time difference if sample_rate is not provided
    if sample_rate is None and len(timestamps) > 1:
        time_diff = (timestamps[1] - timestamps[0]).total_seconds() / 3600  # hours
        sample_rate = 1 / time_diff
    elif sample_rate is None:
        sample_rate = 1  # Default to 1 sample per hour

    # Extract all velocity data and depths
    n_locations = len(df)
    all_velocities = np.zeros((n_locations, n_layers))
    all_depths = np.zeros((n_locations, n_layers))

    # Fill the arrays with data from each location and layer
    for loc in range(n_locations):
        for layer in range(n_layers):
            all_velocities[loc, layer] = df.iloc[loc][
                f"vap_sea_water_speed_layer_{layer}"
            ]
            all_depths[loc, layer] = df.iloc[loc][f"vap_sigma_depth_layer_{layer}"]

    # Calculate min/max depths for each layer
    min_depths = np.min(all_depths, axis=0)
    max_depths = np.max(all_depths, axis=0)

    # Create figure for FFT plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    # Create FFT plot for each layer
    for i in range(n_layers):
        ax = axes[i]

        # Extract velocity time series for this layer
        velocity_data = all_velocities[:, i]

        # Remove mean (de-trend)
        velocity_data = velocity_data - np.mean(velocity_data)

        # Apply a Hanning window to reduce spectral leakage
        window = np.hanning(len(velocity_data))
        windowed_data = velocity_data * window

        # Perform FFT
        fft_result = np.fft.rfft(windowed_data)
        fft_freq = np.fft.rfftfreq(
            len(velocity_data), d=1 / sample_rate
        )  # Freq in cycles per hour

        # Convert to periods in hours for better readability
        # Skip the first element (DC component)
        periods = 1 / fft_freq[1:]
        amplitudes = np.abs(fft_result)[1:]

        # Plot the FFT spectrum using periods
        ax.plot(periods, amplitudes, "-", color=colors[0], linewidth=1.5)

        # Mark significant tidal periods
        significant_periods = [12.42, 24.0, 6.21, 12.0, 25.82]  # hours
        significant_names = ["M2", "K1", "M4", "S2", "O1"]  # tidal constituent names

        for period, name in zip(significant_periods, significant_names):
            # Find the closest index to this period
            if len(periods) > 0:  # Check if we have data points
                closest_idx = np.argmin(np.abs(periods - period))
                if closest_idx < len(amplitudes):
                    # Mark with a vertical line if in range
                    ax.axvline(
                        x=periods[closest_idx], color="red", alpha=0.3, linestyle="--"
                    )
                    # Add text annotation if amplitude is significant
                    if amplitudes[closest_idx] > 0.1 * np.max(amplitudes):
                        ax.text(
                            periods[closest_idx],
                            amplitudes[closest_idx] * 1.1,
                            name,
                            ha="center",
                            fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.8),
                        )

        # Add depth range label
        depth_label = f"Depth: {min_depths[i]:.1f} - {max_depths[i]:.1f} m"
        ax.text(
            0.02,
            0.95,
            depth_label,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Set log scale for x-axis to better visualize the tidal constituents
        ax.set_xscale("log")

        # Set x-limits to focus on tidal periods (4-30 hours)
        ax.set_xlim(4, 30)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add labels
        if i >= 5:  # Bottom row
            ax.set_xlabel("Period (hours)")
        if i % 5 == 0:  # First column
            ax.set_ylabel("Amplitude")

    # Add a title to the entire figure
    fig.suptitle("FFT Analysis of Current Speed by Depth Layer", fontsize=16)

    # Adjust spacing
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title

    return fig


def plot_speed_mesh(df):
    """
    Create a 2D heatmap of current speed over time throughout the water column.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing tidal current data with columns:
        - vap_sea_water_speed_layer_{i} : Current speed at layer i
        - vap_sigma_depth_layer_{i} : Depth of layer i

    Returns:
    --------
    fig : matplotlib Figure
        Figure containing the 2D heatmap of current speed
    """
    # Get the number of layers (assuming 10 layers)
    n_layers = 10

    # Extract timestamps
    timestamps = df.index
    n_locations = len(timestamps)

    # Extract all velocity data and depths
    all_velocities = np.zeros((n_locations, n_layers))
    all_depths = np.zeros((n_locations, n_layers))

    # Fill the arrays with data from each location and layer
    for loc in range(n_locations):
        for layer in range(n_layers):
            all_velocities[loc, layer] = df.iloc[loc][
                f"vap_sea_water_speed_layer_{layer}"
            ]
            all_depths[loc, layer] = df.iloc[loc][f"vap_sigma_depth_layer_{layer}"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colormap - switching to viridis
    cmap = plt.cm.viridis

    # Get min and max speed for color scaling
    vmin = np.min(all_velocities)
    vmax = np.max(all_velocities)

    # Convert timestamps to matplotlib dates for plotting
    time_num = mdates.date2num(timestamps)

    # Get the deepest point at each time (for bottom line)
    bottom_depths = np.max(all_depths, axis=1)

    # Simulate surface elevation change
    # (we'll create a simple sinusoidal variation with a 12.4 hour period to simulate tides)
    # Amplitude of 0.5 meters for the variation
    hours = np.array([(t - timestamps[0]).total_seconds() / 3600 for t in timestamps])
    tide_period = 12.4  # M2 tidal period in hours
    surface_elevation = -0.5 * np.sin(2 * np.pi * hours / tide_period)

    # Create the data visualization with imshow
    im = ax.imshow(
        all_velocities.T,  # Transpose to get depth on the y-axis
        aspect="auto",  # Adjust aspect ratio to fill the plot
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            time_num[0],
            time_num[-1],
            np.max(bottom_depths),
            np.min(surface_elevation),
        ],
        origin="upper",  # This makes the image start at the top (shallow)
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Current Speed (m/s)")

    # Convert the indices to actual dates
    ax.xaxis_date()
    date_format = mdates.DateFormatter("%b %d")
    ax.xaxis.set_major_formatter(date_format)

    # Remove the bottom and surface lines
    # ax.plot(time_num, bottom_depths, 'k-', linewidth=2, label='Bottom')
    # ax.plot(time_num, surface_elevation, 'b-', linewidth=2, label='Surface')

    # Add labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Current Speed Throughout Water Column")

    # Rotate date labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Remove legend since we removed the lines
    # ax.legend(loc='upper right')

    # Adjust layout
    fig.tight_layout()

    return fig


def add_sigma_depth_layer_bounds(df):
    for i in range(10):
        print(f"Sigma {i} depth", df[f"vap_sigma_depth_layer_{i}"].iloc[0])

    for i in range(11):
        if i == 0:
            df[f"vap_sigma_depth_bound_{i}"] = 0
        elif i == 10:
            # df[f"vap_sigma_depth_bound_{i}"] = df["vap_surface_elevation"]
            df[f"vap_sigma_depth_bound_{i}"] = (
                df["vap_surface_elevation"] + df["vap_sea_floor_depth"]
            )
            # df[f"vap_sigma_depth_bound_{i}"] = df[f"vap_sigma_depth_bound_{i - 1}"] + (
            #     df[f"vap_sigma_depth_bound_{i - 1}"]
            #     - df[f"vap_sigma_depth_layer_{i - 2}"]
            # )
        else:
            df[f"vap_sigma_depth_bound_{i}"] = (
                df[f"vap_sigma_depth_layer_{i - 1}"] + df[f"vap_sigma_depth_layer_{i}"]
            ) / 2

    return df


def calculate_tidal_periods(surface_elevation, times):
    """
    Calculate tidal period statistics and ranges between consecutive tidal cycles.

    Parameters:
    -----------
    surface_elevation : array-like
        Array of modeled water surface elevations
    times : array-like
        Array of timestamps corresponding to surface positions

    Returns:
    --------
    dict
        Dictionary containing tidal period statistics and cycle-specific data
    """

    # Find peaks (high tides) with appropriate prominence
    high_tide_indices, _ = find_peaks(surface_elevation, prominence=0.05)

    # Also find troughs (low tides)
    low_tide_indices, _ = find_peaks(-surface_elevation, prominence=0.05)

    # Sort the indices chronologically
    high_tide_indices = np.sort(high_tide_indices)
    low_tide_indices = np.sort(low_tide_indices)

    # Calculate time differences between consecutive high tides (semi-diurnal period)
    high_tide_periods = []
    low_tide_periods = []
    tidal_ranges = []
    tidal_cycles_data = []

    if len(high_tide_indices) < 2:
        # Not enough peaks to calculate periods
        return {
            "average_period_seconds": 0,
            "min_period_seconds": 0,
            "max_period_seconds": 0,
            "average_period_str": "0.00h",
            "min_period_str": "0.00h",
            "max_period_str": "0.00h",
            "tide_type": "Unknown",
            "average_range": 0,
            "min_range": 0,
            "max_range": 0,
            "min_range_cycle": None,
            "max_range_cycle": None,
            "tidal_ranges": [],
            "cycle_data": [],
        }

    # Calculate high tide periods and tidal ranges for consecutive cycles
    for i in range(1, len(high_tide_indices)):
        prev_idx = high_tide_indices[i - 1]
        curr_idx = high_tide_indices[i]

        # Find the low tide(s) between consecutive high tides
        between_low_indices = low_tide_indices[
            (low_tide_indices > prev_idx) & (low_tide_indices < curr_idx)
        ]

        # Only process valid tidal cycles with low tides between consecutive high tides
        if len(between_low_indices) > 0:
            # Use the lowest low tide between consecutive high tides
            lowest_low_idx = between_low_indices[
                np.argmin(surface_elevation[between_low_indices])
            ]

            # Calculate range for this tidal cycle
            high_tide_value = surface_elevation[prev_idx]
            low_tide_value = surface_elevation[lowest_low_idx]
            tidal_range = high_tide_value - low_tide_value
            tidal_ranges.append(tidal_range)

            # Get timestamps if available
            high_tide_time = None
            low_tide_time = None

            if times is not None:
                try:
                    if isinstance(times, pd.DatetimeIndex):
                        high_tide_time = times[prev_idx]
                        low_tide_time = times[lowest_low_idx]
                    elif hasattr(times, "iloc"):
                        high_tide_time = times.iloc[prev_idx]
                        low_tide_time = times.iloc[lowest_low_idx]
                    else:
                        high_tide_time = times[prev_idx]
                        low_tide_time = times[lowest_low_idx]
                except Exception:
                    pass

            # Record detailed data for this tidal cycle
            cycle_data = {
                "high_tide_index": prev_idx,
                "high_tide_value": high_tide_value,
                "high_tide_time": high_tide_time,
                "low_tide_index": lowest_low_idx,
                "low_tide_value": low_tide_value,
                "low_tide_time": low_tide_time,
                "tidal_range": tidal_range,
            }
            tidal_cycles_data.append(cycle_data)

        # Calculate time difference between consecutive high tides
        if times is not None:
            try:
                if isinstance(times, pd.DatetimeIndex):
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()
                elif hasattr(times, "iloc"):
                    time_diff = (
                        times.iloc[curr_idx] - times.iloc[prev_idx]
                    ).total_seconds()
                else:
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()

                # Only include reasonable periods (10-14 hours for semi-diurnal, 20-26 hours for diurnal)
                if (
                    10 * 3600 < time_diff < 14 * 3600
                    or 20 * 3600 < time_diff < 26 * 3600
                ):
                    high_tide_periods.append(time_diff)
            except Exception:
                continue

    # Calculate statistics for tidal periods
    all_periods = high_tide_periods + low_tide_periods

    # Calculate statistics if we have valid periods
    if all_periods:
        avg_period = np.mean(all_periods)
        min_period = np.min(all_periods)
        max_period = np.max(all_periods)

        # Use plain language descriptions for tide patterns
        if 10 * 3600 < avg_period < 14 * 3600:
            tide_type = "Twice Daily Tides"
        elif 20 * 3600 < avg_period < 26 * 3600:
            tide_type = "Once Daily Tides"
        else:
            tide_type = "Mixed Pattern Tides"
    else:
        # No valid periods found
        return {
            "average_period_seconds": 0,
            "min_period_seconds": 0,
            "max_period_seconds": 0,
            "average_period_str": "0.00h",
            "min_period_str": "0.00h",
            "max_period_str": "0.00h",
            "tide_type": "Unknown",
            "average_range": 0,
            "min_range": 0,
            "max_range": 0,
            "min_range_cycle": None,
            "max_range_cycle": None,
            "tidal_ranges": [],
            "cycle_data": tidal_cycles_data,
        }

    # Calculate tidal range statistics
    if tidal_ranges:
        avg_range = np.mean(tidal_ranges)
        min_range = np.min(tidal_ranges)
        max_range = np.max(tidal_ranges)

        # Find the indices of min and max ranges for reporting
        min_range_idx = np.argmin(tidal_ranges)
        max_range_idx = np.argmax(tidal_ranges)

        # Get the corresponding cycle data
        min_range_cycle = (
            tidal_cycles_data[min_range_idx]
            if min_range_idx < len(tidal_cycles_data)
            else None
        )
        max_range_cycle = (
            tidal_cycles_data[max_range_idx]
            if max_range_idx < len(tidal_cycles_data)
            else None
        )
    else:
        avg_range = 0
        min_range = 0
        max_range = 0
        min_range_cycle = None
        max_range_cycle = None

    def format_seconds_to_decimal_hours(seconds):
        # Convert seconds to hours (as a float)
        hours = seconds / 3600

        # Format to 2 decimal places and add 'h' suffix
        return f"{hours:.2f}h"

    # Create the return dictionary with all needed keys
    period_stats = {
        "average_period_seconds": avg_period if all_periods else 0,
        "min_period_seconds": min_period if all_periods else 0,
        "max_period_seconds": max_period if all_periods else 0,
        "average_period_str": format_seconds_to_decimal_hours(avg_period)
        if all_periods
        else "0.00h",
        "min_period_str": format_seconds_to_decimal_hours(min_period)
        if all_periods
        else "0.00h",
        "max_period_str": format_seconds_to_decimal_hours(max_period)
        if all_periods
        else "0.00h",
        "tide_type": tide_type if all_periods else "Unknown",
        "average_range": avg_range,
        "min_range": min_range,
        "max_range": max_range,
        "min_range_cycle": min_range_cycle,
        "max_range_cycle": max_range_cycle,
        "tidal_ranges": tidal_ranges,
        "cycle_data": tidal_cycles_data,
    }

    return period_stats


def calculate_tidal_levels(surface_positions, times=None):
    """
    Calculate model-derived tidal reference levels using plain language terminology.

    Parameters:
    -----------
    surface_positions : array-like
        Array of modeled water surface elevations
    times : array-like, optional
        Array of timestamps corresponding to surface_positions

    Returns:
    --------
    dict
        Dictionary containing tidal reference levels with plain language keys
    """

    # Convert to numpy array for calculations
    surface_positions = np.array(surface_positions)

    # Calculate Mean Water Level - the average of all water levels
    model_msl = np.mean(surface_positions)

    # Find peaks (high tides) and troughs (low tides)
    high_tide_indices, _ = find_peaks(surface_positions, prominence=0.05)
    low_tide_indices, _ = find_peaks(-surface_positions, prominence=0.05)

    # If no peaks or troughs are found, adjust the parameters or use simpler method
    if len(high_tide_indices) == 0 or len(low_tide_indices) == 0:
        # Fallback method: use simple percentiles
        print("Warning: Could not detect peaks and troughs. Using simplified method.")
        high_tides = np.sort(surface_positions)[
            -int(len(surface_positions) * 0.2) :
        ]  # Top 20%
        low_tides = np.sort(surface_positions)[
            : int(len(surface_positions) * 0.2)
        ]  # Bottom 20%

        # Create simple indices for visualization
        high_tide_indices = np.argsort(surface_positions)[
            -int(len(surface_positions) * 0.2) :
        ]
        low_tide_indices = np.argsort(surface_positions)[
            : int(len(surface_positions) * 0.2)
        ]
    else:
        # Get the water levels at high and low tides
        high_tides = surface_positions[high_tide_indices]
        low_tides = surface_positions[low_tide_indices]

    # Following simplified naming:
    # Max High Tide - The highest high tide in the record
    # Mean High Tide - average of all high tides
    # Mean Water Level - average of all water levels
    # Mean Low Tide - average of all low tides
    # Min Low Tide - The lowest low tide in the record

    # Calculate statistics with clear naming
    max_high_tide = np.max(high_tides)
    min_high_tide = np.min(high_tides)  # Lowest high tide
    mean_high_tide = np.mean(high_tides)
    mean_water_level = model_msl
    max_low_tide = np.max(low_tides)  # Highest low tide
    mean_low_tide = np.mean(low_tides)
    min_low_tide = np.min(low_tides)

    # Create a dictionary of tidal levels and associated data using simplified terminology
    tidal_data = {
        "Max High Tide": max_high_tide,  # Maximum observed high tide
        "Min High Tide": min_high_tide,  # Minimum (lowest) high tide
        "Mean High Tide": mean_high_tide,  # Average of all high tides
        "Mean Water Level": mean_water_level,  # Average of all water levels
        "Max Low Tide": max_low_tide,  # Maximum (highest) low tide
        "Mean Low Tide": mean_low_tide,  # Average of all low tides
        "Min Low Tide": min_low_tide,  # Minimum observed low tide
        "high_tide_indices": high_tide_indices,
        "low_tide_indices": low_tide_indices,
    }

    return tidal_data


def plot_tidal_statistics(
    surface_positions,
    times=None,
    x_positions=None,
    ax=None,
    model_name="FVCOM",
    data_year=None,
    location_label="",
):
    """
    Plot tidal statistics and reference levels using plain language terminology.

    Parameters:
    -----------
    surface_positions : array-like
        Array of water surface elevations
    times : array-like, optional
        Array of timestamps corresponding to surface positions
    x_positions : array-like, optional
        X-coordinates for plotting surface positions
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis will be created.
    model_name : str, default="FVCOM"
        Name of the model used for the simulation (e.g., "FVCOM", "ADCIRC")
    data_year : str, optional
        Year or year range of the simulation (e.g., "2020" or "2020-2021")
    location_label : str, optional
        Location description to include in the title

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axis object
    tidal_data : dict
        Dictionary containing calculated tidal reference levels
    """

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # If x_positions not provided, create default
    if x_positions is None:
        if times is not None and isinstance(times, pd.DatetimeIndex):
            x_positions = times
        else:
            x_positions = np.arange(len(surface_positions))

    n_times = len(x_positions)

    surface_positions = surface_positions - np.mean(
        surface_positions
    )  # Center around zero

    # Calculate tidal reference levels
    tidal_data = calculate_tidal_levels(surface_positions, times)

    # Extract tidal reference levels using clearer naming
    max_high = tidal_data["Max High Tide"]  # Maximum high tide
    min_high = tidal_data["Min High Tide"]  # Minimum (lowest) high tide
    mean_high = tidal_data["Mean High Tide"]  # Average high tide
    mean_water = tidal_data["Mean Water Level"]  # Average water level
    max_low = tidal_data["Max Low Tide"]  # Maximum (highest) low tide
    mean_low = tidal_data["Mean Low Tide"]  # Average low tide
    min_low = tidal_data["Min Low Tide"]  # Minimum low tide

    high_tide_indices = tidal_data["high_tide_indices"]
    low_tide_indices = tidal_data["low_tide_indices"]

    # Calculate tidal period statistics
    period_stats = calculate_tidal_periods(surface_positions, times)

    # Calculate total range from extremes
    tidal_range = max_high - min_low

    # Count tidal cycles
    tidal_cycles = len(high_tide_indices)
    days_of_data = 1
    if isinstance(times, pd.DatetimeIndex):
        days_of_data = (times[-1] - times[0]).total_seconds() / (24 * 3600)
    cycles_per_day = tidal_cycles / days_of_data if days_of_data > 0 else 0

    # Determine simulation period text from times
    sim_period_text = ""
    if data_year:
        sim_period_text = f"{data_year}"
    elif times is not None:
        if isinstance(times, pd.DatetimeIndex) or (
            hasattr(times, "iloc") and isinstance(times.iloc[0], pd.Timestamp)
        ):
            if isinstance(times, pd.DatetimeIndex):
                start_time = times[0]
                end_time = times[-1]
            else:
                start_time = times.iloc[0]
                end_time = times.iloc[-1]

            if start_time.year == end_time.year:
                sim_period_text = f"{start_time.year}"
            else:
                sim_period_text = f"{start_time.year}-{end_time.year}"

            # Add more detailed period information
            start_str = start_time.strftime("%b %d, %Y")
            end_str = end_time.strftime("%b %d, %Y")
            sim_period_text = f"{start_str} to {end_str}"

    # Detailed labels with values for y-axis annotations using plain language
    tidal_level_dict = {
        "Max High Tide*": max_high,
        "Min High Tide*": min_high,
        "Mean High Tide*": mean_high,
        "Mean Water Level*": mean_water,
        "Max Low Tide*": max_low,
        "Mean Low Tide*": mean_low,
        "Min Low Tide*": min_low,
    }

    # Colors for the tidal reference levels - using a more distinct color palette
    line_colors = {
        "Max High Tide": "#D81B60",  # Bright red
        "Min High Tide": "#E57373",  # Lighter red
        "Mean High Tide": "#8E24AA",  # Purple
        "Mean Water Level": "#000000",  # Black
        "Max Low Tide": "#64B5F6",  # Light blue
        "Mean Low Tide": "#0288D1",  # Medium blue
        "Min Low Tide": "#00838F",  # Teal
    }

    # Line styles for different types of datums - more distinct
    line_styles = {
        "Max High Tide": (0, (5, 5)),  # Dashed for extremes
        "Min High Tide": (0, (1, 1)),  # Dotted for min/max
        "Mean High Tide": "-",  # Solid for means
        "Mean Water Level": "-",  # Solid for mean water level
        "Max Low Tide": (0, (1, 1)),  # Dotted for min/max
        "Mean Low Tide": "-",  # Solid for means
        "Min Low Tide": (0, (5, 5)),  # Dashed for extremes
    }

    # Plot the water surface elevation with a pleasing blue color
    ax.plot(
        x_positions,
        surface_positions,
        color="#1E88E5",  # A pleasing shade of blue
        linewidth=0.5,
        zorder=10,
    )

    # Set up dictionary for legend elements
    legend_elements = []
    legend_elements.append(
        Line2D([0], [0], color="#1E88E5", lw=1.5, label="Water Surface Elevation")
    )

    # Add horizontal lines for each tidal level without direct annotations
    for label, value in tidal_level_dict.items():
        # Extract base key for color and style lookup
        base_key = label.replace("*", "")

        # Add the horizontal line
        line = ax.axhline(
            y=value,
            color=line_colors.get(base_key, "gray"),
            linestyle=line_styles.get(base_key, "--"),
            linewidth=1.2,
            alpha=0.8,
            zorder=5,
        )

        # Add to legend elements
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=line_colors.get(base_key, "gray"),
                linestyle=line_styles.get(base_key, "--"),
                lw=1.2,
                label=f"{label}: {value:.2f} m",
            )
        )

    # Format min and max range date information
    min_range_date_str = "N/A"
    max_range_date_str = "N/A"

    if period_stats.get("min_range_cycle") and period_stats["min_range_cycle"].get(
        "high_tide_time"
    ):
        min_ht = period_stats["min_range_cycle"]["high_tide_time"]
        min_lt = period_stats["min_range_cycle"]["low_tide_time"]
        min_range_val = period_stats["min_range_cycle"]["tidal_range"]

        if hasattr(min_ht, "strftime"):
            min_range_date_str = (
                f"{min_ht.strftime('%b %d, %Y')}, {min_range_val:.2f} m"
            )

    if period_stats.get("max_range_cycle") and period_stats["max_range_cycle"].get(
        "high_tide_time"
    ):
        max_ht = period_stats["max_range_cycle"]["high_tide_time"]
        max_lt = period_stats["max_range_cycle"]["low_tide_time"]
        max_range_val = period_stats["max_range_cycle"]["tidal_range"]

        if hasattr(max_ht, "strftime"):
            max_range_date_str = (
                f"{max_ht.strftime('%b %d, %Y')}, {max_range_val:.2f} m"
            )

    # Create a text box with tidal statistics using the simplest possible terms
    stats_text = (
        f"Surface Elevation Statistics ({sim_period_text}):\n"
        f"Total Range (Max to Min): {tidal_range:.2f} m\n"
        f"High Tide Range: {max_high - min_high:.2f} m\n"
        f"Low Tide Range: {max_low - min_low:.2f} m\n"
        f"Tidal Range: Avg: {period_stats['average_range']:.2f} m, Min: {period_stats['min_range']:.2f} m, Max: {period_stats['max_range']:.2f} m\n"
        f"Min Range Date: {min_range_date_str}\n"
        f"Max Range Date: {max_range_date_str}\n"
        f"Number of Tide Cycles: {tidal_cycles} ({cycles_per_day:.1f}/day)\n"
        f"Tide Period: Avg: {period_stats['average_period_str']}, Range: {period_stats['min_period_str']} to {period_stats['max_period_str']}\n"
        f"Tide Pattern: {period_stats.get('tide_type', 'Unknown')}\n"
        f"* Based on model data, not actual measurements"
    )

    # Add the statistics text box
    ax.text(
        1.01,
        -0.01,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            ec="gray",
        ),
    )

    # Process tidal events (high and low tides)
    if len(high_tide_indices) > 0:
        # Draw markers for high tide events
        for idx in high_tide_indices:
            if idx < len(surface_positions):
                if isinstance(x_positions, pd.DatetimeIndex) or (
                    hasattr(x_positions, "iloc")
                    and isinstance(x_positions.iloc[0], pd.Timestamp)
                ):
                    event_x = x_positions[idx]
                else:
                    event_x = idx

                # Add a marker for the high tide point
                ax.plot(event_x, surface_positions[idx], "ro", markersize=3, alpha=0.7)

                # Add a short horizontal line to mark the high tide
                if isinstance(event_x, pd.Timestamp):
                    start_x = event_x - pd.Timedelta(minutes=30)
                    end_x = event_x + pd.Timedelta(minutes=30)
                else:
                    start_x = event_x - n_times * 0.005
                    end_x = event_x + n_times * 0.005

                ax.plot(
                    [start_x, end_x],
                    [surface_positions[idx], surface_positions[idx]],
                    color="red",
                    linewidth=1,
                    alpha=0.5,
                )

    if len(low_tide_indices) > 0:
        # Draw markers for low tide events
        for idx in low_tide_indices:
            if idx < len(surface_positions):
                if isinstance(x_positions, pd.DatetimeIndex) or (
                    hasattr(x_positions, "iloc")
                    and isinstance(x_positions.iloc[0], pd.Timestamp)
                ):
                    event_x = x_positions[idx]
                else:
                    event_x = idx

                # Add a marker for the low tide point
                ax.plot(event_x, surface_positions[idx], "bo", markersize=3, alpha=0.7)

                # Add a short horizontal line to mark the low tide
                if isinstance(event_x, pd.Timestamp):
                    start_x = event_x - pd.Timedelta(minutes=30)
                    end_x = event_x + pd.Timedelta(minutes=30)
                else:
                    start_x = event_x - n_times * 0.005
                    end_x = event_x + n_times * 0.005

                ax.plot(
                    [start_x, end_x],
                    [surface_positions[idx], surface_positions[idx]],
                    color="blue",
                    linewidth=1,
                    alpha=0.5,
                )

    # Add high/low tide markers to the legend
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=6,
            label="High Tide",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=6,
            label="Low Tide",
        )
    )
    #
    # # Set up x-axis based on datetime if available
    # if isinstance(x_positions, pd.DatetimeIndex) or (
    #     hasattr(x_positions, "iloc") and isinstance(x_positions.iloc[0], pd.Timestamp)
    # ):
    #     # Format datetime x-axis with month names
    #     if isinstance(x_positions, pd.DatetimeIndex):
    #         date_range = (x_positions[-1] - x_positions[0]).days
    #     else:
    #         date_range = (x_positions.iloc[-1] - x_positions.iloc[0]).days
    #
    #     if date_range <= 3:
    #         # For a few days of data, show hours
    #         ax.xaxis.set_major_formatter(
    #             lambda x, pos: format_date_label(pd.Timestamp(mdates.num2date(x)))
    #         )
    #         ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    #         ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    #     elif date_range <= 14:
    #         # For up to two weeks, show days
    #         ax.xaxis.set_major_formatter(
    #             lambda x, pos: format_date_label(pd.Timestamp(mdates.num2date(x)))
    #         )
    #         ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    #         ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    #     else:
    #         # For longer periods, show weeks with day minor ticks
    #         ax.xaxis.set_major_formatter(
    #             lambda x, pos: format_date_label(pd.Timestamp(mdates.num2date(x)))
    #         )
    #         ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    #         ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    #
    #     plt.xticks(rotation=90)
    #     plt.xlabel("Time")
    # else:
    #     # For numeric indices, use regular spacing
    #     if len(x_positions) > 20:
    #         tick_interval = len(x_positions) // 10
    #         ax.set_xticks(range(0, len(x_positions), tick_interval))
    #
    #     plt.xlabel("Time Index")
    #
    # Ensure full tidal range is visible with padding
    y_padding = 0.05 * (max_high - min_low)
    ax.set_ylim(min_low - y_padding, max_high + y_padding)

    plt.xticks(rotation=0)

    # Set labels and grid
    plt.ylabel("Surface Elevation [m]")
    plt.xlabel("Time [UTC]")
    plt.grid(True, alpha=0.3, which="both")

    # Create title with model information
    # if not location_label:
    #     location_label = "Piscataqua River, New Hampshire"

    # plt.title(
    #     f"H2O High Resolution Tidal Model\n{location_label}\n"
    #     f"{model_name} Tidal Analysis ({sim_period_text})"
    # )

    plt.title(sim_period_text)

    # Add a title for the legend with increased font size
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Legend",
        title_fontsize=10,
    )

    plt.tight_layout()

    return fig, ax, tidal_data


def format_date_label(timestamp):
    """
    Format date labels for tidal event annotations.
    - Only show year when it changes
    - Use month names instead of numbers

    Parameters:
    -----------
    timestamp : datetime
        The timestamp to format

    Returns:
    --------
    str
        Formatted date string
    """
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Get month name
    month_name = month_names[timestamp.month - 1]

    # Different formats depending on the date parts needed
    if timestamp.hour == 0 and timestamp.minute == 0:
        # Just show the date
        return f"{month_name} {timestamp.day}"
    else:
        # Show date and time
        return (
            f"{month_name} {timestamp.day}, {timestamp.hour:02d}:{timestamp.minute:02d}"
        )


def plot_sigma_layers(
    df,
    cmap,
    layer_name="speed",
    units="m/s",
    # dataset_label=None,
    # lat=None,
    # lon=None,
    fixed_bottom=True,
    ax=None,
):
    """
    Plot water speed at different sigma layers over time with tidal reference levels.
    - Each sigma layer is represented as a polygon
    - Surface is at 0, positive depth values increase downward
    - Bottom remains fixed
    - 10 uniform sigma layers (11 sigma levels)
    - Includes standard oceanographic tidal reference levels

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing sigma layer data
    cmap : matplotlib.colors.Colormap
        Colormap for the speed values
    layer_name : str, default="speed"
        Name of the layer data to visualize
    units : str, default="m/s"
        Units for the layer data
    dataset_label : str, optional
        Label for the dataset in the title
    lat : float, optional
        Latitude for the title
    lon : float, optional
        Longitude for the title
    fixed_bottom : bool, default=False
        If True, keeps the seafloor at a fixed position and shows surface
        elevation changes (tides). If False, shows standard depth representation.
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axes instead of creating a new figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure
    ax : matplotlib.axes.Axes
        The axes
    """
    label = layer_name.replace("_", " ").title()

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Get the time steps (assuming datetime index)
    times = df.index
    n_times = len(times)

    # Number of sigma layers (10 layers, 11 levels)
    n_layers = 10

    # Create arrays to store the polygons and colors
    verts = []
    speeds = []

    if fixed_bottom:
        # Calculate offset for each time step to align seafloor at a fixed position
        max_seafloor_depth = df["vap_sea_floor_depth"].max()

        # Create arrays to store surface elevation for tidal analysis
        surface_positions = []

        # For each time step
        for t in range(n_times):
            # Get the current seafloor depth
            current_seafloor_depth = df["vap_sea_floor_depth"].iloc[t]

            # Calculate the offset needed to align the seafloor at the fixed depth
            seafloor_offset = max_seafloor_depth - current_seafloor_depth

            # Store surface position (this represents the tide level at each timestep)
            surface_positions.append(seafloor_offset)

            # For each sigma layer
            for i in range(n_layers):
                # Get the depth bounds for this layer (from the surface)
                top_bound = df[f"vap_sigma_depth_bound_{i}"].iloc[t]
                bottom_bound = df[f"vap_sigma_depth_bound_{i + 1}"].iloc[t]

                # Apply the offset to transform depths to elevations with fixed seafloor
                top_bound_transformed = top_bound + seafloor_offset
                bottom_bound_transformed = bottom_bound + seafloor_offset

                # Get the speed for this layer
                speed = df[f"vap_sea_water_{layer_name}_layer_{i}"].iloc[t]
                speeds.append(speed)

                # Create polygon vertices with transformed bounds
                verts.append(
                    [
                        (t, top_bound_transformed),
                        (t + 1, top_bound_transformed),
                        (t + 1, bottom_bound_transformed),
                        (t, bottom_bound_transformed),
                    ]
                )
    else:
        # Standard depth representation (original code)
        for t in range(n_times):
            for i in range(n_layers):
                top_bound = df[f"vap_sigma_depth_bound_{i}"].iloc[t]
                bottom_bound = df[f"vap_sigma_depth_bound_{i + 1}"].iloc[t]
                speed = df[f"vap_sea_water_{layer_name}_layer_{i}"].iloc[t]
                speeds.append(speed)
                verts.append(
                    [
                        (t, top_bound),
                        (t + 1, top_bound),
                        (t + 1, bottom_bound),
                        (t, bottom_bound),
                    ]
                )

    # Create a PolyCollection from the vertices
    poly = PolyCollection(verts, array=np.array(speeds), cmap=cmap, edgecolors="face")

    # Add the collection to the plot
    ax.add_collection(poly)

    # Plot the surface elevation line
    x_positions = np.arange(len(df.index))

    if fixed_bottom:
        # # Calculate tidal reference levels
        # tidal_data = calculate_tidal_levels(surface_positions, times)
        #
        # # Plot tidal statistics and reference levels
        # handles = plot_tidal_statistics(
        #     ax, tidal_data, surface_positions, times, x_positions
        # )
        ax.plot(
            x_positions,
            surface_positions,
            color="#2979FF",
            linewidth=0.5,
            label="Surface Elevation",
        )

    # else:
    #     # Original surface elevation (at depth 0)
    #     ax.plot(
    #         x_positions,
    #         np.zeros(
    #             len(df.index)
    #         ),  # Surface is always at depth 0 in sigma coordinates
    #         color="#2979FF",
    #         linewidth=0.5,
    #         zorder=10,
    #         label="Surface Elevation",
    #     )
    #
    #     # Set the axis limits for standard view
    #     max_depth = df["vap_sea_floor_depth"].max()
    #     ax.set_ylim(-0.05, max_depth * 1.05)

    # Add a colorbar
    cbar = plt.colorbar(poly, ax=ax)
    cbar.set_label(f"{label} [{units}]")

    # Set the axis limits
    ax.set_xlim(0, n_times)

    # Invert the y-axis for both visualization types
    ax.invert_yaxis()

    # Set labels and title
    ax.set_xlabel("Time [UTC]")

    # Use proper oceanographic labels for y-axis
    if fixed_bottom is True:
        ax.set_ylabel("Elevation [m]")
    else:
        ax.set_ylabel("Depth [m]")

    # if dataset_label is not None and lat is not None and lon is not None:
    #     ax.set_title(
    #         f"H2O High Resolution Tidal Hindcast\n{dataset_label} | {lat:.06f}, {lon:.06f},\nSea Water {label} [{units}] with Tidal Reference Levels"
    #     )
    # else:
    #     ax.set_title(f"Sea Water {label} [{units}] with Tidal Reference Levels")

    # Improve the x-axis labels if datetime index
    if isinstance(times, pd.DatetimeIndex):
        n_ticks = min(12, n_times)
        tick_indices = np.linspace(0, n_times - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [times[i].strftime("%Y-%m-%d %H:%M") for i in tick_indices], rotation=45
        )

    # Add a legend
    # if fixed_bottom is True:
    # Add legend with tidal zones
    # first_legend = ax.legend(handles=handles, loc="upper left", title="Tidal Zones")
    # ax.add_artist(first_legend)
    #
    # # Add another legend for the lines
    # ax.legend(loc="lower right", title="Reference Lines")

    plt.tight_layout()

    # Print the calculated tidal reference values
    # if fixed_bottom:
    #     print("Calculated Tidal Reference Levels:")
    #     print(f"HAT (Highest Astronomical Tide): {tidal_data['HAT']:.3f} m")
    #     print(f"MHWS (Mean High Water Springs): {tidal_data['MHWS']:.3f} m")
    #     print(f"MHWN (Mean High Water Neaps): {tidal_data['MHWN']:.3f} m")
    #     print(f"MSL (Mean Sea Level): {tidal_data['MSL']:.3f} m")
    #     print(f"MLWN (Mean Low Water Neaps): {tidal_data['MLWN']:.3f} m")
    #     print(f"MLWS (Mean Low Water Springs): {tidal_data['MLWS']:.3f} m")
    #     print(f"LAT (Lowest Astronomical Tide): {tidal_data['LAT']:.3f} m")

    return fig, ax


def plot_sigma_layers_speed(
    df,
    # dataset_label=None,
    # lat=None,
    # lon=None,
    fixed_bottom=True,
    ax=None,
):
    return plot_sigma_layers(
        df,
        cmocean.cm.thermal,
        layer_name="speed",
        units="m/s",
        # dataset_label=dataset_label,
        # lat=lat,
        # lon=lon,
        fixed_bottom=fixed_bottom,
        ax=ax,
    )


def plot_sigma_layers_direction(
    # df, dataset_label=None, lat=None, lon=None, fixed_bottom=False
    df,
    fixed_bottom=True,
):
    return plot_sigma_layers(
        df,
        # "twilight",
        cmocean.cm.phase,
        layer_name="to_direction",
        units="Degrees CW from True North",
        # dataset_label=dataset_label,
        # lat=lat,
        # lon=lon,
        fixed_bottom=fixed_bottom,
    )


def add_sigma_depth_bounds(df):
    """
    Vectorized version to calculate sigma depth bounds for each time step.
    - Surface elevation changes
    - Bottom is at sea floor depth (fixed)
    - 10 uniform sigma layers (11 sigma levels)
    """
    # Extract sea floor depths as a Series
    sea_floor = df["vap_sea_floor_depth"]

    # Calculate sigma layer depths (centers)
    for i in range(10):
        # Vectorized calculation for all time steps at once
        sigma_factor = (i + 0.5) / 10
        df[f"vap_sigma_depth_layer_{i}"] = sea_floor * sigma_factor

    # Calculate sigma level bounds (11 bounds for 10 layers)
    for i in range(11):
        if i == 0:
            # Surface is always at 0
            df[f"vap_sigma_depth_bound_{i}"] = 0
        elif i == 10:
            # Bottom is at sea floor depth
            df[f"vap_sigma_depth_bound_{i}"] = sea_floor
        else:
            # Intermediate bounds at uniform intervals
            df[f"vap_sigma_depth_bound_{i}"] = sea_floor * (i / 10)

    return df


def render_plot(
    label, file_label, plot_label, output_dir, plot_function, skip_if_exists=True
):
    plt.close()
    plt.clf()
    print(f"Plotting {plot_label}...")
    file_plot_label = plot_label.title().replace(" ", "_").lower()

    output_file_path = Path(output_dir, f"{file_label}.{file_plot_label}.png")
    if skip_if_exists is True and output_file_path.exists():
        print(f"Skipping {output_file_path} as it already exists.")
        return

    try:
        plot_function()
        plt.suptitle(f"{label}\n{plot_label}")
        plt.tight_layout()
        plt.savefig(
            output_file_path,
            dpi=300,
            bbox_inches="tight",
        )

    except Exception as e:
        print(f"Error plotting {plot_label}: {e}")
        return


def calculate_haversine_distance_meters(point_a, point_b):
    """
    Calculate the Haversine distance between two points on the Earth.
    Returns the distance in meters.
    """

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    # point_a_in_radians = [radians(_) for _ in point_a]
    # point_b_in_radians = [radians(_) for _ in point_b]
    # raw_distance = haversine_distances([point_a_in_radians, point_b_in_radians])
    # print(f"Raw Haversine Distance: {raw_distance}")
    # distance = float(np.max(haversine_distances([point_a_in_radians, point_b_in_radians])))
    # distance_km = distance * 6371000/1000  # multiply by Earth radius to get kilometers
    # distance_meters = distance_km / 1000

    distance_km = haversine(point_a, point_b)
    distance_meters = distance_km * 1000  # convert kilometers to meters

    print(
        f"The Distance between {point_a} and {point_b} is {distance_meters:.2f} meters."
    )
    return distance_meters


# output_dir = Path("analysis/by_point/")
output_dir = OUTPUT_PATH
output_dir.mkdir(parents=True, exist_ok=True)
# output_dir = Path(output_dir, "katie_puget_sound")
# output_dir.mkdir(parents=True, exist_ok=True)

# Coordinates from Katie: 48°09'14.0"N 122°46'26.8"W
# target_lat = 48.1538889
# target_lon = -122.7741111


def read_netcdf_metadata_from_parquet(parquet_path):
    """
    Read data and netCDF-compatible metadata from a parquet file.

    Args:
        parquet_path: path to parquet file

    Returns:
        tuple: (df, file_meta, var_meta)
    """
    # Read parquet as PyArrow table first
    table = pq.read_table(parquet_path)

    # Convert to pandas DataFrame, preserving index
    df = table.to_pandas()

    df = df.set_index("time", drop=True)

    # 1. Read FILE-LEVEL metadata (global attributes) - from table schema metadata
    file_metadata = {}
    if table.schema.metadata:
        for key, value in table.schema.metadata.items():
            # Convert bytes back to strings for keys and values
            file_metadata[key.decode("utf-8")] = value.decode("utf-8")

    # 2. Read SCHEMA-LEVEL metadata (variable attributes) - from field metadata

    print("=== SCHEMA DEBUG INFO ===")
    print(f"Schema: {table.schema}")
    print(f"Schema metadata: {table.schema.metadata}")

    print("\n=== FIELD DETAILS ===")
    for i, field in enumerate(table.schema):
        print(f"\nField {i}: {field.name}")
        print(f"  Type: {field.type}")
        print(f"  Nullable: {field.nullable}")
        print(f"  Metadata: {field.metadata}")
        print(f"  Metadata type: {type(field.metadata)}")

        if field.metadata:
            print("  Metadata items:")
            for key, value in field.metadata.items():
                print(f"    {key} ({type(key)}): {value} ({type(value)})")

    var_metadata = {}
    for field in table.schema:
        if field.metadata:
            field_attrs = {}
            for key, value in field.metadata.items():
                # Convert bytes back to strings
                field_attrs[key.decode("utf-8")] = value.decode("utf-8")

            if field_attrs:  # Only add if there are attributes
                var_metadata[field.name] = field_attrs

    return df, file_metadata, var_metadata


# for parquet_file in parquet_files[:1]:  # UNH Living Bridge
for parquet_file in parquet_files:  # UNH Living Bridge
    # for parquet_file in parquet_files[0:1]:  # Minas Basin
    # for parquet_file in parquet_files[0:1]:
    print(f"Starting analysis on {parquet_file}...")

    # df = pd.read_parquet(parquet_file)
    df, file_meta, var_meta = read_netcdf_metadata_from_parquet(parquet_file)
    # print(f"DF read in {time.time() - start:.2f} seconds..")
    print("Data loaded successfully!")

    print(var_meta["vap_surface_elevation"])

    # meta = pq.read_metadata(parquet_file)

    # print(df.info())
    # print(file_meta)
    for key, value in var_meta.items():
        print(f"Variable: {key}")
        if key not in df.columns:
            print(f"  Variable {key} not found in DataFrame columns.")
            exit()
        for attr_key, attr_value in value.items():
            print(f"  {attr_key}: {attr_value}")

    # print(meta)

    # exit()

    for col in df.columns:
        print(col)

    # exit()

    print("Generating sigma depth bounds...")
    df = add_sigma_depth_bounds(df)
    for i in range(11):
        iloc = 0
        print(f"Sigma Depth [{i}]: {df[f'vap_sigma_depth_bound_{i}'].iloc[iloc]}")

    # df = df.iloc[:500]
    #
    # for i in range(10):
    #     df[f"vap_sigma_depth_layer_{i}"].plot(label=f"Layer {i} Depth")
    #
    # plt.legend()
    #
    # plt.show()
    #
    # exit()

    # for col in df.columns:
    #     print(col)
    # exit()

    # print(df["vap_surface_elevation"].iloc[0])

    if "lat" in df.columns:
        df["lat_center"] = df["lat"]

    if "lon" in df.columns:
        df["lon_center"] = df["lon"]

    schema = pq.read_schema(parquet_file)
    meta = standardize_metadata(schema.metadata)

    dataset_name = meta.get("dataset_name", "Unknown Dataset")
    datastream = meta.get("datastream", "Unknown Datastream")
    lat_str = f"{df['lat_center'].iloc[0]:.06f}"
    lon_str = f"{df['lon_center'].iloc[0]:.06f}"

    print(f"Dataset Name: {dataset_name}")
    print(f"Datastream: {datastream}")

    # distance_to_target = calculate_haversine_distance_meters([target_lat, target_lon], [df["lat_center"].iloc[0], df["lon_center"].iloc[0]])
    # distance_to_target = calculate_haversine_distance_meters(
    #     (target_lat, target_lon), (df["lat_center"].iloc[0], df["lon_center"].iloc[0])
    # )
    # print(f"Distance to target point: {distance_to_target:.2f} meters")

    # this_output_dir = Path(
    #     output_dir,
    #     f"puget_sound.lat={lat_str}.lon={lon_str}.target_lat={target_lat}.target_lon={target_lon}.distance_to_target={distance_to_target:.2f}m",
    # )
    this_output_dir = Path(output_dir, dataset_name)
    this_output_dir.mkdir(parents=True, exist_ok=True)

    label = (
        f"H2O High Resolution Tidal Hindcast\n{dataset_name} | {lat_str}, {lon_str}"
    )
    short_label = f"{dataset_name} | {lat_str}, {lon_str}"
    file_label = (
        f"{dataset_name}.doe_h2o_high_res_tidal_hindcast.lat={lat_str}.lon={lon_str}"
    )
    # dataset_name = meta["global:dataset_name"]
    # datastream = meta["global:datastream"]

    # New Data
    # dataset_name = meta["global:dataset_name"]
    # datastream = meta["global:datastream"]

    df["dataset_name"] = dataset_name
    df["datastream"] = datastream

    # print("Plotting velocity profile with histograms...")
    render_plot(
        label,
        file_label,
        "Velocity and Direction Overview",
        this_output_dir,
        lambda: plot_velocity_profile_with_histograms(df),
    )
    # fig = plot_velocity_profile_with_histograms(df)
    # plt.suptitle(f"{label}\nVelocity and Direction Overview")
    # plt.tight_layout()
    # plt.savefig(
    #     Path(this_output_dir, f"{file_label}.velocity_and_direction_overview.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()
    # exit()
    for i in range(10):
        min_depth = df[f"vap_sigma_depth_layer_{i}"].min()
        max_depth = df[f"vap_sigma_depth_layer_{i}"].max()
        depth_range_str = f"{min_depth:.2f} to {max_depth:.2f} m"
        render_plot(
            label,
            file_label,
            f"Tidal Joint Probability Distribution Sigma Layer {i} Depth Range {depth_range_str}",
            this_output_dir,
            lambda: generate_tidal_joint_probability(df, i),
        )
    # plt.show()
    # plot_velocity_profile(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Velocity Profile",
    #     this_output_dir,
    #     lambda: plot_velocity_profile(df),
    # )
    # plot_current_rose(df, layer=4)
    # for i in range(10):
    #     render_plot(
    #         label,
    #         file_label,
    #         f"Sea Water Speed Speed Rose Sigma Layer {i}",
    #         this_output_dir,
    #         lambda: plot_current_rose(df, layer=i),
    #     )
    # plot_tidal_time_series(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Time Series",
    #     this_output_dir,
    #     lambda: plot_tidal_time_series(df),
    # )
    # plot_velocity_exceedance(df)
    render_plot(
        label,
        file_label,
        "Velocity Exceedance Probability",
        this_output_dir,
        lambda: plot_velocity_exceedance(df),
    )
    # analyze_power_density(df, layer=4)
    # for i in range(10):
    #     render_plot(
    #         label,
    #         file_label,
    #         f"Power Density Sigma Layer {i}",
    #         this_output_dir,
    #         lambda: analyze_power_density(df, layer=i),
    #     )
    # plot_tidal_velocity_profile(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Velocity Profile",
    #     this_output_dir,
    #     lambda: plot_tidal_velocity_profile(df),
    # )
    # plot_power_density_profile(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Power Density Profile",
    #     this_output_dir,
    #     lambda: plot_power_density_profile(df),
    # )
    # plot_tidal_rose(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Rose",
    #     this_output_dir,
    #     lambda: plot_tidal_rose(df),
    # )
    # plot_tidal_exceedance(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Exceedance",
    #     this_output_dir,
    #     lambda: plot_tidal_exceedance(df),
    # )
    # plot_tidal_harmonics(df, layer=4)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Harmonics Sigma Layer 0",
    #     this_output_dir,
    #     lambda: plot_tidal_harmonics(df, layer=0),
    # )
    # plot_tidal_harmonic_analysis(df, layer=4)
    for i in range(10):
        render_plot(
            label,
            file_label,
            f"Tidal Harmonics Sigma Layer {i}",
            this_output_dir,
            lambda: plot_tidal_harmonic_analysis(df, layer=i),
        )
    # create_tidal_resource_dashboard(df)
    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Resource Dashboard",
    #     this_output_dir,
    #     lambda: create_tidal_resource_dashboard(df),
    # )

    for i in range(10):
        render_plot(
            label,
            file_label,
            f"Tidal Asymmetry Sigma Layer {i}",
            this_output_dir,
            lambda: plot_tidal_asymmetry(df, layer=i),
        )

    for i in range(10):
        min_depth = df[f"vap_sigma_depth_layer_{i}"].min()
        max_depth = df[f"vap_sigma_depth_layer_{i}"].max()
        depth_range_str = f"{min_depth:.2f} to {max_depth:.2f} m"
        render_plot(
            label,
            file_label,
            f"Tidal Energy Yield Sigma Layer {i} Depth Range {depth_range_str}",
            this_output_dir,
            lambda: plot_tidal_energy_yield(
                df,
                layer=i,
            ),
        )

    render_plot(
        label,
        file_label,
        "Velocity Shear Profile",
        this_output_dir,
        lambda: plot_velocity_shear_profile(df),
    )

    # for i in range(10):
    #     render_plot(
    #         label,
    #         file_label,
    #         f"Tidal Phase Analysis Sigma Layer {i}",
    #         this_output_dir,
    #         lambda: plot_tidal_phase_analysis(df, layer=i),
    #     )

    # render_plot(
    #     label,
    #     file_label,
    #     "Tidal Site Assessment",
    #     this_output_dir,
    #     lambda: generate_tidal_site_assessment(df, site_name=short_label),
    # )

    render_plot(
        label,
        file_label,
        "FFT Analysis",
        this_output_dir,
        lambda: plot_fft(df),
    )

    # render_plot(
    #     label,
    #     file_label,
    #     "Speed Mesh",
    #     this_output_dir,
    #     lambda: plot_speed_mesh(df),
    # )

    render_plot(
        label,
        file_label,
        "Sea Water Speed",
        this_output_dir,
        lambda: plot_sigma_layers_speed(df),
    )

    render_plot(
        label,
        file_label,
        "Sea Water To Direction",
        this_output_dir,
        lambda: plot_sigma_layers_direction(df),
    )
    render_plot(
        label,
        file_label,
        "Surface Elevation Analysis",
        this_output_dir,
        lambda: plot_tidal_statistics(df["vap_surface_elevation"], df.index),
    )

    base_label = label
    base_file_label = file_label

    monthly_groups = df.groupby(pd.Grouper(freq="M"))
    for period, monthly_df in monthly_groups:
        if len(monthly_df) == 0:  # Skip empty periods
            continue

        month_str = period.strftime("%Y-%m")
        this_label = f"{base_label} - Monthly ({month_str})"
        this_file_label = f"{base_file_label}_monthly_{month_str}"

        render_plot(
            this_label,
            this_file_label,
            "Sea Water Speed",
            this_output_dir,
            lambda df=monthly_df: plot_sigma_layers_speed(df),
        )

        render_plot(
            this_label,
            this_file_label,
            "Sea Water To Direction",
            this_output_dir,
            lambda df=monthly_df: plot_sigma_layers_direction(df),
        )
        render_plot(
            this_label,
            this_file_label,
            "Surface Elevation Analysis",
            this_output_dir,
            lambda df=monthly_df: plot_tidal_statistics(
                df["vap_surface_elevation"], df.index
            ),
        )

    # 7-day splits
    weekly_groups = df.groupby(pd.Grouper(freq="7D"))
    # for period, weekly_df in weekly_groups:
    for period, weekly_df in list(weekly_groups)[:3]:
        if len(weekly_df) == 0:  # Skip empty periods
            continue

        start_date = period.strftime("%Y-%m-%d")
        end_date = (period + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        label = f"{base_label} - 7 Days ({start_date} to {end_date})"
        file_label = f"{base_file_label}_7days_{start_date}"

        render_plot(
            label,
            file_label,
            "Sea Water Speed",
            this_output_dir,
            lambda df=weekly_df: plot_sigma_layers_speed(df),
        )
        render_plot(
            label,
            file_label,
            "Sea Water To Direction",
            this_output_dir,
            lambda df=weekly_df: plot_sigma_layers_direction(df),
        )
        render_plot(
            label,
            file_label,
            "Surface Elevation Analysis",
            this_output_dir,
            lambda df=weekly_df: plot_tidal_statistics(
                df["vap_surface_elevation"], df.index
            ),
        )

    # exit()
    # plt.show()
    # fig, ax = plt.subplots(1, 1)
    # tidal_stats = calculate_tidal_levels(df["vap_surface_elevation"], df.index)
    # Set custom titles for each subplot
    # axes[0].set_title("Standard Depth View - Sea Water Speed [m/s]")
    # axes[1].set_title("Fixed Bottom View - Sea Water Speed [m/s]")
    #
    # # Add an overall title for the entire figure
    # fig.suptitle(
    #     f"H2O High Resolution Tidal Hindcast\nPiscataqua River, New Hampshire | {df['lat_center'].iloc[0]:.06f}, {df['lon_center'].iloc[0]:.06f}",
    #     fontsize=16,
    # )
    #
    # plt.tight_layout()
    # fig.subplots_adjust(top=0.92)
    # plt.show()
    # plot_sigma_layers_direction(
    #     df,
    #     # dataset_label="Piscataqua River, New Hampshire",
    #     # dataset_label="Western Passage, Maine",
    #     dataset_label=dataset_name,
    #     lat=df["lat_center"].iloc[0],
    #     lon=df["lon_center"].iloc[0],
    #     fixed_bottom=True,
    # )
    # plt.show()
    #
    # plot_tidal_statistics(df["vap_surface_elevation"], df.index)
    # plt.show()
