import logging
import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from scipy import ndimage
from sklearn.base import BaseEstimator
from tqdm import tqdm

import ccf_streamlines.projection as ccfproj
from cuml.svm import SVC

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress all warnings from CUML
warnings.filterwarnings("ignore", module="cuml.*")

class DecisionBoundaryEdgeDisplay:
    """Visualization of classifier decision boundaries using edge detection.
    
    This class creates a visualization that focuses on the boundaries between
    different decision regions in a classification model, highlighting exactly
    where the model transitions from predicting one class to another.
    
    Attributes:
        xx0: np.ndarray
            First axis grid coordinates.
        xx1: np.ndarray
            Second axis grid coordinates.
        response: np.ndarray
            The classifier's predicted classes across the grid.
        boundary_mask: np.ndarray
            Boolean mask indicating boundary pixels.
        estimator_: BaseEstimator
            The fitted classifier.
        ax_: plt.Axes
            The matplotlib axes.
        surface_: Union[plt.QuadMesh, plt.QuadContourSet]
            The visualization surface.
    """
    
    def __init__(
        self, 
        xx0: np.ndarray, 
        xx1: np.ndarray, 
        response: np.ndarray,
        boundary_mask: np.ndarray,
        estimator: BaseEstimator,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Initialize the DecisionBoundaryEdgeDisplay.
        
        Parameters:
            xx0: np.ndarray
                First axis grid coordinates.
            xx1: np.ndarray
                Second axis grid coordinates.
            response: np.ndarray
                The classifier's predicted classes across the grid.
            boundary_mask: np.ndarray
                Boolean mask indicating boundary pixels.
            estimator: BaseEstimator
                The fitted classifier.
            ax: Optional[plt.Axes]
                The matplotlib axes to plot on, creates new axes if None.
        """
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.response_mesh = response.copy()
        self.boundary_mask = boundary_mask
        self.estimator_ = estimator
        self.ax_ = ax or plt.gca()
        
        # These will be set in plot()
        self.surface_ = None
        self.colorbar_ = None
    
    def plot(
        self, 
        fill_regions: bool = False,
        boundary_color: str = "black",
        boundary_width: float = 1.0,
        boundary_alpha: float = 1.0,
        regions_alpha: float = 0.5,
        regions_cmap: Optional[Union[str, LinearSegmentedColormap]] = "viridis",
        colorbar: bool = False,
        **kwargs: Any
    ) -> "DecisionBoundaryEdgeDisplay":
        """Plot the decision boundary edges.
        
        Parameters:
            fill_regions: bool
                Whether to fill the decision regions with colors.
            boundary_color: str
                The color for the boundary lines.
            boundary_width: float
                The line width for the boundary.
            boundary_alpha: float
                The alpha (transparency) value for the boundary lines.
            regions_alpha: float
                The alpha blending value for filled regions.
            regions_cmap: Optional[Union[str, LinearSegmentedColormap]]
                The colormap to use for regions if filled. If None and 'cmap' is in kwargs,
                the kwargs['cmap'] will be used.
            colorbar: bool
                Whether to display a colorbar for the filled regions.
            **kwargs: Any
                Additional keyword arguments passed to the plotting method.
                
        Returns:
            DecisionBoundaryEdgeDisplay: self
        """
        # Optionally show filled decision regions
        if fill_regions:
            # Create a copy of kwargs for pcolormesh
            plot_kwargs = kwargs.copy()
            
            # Only use regions_cmap if it's not None and cmap is not in kwargs
            if regions_cmap is not None and 'cmap' not in plot_kwargs:
                plot_kwargs['cmap'] = regions_cmap
            
            self.surface_ = self.ax_.pcolormesh(
                self.xx0, self.xx1, self.response, 
                alpha=regions_alpha,
                **plot_kwargs
            )
            if colorbar:
                self.colorbar_ = plt.colorbar(self.surface_, ax=self.ax_)
        
        # Plot boundary mask
        boundary_regions = np.ma.masked_where(~self.boundary_mask, self.boundary_mask)
        self.boundary_surface_ = self.ax_.pcolormesh(
            self.xx0, self.xx1, boundary_regions,
            cmap=LinearSegmentedColormap.from_list("", [boundary_color, boundary_color]),
            alpha=boundary_alpha,
            linewidth=boundary_width
        )
        
        return self
        
    def plot_samples(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        markers: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        scatter_kwargs: Optional[Dict[str, Any]] = None
    ) -> "DecisionBoundaryEdgeDisplay":
        """Plot the samples used to train the classifier.
        
        Parameters:
            X: np.ndarray
                The feature data, shape (n_samples, 2).
            y: np.ndarray
                The target data, shape (n_samples,).
            markers: Optional[List[str]]
                List of markers to use for each class.
            colors: Optional[List[str]]
                List of colors to use for each class.
            scatter_kwargs: Optional[Dict[str, Any]]
                Additional arguments passed to plt.scatter.
                
        Returns:
            DecisionBoundaryEdgeDisplay: self
        """
        scatter_kwargs = scatter_kwargs or {}
        markers = markers or ["o", "s", "^", "v", "<", ">", "d", "p", "*"]
        
        classes = np.unique(y)
        for i, cls in enumerate(classes):
            mask = y == cls
            self.ax_.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[colors[i]] if colors else None,
                marker=markers[i % len(markers)],
                label=f"Class {cls}",
                **scatter_kwargs
            )
        
        self.ax_.legend()
        return self
        
    def apply_mask_from_boundaries(
        self, 
        boundaries: Union[List[np.ndarray], Dict[Any, np.ndarray]],
        invert: bool = True,
        draw: bool = True,
    ) -> "DecisionBoundaryEdgeDisplay":
        """Apply a mask to the visualization based on a set of boundary polygons.
        
        This method masks the visualization to show only points inside (or outside)
        the union of the provided boundaries.
        
        Parameters:
            boundaries: Union[List[np.ndarray], Dict[Any, np.ndarray]]
                List or dictionary of boundary polygons. Each polygon should be
                a numpy array of shape (n_points, 2) defining the boundary vertices.
            invert: bool
                If True, mask points outside the boundaries. If False, mask points
                inside the boundaries.
                
        Returns:
            DecisionBoundaryEdgeDisplay: self
            
        Raises:
            ImportError: If matplotlib.path.Path is not available.
            ValueError: If no surface has been created yet.
        """
        try:
            from matplotlib.path import Path
        except ImportError:
            raise ImportError("matplotlib.path.Path is required for boundary masking.")
        
        if self.boundary_surface_ is None:
            raise ValueError("No boundary surface to mask. Call plot() first.")
            
        # Convert dictionary to list if needed
        boundary_list = list(boundaries.values()) if isinstance(boundaries, dict) else boundaries
        
        # Get mesh points
        xv, yv = np.meshgrid(
            np.linspace(self.ax_.get_xlim()[0], self.ax_.get_xlim()[1], self.xx0.shape[1]),
            np.linspace(self.ax_.get_ylim()[0], self.ax_.get_ylim()[1], self.xx0.shape[0])
        )
        points = np.column_stack((xv.ravel(), yv.ravel()))
        
        # Create combined mask using all boundaries
        combined_mask = np.zeros(points.shape[0], dtype=bool)
        for boundary in boundary_list:
            path = Path(boundary)
            combined_mask |= path.contains_points(points)
            
        # Invert mask if requested
        if invert:
            combined_mask = ~combined_mask
            
        # Reshape mask to match surface array
        mask_shape = self.response.shape
        mask = combined_mask.reshape(mask_shape)
        
        # Apply mask to surfaces
        if hasattr(self.boundary_surface_, 'set_array'):
            # Mask boundary surface
            current_array = self.boundary_surface_.get_array()
            self.boundary_surface_.set_array(np.ma.array(current_array, mask=mask))
            
            # Mask regions surface if it exists
            if self.surface_ is not None and hasattr(self.surface_, 'set_array'):
                current_region_array = self.surface_.get_array()
                self.surface_.set_array(np.ma.array(current_region_array, mask=mask))
        
        # Draw boundaries if requested
        if draw:
            for boundary in boundary_list:
                self.ax_.plot(*boundary.T, c="k", lw=0.5)
            
        return self


def from_estimator(
    estimator: BaseEstimator,
    X: np.ndarray,
    grid_resolution: int = 200,
    eps: float = 0.01,
    ax: Optional[plt.Axes] = None,
    detection_method: Literal["difference", "gradient", "sobel"] = "difference",
    fill_regions: bool = False,
    boundary_color: str = "black",
    boundary_width: float = 1.0,
    boundary_alpha: float = 1.0,
    regions_cmap: Union[str, LinearSegmentedColormap, Dict[Any, str]] = "viridis",
    regions_alpha: float = 0.5,
    color_map: Optional[Dict[Any, str]] = None,  # Direct mapping from class labels to colors
    **kwargs: Any
) -> 'DecisionBoundaryEdgeDisplay':
    """Create a DecisionBoundaryEdgeDisplay from a fitted estimator.
    
    This function creates a 2D visualization that highlights the decision boundaries
    between different classes, showing exactly where the classifier's prediction changes.
    
    Parameters:
        estimator: BaseEstimator
            Fitted classifier.
        X: np.ndarray
            Input data used to fit the estimator and to compute the grid.
        grid_resolution: int
            Number of points to use for each grid dimension.
        eps: float
            Extends the range of the grid to avoid boundary effects.
        ax: Optional[plt.Axes]
            Axes to plot on, creates new axes if None.
        detection_method: Literal["difference", "gradient", "sobel"]
            Method to detect class boundaries:
            - "difference": Detects changes between adjacent pixels
            - "gradient": Uses gradient magnitude of the prediction
            - "sobel": Uses Sobel filter for edge detection
        fill_regions: bool
            Whether to show filled decision regions.
        boundary_color: str
            The color for the boundary lines.
        boundary_width: float
            The line width for the boundary.
        boundary_alpha: float
            The alpha (transparency) value for the boundary lines.
        regions_cmap: Union[str, LinearSegmentedColormap, Dict[Any, str]]
            The colormap to use for decision regions if filled. Can be a string (matplotlib 
            colormap name), a LinearSegmentedColormap object, or a dictionary mapping class 
            labels directly to colors.
        regions_alpha: float
            The alpha blending value for decision regions if filled.
        color_map: Optional[Dict[Any, str]]
            Direct mapping from class labels to colors. If provided, this overrides regions_cmap.
            Example: {0: 'blue', 1: 'red', 2: 'green'}
        **kwargs: Any
            Additional arguments passed to the plotting method.
            
    Returns:
        DecisionBoundaryEdgeDisplay: The configured display object.
        
    Raises:
        ValueError: If X is not 2D.
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"Expected 2 features, got {X.shape[1]}. DecisionBoundaryEdgeDisplay only supports 2D visualization."
        )
    
    # Allow regions_cmap to be a color_map dictionary for backward compatibility
    if isinstance(regions_cmap, dict) and color_map is None:
        color_map = regions_cmap
        regions_cmap = None  # Clear to avoid confusion
        
    # Create the grid
    x0_min, x0_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    x1_min, x1_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution)
    )
    
    # Get predictions for each grid point
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]
    y_pred_raw = estimator.predict(X_grid)
    
    # Reshape the raw predictions to the grid shape
    y_pred = y_pred_raw.reshape(xx0.shape)
    
    # For boundary detection, we need numeric values
    if hasattr(estimator, "classes_"):
        # Create a mapping from class labels to indices for boundary detection
        class_to_index = {cls: idx for idx, cls in enumerate(estimator.classes_)}
        # Create a detection array with numeric indices for boundary detection
        detection_array = np.zeros_like(y_pred, dtype=int)
        for i, val in enumerate(y_pred_raw):
            detection_array.flat[i] = class_to_index[val]
    else:
        # For non-classifiers or numeric classes, use predictions directly
        detection_array = y_pred
    
    # Create boundary mask based on chosen method
    if detection_method == "difference":
        # Detect changes in class prediction
        horizontal_diff = np.diff(detection_array, axis=1)
        vertical_diff = np.diff(detection_array, axis=0)
        
        # Initialize boundary mask
        boundary_mask = np.zeros_like(detection_array, dtype=bool)
        
        # Mark horizontal boundaries
        boundary_mask[:, :-1] |= (horizontal_diff != 0)
        boundary_mask[:, 1:] |= (horizontal_diff != 0)
        
        # Mark vertical boundaries
        boundary_mask[:-1, :] |= (vertical_diff != 0)
        boundary_mask[1:, :] |= (vertical_diff != 0)
        
    elif detection_method == "gradient":
        # Use gradient magnitude to detect boundaries
        gradient_y, gradient_x = np.gradient(detection_array.astype(float))
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold to get boundary mask
        boundary_mask = gradient_magnitude > 0
        
    elif detection_method == "sobel":
        # Use Sobel filter for edge detection
        sobel_h = ndimage.sobel(detection_array.astype(float), axis=0)
        sobel_v = ndimage.sobel(detection_array.astype(float), axis=1)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Threshold to get boundary mask
        boundary_mask = magnitude > 0
    else:
        raise ValueError(
            f"Detection method {detection_method} not supported. "
            "Use 'difference', 'gradient', or 'sobel'."
        )
    
    # Handle the color mapping based on its type
    plot_kwargs = kwargs.copy()
    
    if color_map is not None:
        # We have a direct mapping from class labels to colors
        
        # Get all possible classes from the estimator
        if hasattr(estimator, "classes_"):
            all_classes = estimator.classes_
        else:
            # For non-classifiers, get unique values from predictions
            all_classes = np.unique(y_pred)
        
        # Decide whether to use the original class values or indices for pcolormesh
        # For non-numeric classes or if there's a gap in numeric classes, use indices
        use_indices = not np.issubdtype(type(all_classes[0]), np.number) or \
                      not np.array_equal(all_classes, np.arange(len(all_classes)))
        
        if use_indices:
            # Create a new mapping from class labels to indices
            # This ensures consistent ordering with the colormap
            label_to_index = {label: i for i, label in enumerate(all_classes)}
            
            # Create a new numeric array for pcolormesh
            # This maps each prediction to its index in the colormap
            y_pred_indices = np.zeros_like(y_pred, dtype=int)
            for i, val in enumerate(y_pred_raw):
                y_pred_indices.flat[i] = label_to_index[val]
            
            # Use these indices for visualization
            y_pred_display = y_pred_indices
            
            # Create a colormap from the color_map that matches these indices
            # Order the colors by the index assigned to each class
            colors = [color_map[cls] for cls in all_classes]
            custom_cmap = ListedColormap(colors)
            plot_kwargs["cmap"] = custom_cmap
        else:
            # Classes are numeric and consecutive - we can use them directly
            # But we need to create a colormap that maps each class to the right color
            y_pred_display = y_pred
            
            # Create a boundary norm to ensure correct mapping
            bounds = np.array(list(all_classes) + [max(all_classes) + 1]) - 0.5
            norm = BoundaryNorm(bounds, len(all_classes))
            plot_kwargs["norm"] = norm
            
            # Create a colormap from the color_map
            colors = [color_map[cls] for cls in all_classes]
            custom_cmap = ListedColormap(colors)
            plot_kwargs["cmap"] = custom_cmap
    else:
        # No color_map provided, use the regions_cmap
        y_pred_display = detection_array if hasattr(estimator, "classes_") else y_pred
        # Let the plot method handle regions_cmap
    
    # Create the display object
    display = DecisionBoundaryEdgeDisplay(
        xx0=xx0,
        xx1=xx1,
        response=y_pred_display,  # Use the prepared response array
        boundary_mask=boundary_mask,
        estimator=estimator,
        ax=ax
    )
    
    # Plot the boundaries - Pass regions_cmap to plot() but don't include it in plot_kwargs
    # The plot() method will handle possible conflicts
    display.plot(
        fill_regions=fill_regions,
        boundary_color=boundary_color,
        boundary_width=boundary_width,
        boundary_alpha=boundary_alpha,
        regions_alpha=regions_alpha,
        regions_cmap=None if 'cmap' in plot_kwargs else regions_cmap,  # Only use regions_cmap if cmap not in plot_kwargs
        **plot_kwargs
    )
    
    return display

def load_ccf_boundaries():
    root_path = os.environ["ROOT_DATA_PATH"]

    ccf_files_path = os.path.join(root_path, "CCF_files")

    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file=    os.path.join(ccf_files_path,"flatmap_butterfly.nrrd"),
        labels_file=    os.path.join(ccf_files_path,"labelDescription_ITKSNAPColor.txt"),
    )

    bf_left_boundaries_flat = bf_boundary_finder.region_boundaries()
    return bf_left_boundaries_flat


def create_decision_boundary_plot_with_density_mask(
    model: SVC,
    X: np.ndarray,
    color_map: Dict[Any, str],
    ax: Optional[plt.Axes] = None,
    grid_resolution: int = 200,
    density_bandwidth: float = 0.1,
    density_threshold: float = 0.05,
    density_mask_alpha: float = 0.9,
    apply_ccf_mask: bool = True,
    smooth_density: bool = True,
    smooth_sigma: float = 1.0,
    batch_size: int = 10000,
    subsample: int = 1
) -> plt.Axes:
    """
    Create decision boundary plot with density-based masking using cuML with batched processing.
    
    Parameters
    ----------
    model : SVC
        Fitted SVC model to visualize decision boundaries for
    X : np.ndarray
        Input data used for visualization, shape (n_samples, 2)
    color_map : Dict[Any, str]
        Mapping from class labels to colors
    ax : Optional[plt.Axes]
        Matplotlib axes to plot on. If None, creates new axes
    grid_resolution : int
        Resolution of the grid for decision boundary visualization
    density_bandwidth : float
        Bandwidth parameter for kernel density estimation
    density_threshold : float
        Threshold for density masking. Areas with density below this value will be masked
    density_mask_alpha : float
        Alpha value for the density mask (1.0 = fully opaque, 0.0 = fully transparent)
    apply_ccf_mask : bool
        Whether to apply CCF boundaries masking
    smooth_density : bool
        Whether to apply smoothing to the density estimate
    smooth_sigma : float
        Sigma parameter for smoothing
    batch_size : int
        Number of points to process at once for KDE to avoid CUDA memory errors
    subsample : int
        Subsampling factor for KDE fitting (use X[::subsample])
    
    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot
    """
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Create initial decision boundary plot
    disp = from_estimator(
        estimator=model,
        X=X,
        grid_resolution=grid_resolution,
        detection_method="difference",
        fill_regions=True,
        color_map=color_map,  # Pass the original color_map dictionary directly
        regions_cmap=None,    # Set to None to avoid conflicts
        regions_alpha=1.0,
        boundary_color="black",
        boundary_width=0.5,
        boundary_alpha=0.8,
        ax=ax
    )
    
    # Apply CCF boundaries masking if requested
    if apply_ccf_mask:
        try:
            bf_left_boundaries_flat = load_ccf_boundaries()
            disp.apply_mask_from_boundaries(
                bf_left_boundaries_flat,
                invert=True,
                draw=False
            )
        except Exception as e:
            logger.warning(f"Failed to apply CCF mask: {e}")
    
    # Set equal aspect ratio and style
    disp.ax_.set_aspect('equal')
    disp.ax_.axis('off')
    disp.ax_.set_ylim(disp.ax_.get_ylim()[::-1])
    
    # Create density-based mask using cuML KernelDensity with batching
    xx0, xx1 = disp.xx0, disp.xx1
    x0_flat, x1_flat = xx0.flatten(), xx1.flatten()
    grid_points = np.vstack([x0_flat, x1_flat]).T
    
    # Compute density estimate using cuML's KernelDensity with batching
    from cuml.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=density_bandwidth, kernel='gaussian')
    kde.fit(X[::subsample])
    
    # Process grid points in batches to avoid CUDA memory errors
    n_samples = grid_points.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    # Pre-allocate the result array
    log_density = np.zeros(n_samples)
    
    # Process each batch
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch = grid_points[start_idx:end_idx]
        batch_log_density = kde.score_samples(batch)
        
        # Convert CuPy array to NumPy if needed
        if hasattr(batch_log_density, 'get'):
            batch_log_density = batch_log_density.get()
            
        log_density[start_idx:end_idx] = batch_log_density
    
    # Convert log density to density
    density = np.exp(log_density)
    
    # Reshape density to match grid shape
    density_map = density.reshape(xx0.shape)
    
    # Apply smoothing if requested
    if smooth_density:
        try:
            # Try to use cupy-based smoothing if available
            import cupy as cp
            import cupyx.scipy.ndimage as cuimg
            
            # Convert to cupy array
            density_cp = cp.asarray(density_map)
            
            # Apply Gaussian filter on GPU
            density_cp = cuimg.gaussian_filter(density_cp, sigma=smooth_sigma)
            
            # Convert back to numpy
            density_map = cp.asnumpy(density_cp)
        except (ImportError, ModuleNotFoundError):
            # Fallback to scipy if cupyx is not available
            from scipy import ndimage
            density_map = ndimage.gaussian_filter(density_map, sigma=smooth_sigma)
    
    # Normalize density to [0, 1] range for easier thresholding
    density_min = density_map.min()
    density_max = density_map.max()
    if density_max > density_min:  # Avoid division by zero
        density_map = (density_map - density_min) / (density_max - density_min)
    
    # Create density mask
    density_mask = density_map < density_threshold
    
    # Create white background for masked regions
    white_background = np.ones_like(disp.response)
    masked_background = np.ma.array(white_background, mask=~density_mask)
    
    # Plot white background for low-density regions
    disp.ax_.pcolormesh(
        xx0, xx1, masked_background,
        cmap=LinearSegmentedColormap.from_list("", ["white", "white"]),
        alpha=density_mask_alpha,
        zorder=10  # Ensure this is drawn on top
    )
    
    return disp.ax_


def create_decision_boundary_plot(
    model: SVC,
    X: np.ndarray,
    color_map: Dict[Any, str],
    ax: Optional[plt.Axes] = None,
    grid_resolution: int = 200
) -> plt.Axes:
    """
    Create decision boundary plot for a single model.
    
    Parameters:
        model: SVC
            Fitted SVC model to visualize decision boundaries for
        X: np.ndarray
            Input data used for visualization
        color_map: Dict[Any, str]
            Mapping from class labels to colors
        ax: Optional[plt.Axes]
            Matplotlib axes to plot on. If None, creates new axes
        grid_resolution: int
            Resolution of the grid for decision boundary visualization
            
    Returns:
        plt.Axes: The matplotlib axes containing the plot
    """
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    bf_left_boundaries_flat = load_ccf_boundaries()

    # Pass the color_map directly to from_estimator
    disp = from_estimator(
        estimator=model,
        X=X,
        grid_resolution=grid_resolution,
        detection_method="difference",
        fill_regions=True,
        color_map=color_map,  # Pass the original color_map dictionary
        regions_cmap=None,    # Set to None to avoid conflicts
        regions_alpha=1.0,
        boundary_color="black",
        boundary_width=0.5,
        boundary_alpha=0.8,
        ax=ax
    )

    # Apply masking using boundaries
    disp.apply_mask_from_boundaries(
        bf_left_boundaries_flat,
        invert=True,
        draw=False
    )

    # Set equal aspect ratio and style
    disp.ax_.set_aspect('equal')
    disp.ax_.axis('off')
    disp.ax_.set_ylim(disp.ax_.get_ylim()[::-1])
    
    return disp.ax_


def plot_scatter_style(x, y, labels, preds, color_map, bf_left_boundaries_flat, alpha=0.5):
    """Plot the scatter-style visualization with three panels"""
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Labels
    colors_labels = np.array([color_map[label] for label in labels])
    ax[0].scatter(x, y, color=colors_labels, alpha=alpha, linewidths=0, s=5)
    ax[0].set_title("Area labels")
    
    # Panel 2: Predictions
    colors_preds = np.array([color_map[pred] for pred in preds])
    ax[1].scatter(x, y, color=colors_preds, alpha=alpha, linewidths=0, s=5)
    ax[1].set_title("Area predictions")
    
    # Panel 3: Errors
    misclassified = labels != preds
    ax[2].scatter(x[misclassified], y[misclassified], color='r', alpha=alpha, linewidths=0, s=5)
    ax[2].set_title("Errors")
    
    # Common styling for all panels
    for a in ax:
        a.grid(False)
        for k, boundary_coords in bf_left_boundaries_flat.items():
            a.plot(*boundary_coords.T, c="k", lw=0.5)
        a.axis('off')
        a.set_aspect('equal')
        a.set_ylim(a.get_ylim()[::-1])
    
    plt.tight_layout()
    return f, ax

def create_master_colormap(adata_list, area_class2area_name):
    all_categories = set()
    # Get unique categories from all datasets
    for adata in adata_list:
        all_categories.update(adata.obs['area_name'].unique())
            # Using area_class2area_name for label names since it's our mapping dictionary
        label_names = area_class2area_name
    # Convert to sorted list for consistent ordering
    all_categories = sorted(list(all_categories))
    
    # Create extended colormap by combining multiple colorschemes
    colormaps = ['tab20', 'tab20b', 'tab20c']
    colors = np.vstack([plt.cm.get_cmap(cmap)(np.linspace(0, 1, 20)) for cmap in colormaps])
    
    # Map categories to colors
    color_indices = np.arange(len(all_categories)) % len(colors)
    color_map = dict(zip(all_categories, colors[color_indices]))

    area_class2area_name_reversed = {v:k for k,v in area_class2area_name.items()}
    color_map = {area_class2area_name_reversed[k]:v for k,v in color_map.items()}
    
    return color_map, label_names

import sys
import io
import re

# For convenience, also create a context manager version
class SuppressOutput:
    """
    Context manager to filter output in Jupyter notebooks.
    
    Specifically designed to suppress CUML warnings regarding 'working set'.
    
    Examples:
        >>> with SuppressOutput(pattern="[CUML] [warning]"):
        ...     model.fit(X, y)
        
        >>> # With custom pattern
        >>> with SuppressOutput(pattern="some specific error to filter"):
        ...     my_function()
    """
    
    def __init__(self, pattern: str = r"\[CUML\] \[warning\] Warning: could not fill working set"):
        """
        Initialize with pattern to suppress.
        
        Args:
            pattern: Regular expression pattern to match against output lines
        """
        self.pattern = re.compile(pattern)
        self.original_stdout = None
        self.original_stderr = None
        self.captured_stdout = None
        self.captured_stderr = None
    
    def __enter__(self):
        """Redirect stdout and stderr to capture output."""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.captured_stdout = io.StringIO()
        self.captured_stderr = io.StringIO()
        sys.stdout = self.captured_stdout
        sys.stderr = self.captured_stderr
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout/stderr and filter captured output."""
        # Get captured content
        stdout_content = self.captured_stdout.getvalue()
        stderr_content = self.captured_stderr.getvalue()
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Filter and write output
        for line in stdout_content.splitlines(True):
            if not self.pattern.search(line):
                self.original_stdout.write(line)
                
        for line in stderr_content.splitlines(True):
            if not self.pattern.search(line):
                self.original_stderr.write(line)