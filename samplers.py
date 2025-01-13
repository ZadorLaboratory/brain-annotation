import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset
import numpy as np
import math
from typing import Iterator, List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
from transformers import Trainer
import collections
from collections import Counter
from scipy.spatial import cKDTree
import wandb
from transformers.trainer_utils import speed_metrics, PredictionOutput, EvaluationStrategy
from transformers.trainer import is_datasets_available
import time
from torch.utils.data import DataLoader
from sklearn.utils import check_random_state

if is_datasets_available():
    import datasets

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
import os

def reflect_points_to_left(coords: np.ndarray) -> np.ndarray:
    """
    Reflects coordinates that are right of x=1176.5 to the left side.
    Points already on the left remain unchanged.

    This is a one-time thing needed for the data from Chen et al. (2020); all data should be left hemisphere.
    
    Args:
        coords: Array of shape (N, 2) containing x,y coordinates
        
    Returns:
        reflected: Array of same shape with right-side points reflected
    """
    x_line = 1176.5
    reflected = coords.copy()
    
    # Only reflect points where x > x_line
    right_side_mask = coords[:, 0] > x_line
    reflected[right_side_mask, 0] = 2 * x_line - coords[right_side_mask, 0]
    
    return reflected

def visualize_hex_grid(
    sampler,
    save_path: str,
    num_groups: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 12)
    ) -> None:
    """
    Visualize the hex grid and sampling pattern.
    
    Args:
        sampler: The HexagonalSpatialGroupSampler instance
        save_path: Path to save the visualization
        num_groups: Number of groups to visualize (default: min(10, len(valid_centers)))
        figsize: Figure size in inches
    """
    plt.figure(figsize=figsize)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Hex Grid Sampling Visualization', fontsize=16)
    
    # Plot 1: Hex Grid Structure
    ax1.set_title('Hex Grid Structure')
    
    # Plot all points
    ax1.scatter(
        sampler.precomputed.coordinates[:, 0],
        sampler.precomputed.coordinates[:, 1],
        c='lightgray',
        s=1,
        alpha=0.5,
        label='All Points'
    )
    
    # Plot hex centers and boundaries
    hex_centers = sampler.precomputed.hex_grid.hex_centers
    valid_indices = sampler.precomputed.hex_grid.valid_hex_indices
    hex_size = sampler.precomputed.hex_grid.hex_size
    points_per_hex = sampler.precomputed.hex_grid.points_per_hex
    
    # Create colormap for point density
    norm = mcolors.Normalize(
        vmin=0,
        vmax=max(points_per_hex[valid_indices])
    )
    
    # Plot hexagons
    for idx in range(len(hex_centers)):
        center = hex_centers[idx]
        color = 'red' if idx in valid_indices else 'lightgray'
        alpha = 0.7 if idx in valid_indices else 0.2
        
        # Create hexagon
        hex_patch = RegularPolygon(
            (center[0], center[1]),
            numVertices=6,
            radius=hex_size,
            orientation=15,
            facecolor=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=0.5
        )
        ax1.add_patch(hex_patch)
    
    # Plot 2: Sampling Pattern
    ax2.set_title('Sampling Pattern')
    
    # Plot all points
    ax2.scatter(
        sampler.precomputed.coordinates[:, 0],
        sampler.precomputed.coordinates[:, 1],
        c='lightgray',
        s=1,
        alpha=0.5,
        label='All Points'
    )
    
    # Get sampling pattern
    if num_groups is None:
        num_groups = len(valid_indices)
    
    # Save current RNG state
    rng_state = sampler.rng.get_state()
    
    # Get groups from sampler
    all_indices = []
    seen_points = set()
    colors = plt.cm.tab20(np.linspace(0, 1, num_groups))
    
    for center_idx in sampler.precomputed.hex_grid.valid_hex_indices[:num_groups]:
        group = sampler._get_spatial_group(center_idx)
        all_indices.extend(group)
        
        # Plot group points with unique color
        group_coords = sampler.precomputed.coordinates[group]
        color = colors[len(seen_points) % len(colors)]
        
        # Plot hex center
        center = hex_centers[center_idx]
        ax2.scatter(
            [center[0]],
            [center[1]],
            c=[color],
            marker='.',
            s=20,
            edgecolor='black'
        )
        
        seen_points.update(group)
    
    # Restore RNG state
    sampler.rng.set_state(rng_state)
    
    # Formatting
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")

@dataclass
class HexGridData:
    """Container for hexagonal grid data."""
    hex_centers: np.ndarray  # Centers of all hexagons (N x 2)
    valid_hex_indices: np.ndarray  # Indices of hexagons with enough points
    hex_size: float  # Size (radius) of hexagons
    points_per_hex: np.ndarray  # Number of points within radius of each hex center

def hex_to_pixel(hex_q: int, hex_r: int, hex_size: float) -> Tuple[float, float]:
    """Convert axial hex coordinates to pixel coordinates."""
    x = hex_size * (3/2 * hex_q)
    y = hex_size * (np.sqrt(3)/2 * hex_q + np.sqrt(3) * hex_r)
    return x, y

def pixel_to_hex(x: float, y: float, hex_size: float) -> Tuple[int, int]:
    """Convert pixel coordinates to axial hex coordinates."""
    q = (2/3 * x) / hex_size
    r = (-1/3 * x + np.sqrt(3)/3 * y) / hex_size
    return _axial_round(q, r)

def _axial_round(q: float, r: float) -> Tuple[int, int]:
    """Round floating point hex coordinates to nearest hex."""
    s = -q - r
    q_rounded = round(q)
    r_rounded = round(r)
    s_rounded = round(s)
    
    q_diff = abs(q_rounded - q)
    r_diff = abs(r_rounded - r)
    s_diff = abs(s_rounded - s)
    
    if q_diff > r_diff and q_diff > s_diff:
        q_rounded = -r_rounded - s_rounded
    elif r_diff > s_diff:
        r_rounded = -q_rounded - s_rounded
        
    return q_rounded, r_rounded

@dataclass
class PrecomputedData:
    """Extended container for precomputed spatial data using KD-tree and hex grid"""
    coordinates: np.ndarray  # Only stores 2D coordinates
    tree: cKDTree
    x_range: float
    y_range: float
    initial_radius: float
    hex_grid: HexGridData = None

class HexagonalSpatialGroupSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int = 32,
        coordinate_key: str = "CCF_streamlines",
        seed: int = 0,
        precomputed: Optional[PrecomputedData] = None,
        add_jitter: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.coordinate_key = coordinate_key
        self.seed = seed
        self.epoch = 0
        self.add_jitter = add_jitter
        
        self.num_samples = len(dataset)
        self.rng = np.random.RandomState(seed)
        
        if precomputed is not None:
            self.precomputed = precomputed
        else:
            self.precomputed = self._precompute_spatial_data()

    def _estimate_hex_size(self, coordinates: np.ndarray) -> float:
        """Estimate appropriate hexagon size based on point density and group size."""
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        area = x_range * y_range
        
        target_hex_count = len(coordinates) / (self.group_size * 1.5)
        hex_area = area / target_hex_count * 1.2 # Slightly larger than needed b/c of anisotropy
        return np.sqrt(hex_area / (2 * np.sqrt(3)))

    def _create_hex_grid(self, coordinates: np.ndarray, hex_size: float, tree: cKDTree) -> HexGridData:
        """Create hexagonal grid and identify valid centers."""
        # Calculate grid dimensions
        x_min, y_min = coordinates.min(axis=0)
        x_max, y_max = coordinates.max(axis=0)
        
        # Add padding to ensure coverage of edge points
        padding = hex_size * 2
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding
        
        # Calculate grid bounds
        q_range = int(np.ceil((x_max - x_min) / (hex_size * 3/2))) + 1
        r_range = int(np.ceil((y_max - y_min) / (hex_size * np.sqrt(3)))) + 1
        
        # Generate hex centers
        hex_centers = []
        for q in range(-q_range, q_range + 1):
            for r in range(-r_range, r_range + 1):
                x, y = hex_to_pixel(q, r, hex_size)
                x += (x_max + x_min) / 2  # Center the grid
                y += (y_max + y_min) / 2
                
                # Skip if outside bounding box with padding
                if x < x_min or x > x_max or y < y_min or y > y_max:
                    continue
                    
                hex_centers.append([x, y])
        
        hex_centers = np.array(hex_centers)
        
        # Count points near each hex center
        points_per_hex = np.zeros(len(hex_centers), dtype=int)
        search_radius = hex_size * .866  # Inscribed circle radius
        
        for i, center in enumerate(hex_centers):
            points_per_hex[i] = len(tree.query_ball_point(
                center, search_radius, workers=-1
            ))
        
        # Identify valid hex centers (those with enough nearby points)
        valid_hex_indices = np.where(points_per_hex >= self.group_size)[0]
        
        return HexGridData(
            hex_centers=hex_centers,
            valid_hex_indices=valid_hex_indices,
            hex_size=hex_size,
            points_per_hex=points_per_hex
        )

    def _precompute_spatial_data(self) -> PrecomputedData:
        """Precompute spatial data including hex grid."""
        print("Precomputing spatial data with hexagonal grid...")
        
        # Extract coordinates
        coordinates = np.array([
            self.dataset[i][self.coordinate_key][:2] 
            for i in range(len(self.dataset))
        ])

        coordinates = reflect_points_to_left(coordinates)
        
        # Build spatial index
        tree = cKDTree(coordinates)
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        
        # Create hex grid using coordinates directly
        hex_size = self._estimate_hex_size(coordinates)
        hex_grid = self._create_hex_grid(coordinates, hex_size, tree)
        
        # Estimate initial radius for point queries
        initial_radius = hex_size * 1.2  # Slightly larger than hex size
        
        print(f"Hex grid created with size {hex_size:.4f} and {len(hex_grid.valid_hex_indices)} valid centers")
        
        return PrecomputedData(
            coordinates=coordinates,
            tree=tree,
            x_range=x_range,
            y_range=y_range,
            initial_radius=initial_radius,
            hex_grid=hex_grid
        )

    def _get_spatial_group(self, center_idx: int) -> np.ndarray:
        """Get indices for one spatial group using hex center with random jitter."""
        base_center = self.precomputed.hex_grid.hex_centers[center_idx]
        hex_size = self.precomputed.hex_grid.hex_size
        
        # Add random jitter within hex (about 50% of hex size)
        jitter = self.rng.normal(0, 0.3 * hex_size, size=2)
        if self.add_jitter:
            center = base_center + jitter
        
        radius = self.precomputed.initial_radius
        while True:
            neighbor_indices = self.precomputed.tree.query_ball_point(
                center, radius, workers=-1
            )
            
            if len(neighbor_indices) >= self.group_size:
                neighbor_coords = self.precomputed.coordinates[neighbor_indices]
                distances = np.sum((neighbor_coords - center) ** 2, axis=1)
                
                k = min(self.group_size, len(distances))
                closest_local_indices = np.argpartition(distances, k-1)[:k]
                selected_indices = np.array(neighbor_indices)[closest_local_indices]
                
                if len(selected_indices) >= self.group_size:
                    return selected_indices[:self.group_size]
            
            radius *= 1.5

    def visualize(
        self,
        save_path: str,
        num_groups: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 12)
        ) -> None:
        """
        Generate and save visualization of the hex grid and sampling pattern.
        
        Args:
            save_path: Path to save the visualization
            num_groups: Number of groups to visualize
            figsize: Figure size in inches
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        visualize_hex_grid(self, save_path, num_groups, figsize)

    def __iter__(self) -> Iterator[int]:
        """Returns iterator of indices where spatial groups are kept together."""
        self.rng = np.random.RandomState(self.seed + self.epoch)
        
        # Generate random permutation of valid centers
        valid_centers = self.precomputed.hex_grid.valid_hex_indices
        permuted_centers = self.rng.permutation(valid_centers)
        
        all_indices = []
        
        # Handle special case of group_size=1
        if self.group_size == 1:
            return iter(range(self.num_samples))
        
        # Iterate through permuted centers
        for center_idx in permuted_centers:
            if len(all_indices) >= self.num_samples:
                break
                
            group = self._get_spatial_group(center_idx)
            remaining_space = self.num_samples - len(all_indices)
            
            if remaining_space >= self.group_size:
                all_indices.extend(group)
            else:
                all_indices.extend(group[:remaining_space])
                break
        
        # If we run out of centers, cycle through them again with different permutation
        while len(all_indices) < self.num_samples:
            permuted_centers = self.rng.permutation(valid_centers)
            for center_idx in permuted_centers:
                if len(all_indices) >= self.num_samples:
                    break
                    
                group = self._get_spatial_group(center_idx)
                remaining_space = self.num_samples - len(all_indices)
                
                if remaining_space >= self.group_size:
                    all_indices.extend(group)
                else:
                    all_indices.extend(group[:remaining_space])
                    break
        
        assert len(all_indices) == self.num_samples
        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

class SpatialGroupSampler(Sampler):
    """
    Sampler that groups sentences based on spatial proximity in 2D space.
    Uses KD-tree for efficient nearest neighbor search.
    
    Args:
        dataset: HuggingFace dataset containing spatial coordinates
        batch_size: Number of examples (groups) per batch
        group_size: Number of sentences per group
        coordinate_key: Key in dataset for accessing spatial coordinates
        seed: Random seed for reproducibility
        add_jitter: Does nothing, here for compatibility with HexagonalSpatialGroupSampler
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int = 32,
        coordinate_key: str = "CCF_streamlines",
        seed: int = 0,
        precomputed: Optional[PrecomputedData] = None,
        add_jitter: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.coordinate_key = coordinate_key
        self.seed = seed
        self.epoch = 0
        
        self.num_samples = len(dataset)
        self.rng = check_random_state(seed)
        
        if precomputed is not None:
            self.precomputed = precomputed
        else:
            self.precomputed = self._precompute_spatial_data()

    def _estimate_initial_radius(self, coordinates: np.ndarray, tree: cKDTree) -> float:
        """
        Estimate initial radius assuming uniform distribution in 2D.
        Uses the formula: πr² = (k/n) * total_area
        where k is group_size and n is total points.
        """
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        total_area = x_range * y_range
        
        # Solve for radius: r = sqrt((k/n * total_area) / π)
        target_area = (self.group_size / len(coordinates)) * total_area
        radius = np.sqrt(target_area / np.pi)
        
        # Add 50% margin to handle non-uniformity
        return radius * 1.5

    def _precompute_spatial_data(self) -> PrecomputedData:
        """Precompute KD-tree and spatial data for efficient neighbor search."""
        print("Precomputing spatial data...")
        
        # Extract only first two dimensions from coordinates
        coordinates = np.array([
            self.dataset[i][self.coordinate_key][:2] 
            for i in range(len(self.dataset))
        ])

        coordinates = reflect_points_to_left(coordinates)
        
        # Build KD-tree with 2D coordinates
        tree = cKDTree(coordinates)
        
        # Calculate ranges
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        
        # Automatically determine initial radius
        initial_radius = self._estimate_initial_radius(coordinates, tree)
        
        print(f"KD-tree construction complete. Initial radius: {initial_radius:.4f}")
        
        return PrecomputedData(
            coordinates=coordinates,
            tree=tree,
            x_range=x_range,
            y_range=y_range,
            initial_radius=initial_radius
        )

    def _get_spatial_group(self, center_idx: int) -> np.ndarray:
        """Get indices for one spatial group using KD-tree search."""
        center = self.precomputed.coordinates[center_idx]
        radius = self.precomputed.initial_radius
        
        # Ensure we don't request more points than possible
        effective_group_size = min(self.group_size, len(self.dataset) - 1)
        
        while True:
            neighbor_indices = self.precomputed.tree.query_ball_point(
                center, radius, workers=-1
            )
            
            if len(neighbor_indices) >= effective_group_size:
                neighbor_coords = self.precomputed.coordinates[neighbor_indices]
                distances = np.sum((neighbor_coords - center) ** 2, axis=1)
                
                # Only partition up to the number of points we actually have
                k = min(effective_group_size, len(distances))
                closest_local_indices = np.argpartition(distances, k-1)[:k]
                selected_indices = np.array(neighbor_indices)[closest_local_indices]
                
                # If we somehow still don't have enough points, expand radius
                if len(selected_indices) >= effective_group_size:
                    return selected_indices[:effective_group_size]
            
            radius *= 2

    def __iter__(self) -> Iterator[int]:
        """Returns iterator of indices where spatial groups are kept together."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        num_complete_groups = self.num_samples // self.group_size

        if self.group_size == 1:
            return iter(range(self.num_samples))
        
        for _ in range(num_complete_groups):
            center_idx = self.rng.randint(0, len(self.dataset))
            group = self._get_spatial_group(center_idx)
            all_indices.extend(group)
        
        # Handle remaining indices if any
        remaining = self.num_samples - len(all_indices)
        if remaining > 0:
            center_idx = self.rng.randint(0, len(self.dataset))
            last_group = self._get_spatial_group(center_idx)
            all_indices.extend(last_group[:remaining])
            
        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedSpatialGroupSampler(Sampler):
    """
    Distributed version of SpatialGroupSampler for multi-GPU training.
    Uses KD-tree for efficient nearest neighbor search.
    
    Args:
        dataset: HuggingFace dataset containing spatial coordinates
        batch_size: Number of examples (groups) per batch
        num_replicas: Number of training processes (defaults to world_size)
        rank: Process rank (defaults to current rank)
        group_size: Number of sentences per group
        coordinate_key: Key in dataset for accessing spatial coordinates
        seed: Random seed for reproducibility
        drop_last: Whether to drop last incomplete batch
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        group_size: int = 32,
        coordinate_key: str = "CCF_streamlines",
        precomputed: Optional[PrecomputedData] = None
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.group_size = group_size
        self.coordinate_key = coordinate_key

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        if precomputed is not None:
            self.precomputed = precomputed
        else:
            self.precomputed = self._precompute_spatial_data()
        
    # Reuse methods from SpatialGroupSampler
    _precompute_spatial_data = SpatialGroupSampler._precompute_spatial_data
    _get_spatial_group = SpatialGroupSampler._get_spatial_group
    _estimate_initial_radius = SpatialGroupSampler._estimate_initial_radius

    def __iter__(self) -> Iterator[int]:
        """Returns distributed iterator of indices where spatial groups are kept together."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        num_complete_groups = self.total_size // self.group_size
        
        for _ in range(num_complete_groups):
            center_idx = self.rng.randint(0, len(self.dataset))
            group = self._get_spatial_group(center_idx)
            all_indices.extend(group)
            
        remaining = self.total_size - len(all_indices)
        if remaining > 0 and not self.drop_last:
            center_idx = self.rng.randint(0, len(self.dataset))
            last_group = self._get_spatial_group(center_idx)
            all_indices.extend(last_group[:remaining])
        
        assert len(all_indices) == self.total_size, \
            f"Expected {self.total_size} indices but got {len(all_indices)}"
            
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

@dataclass
class SpatialGroupCollator:
    """
    Collator that handles grouping of spatially-related sequences and their labels.
    Basically, this reshapes the data appropriately to form batches of (batch_size, group_size, seq_len).
    Optionally handles attention masks and indices if present in input features.
    """
    group_size: int
    label_key: str
    feature_keys: Optional[List[str]] = None
    pad_token_id: int = 0
    padding: str = "max_length"
    add_single_cell_labels: bool = True
    index_key: Optional[str] = None
    relative_positions: bool = False  
    coordinate_key: str = "CCF_streamlines"
    absolute_Z: bool = False 
    
    def __post_init__(self):
        if self.feature_keys is None:
            self.feature_keys = ["input_ids"]
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Group features into spatial groups and aggregate their labels.
        Handles partial batches and includes attention masks and indices if present.
        """
        if not features:
            return {}
            
        # Determine which features are present in the input
        available_features = set(features[0].keys())
        feature_keys = [key for key in self.feature_keys if key in available_features]
        if "attention_mask" in available_features and "attention_mask" not in feature_keys:
            feature_keys.append("attention_mask")
            
        # Handle case where we have fewer features than group_size
        num_features = len(features)
        if num_features < self.group_size:
            padding_needed = self.group_size - num_features
            features.extend([features[-1]] * padding_needed)
            print(f"Warning: Padding batch from {num_features} to {self.group_size} features")
            
        num_groups = len(features) // self.group_size
        grouped_features = []
        
        for group_idx in range(num_groups):
            start_idx = group_idx * self.group_size
            end_idx = start_idx + self.group_size
            group = features[start_idx:end_idx]
            
            # Initialize collectors for this group
            group_dict = {key: [] for key in feature_keys}
            if self.add_single_cell_labels:
                group_dict["single_cell_labels"] = []
            if self.index_key and self.index_key in available_features:
                group_dict["indices"] = []  # Add collector for indices
            group_labels = []
            
            # Get max length for this group
            max_len = max(len(item["input_ids"]) for item in group)
            
            # Process each item in the group
            for item in group:
                # Handle features
                for key in feature_keys:
                    tensor = torch.tensor(item[key])
                    curr_len = tensor.size(0)
                    
                    if curr_len < max_len:
                        # Determine padding value based on feature type
                        pad_value = 0 if key == "attention_mask" else self.pad_token_id
                        padding = torch.full((max_len - curr_len,), pad_value, dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding])
                    
                    group_dict[key].append(tensor)
                
                # Handle label
                label = torch.tensor(item[self.label_key])
                group_labels.append(label)

                # Add single-cell labels to group
                if self.add_single_cell_labels:
                    group_dict["single_cell_labels"].append(label)
                    
                # Handle indices if present
                if self.index_key and self.index_key in item:
                    group_dict["indices"].append(torch.tensor(item[self.index_key]))

            # Add relative positions
            if self.relative_positions:
                coordinates = np.array([item[self.coordinate_key] for item in group])
                mean_position = np.mean(coordinates, axis=0, keepdims=True)
                relative_positions = coordinates - mean_position
                group_dict["relative_positions"] = torch.tensor(relative_positions, dtype=torch.float32)
            if self.absolute_Z:
                coordinates[:,:2] = coordinates[:,:2] - mean_position[:,:2] # Center around mean, but not for Z
                group_dict["relative_positions"] = torch.tensor(coordinates, dtype=torch.float32)
            
            # Stack features
            for key in feature_keys:
                group_dict[key] = torch.stack(group_dict[key])
                
            if self.add_single_cell_labels:    
                group_dict["single_cell_labels"] = torch.stack(group_dict["single_cell_labels"])
                
            # Stack indices if present
            if self.index_key and "indices" in group_dict:
                group_dict["indices"] = torch.stack(group_dict["indices"])
                        
            # Compute majority label for the group
            group_labels = torch.stack(group_labels)
            label_counts = Counter(group_labels.tolist())
            majority_label = label_counts.most_common(1)[0][0]
            group_dict[self.label_key] = torch.tensor(majority_label)
            
            grouped_features.append(group_dict)
        
        if not grouped_features:
            return {}
        
        # Combine all groups into final batch
        batch = {
            key: torch.stack([group[key] for group in grouped_features])
            for key in grouped_features[0].keys()
        }
        
        return batch

class MultiformerTrainer(Trainer):
    def __init__(self, *args, add_single_cell_labels=True, 
                 spatial_group_size=32, 
                 spatial_label_key='area_labels', 
                 index_key='uuid',
                 coordinate_key='CCF_streamlines',
                 relative_positions=False,
                 absolute_Z=False,
                 additional_feature_keys=[],
                 sampling_strategy='random',
                 **kwargs):
        kwargs["data_collator"] = SpatialGroupCollator(
            group_size=spatial_group_size,
            label_key=spatial_label_key,
            feature_keys=["input_ids"] + additional_feature_keys,  # Add other feature keys as needed
            pad_token_id=0, # Adjust as needed
            add_single_cell_labels=add_single_cell_labels,
            index_key=index_key,  # Add index tracking
            coordinate_key="CCF_streamlines",
            relative_positions=relative_positions,
            absolute_Z=absolute_Z
        )

        if sampling_strategy == 'hex':
            self.sampler_class = HexagonalSpatialGroupSampler
        elif sampling_strategy == 'random':
            self.sampler_class = SpatialGroupSampler
        else:
            raise ValueError(f"Invalid group strategy: {sampling_strategy}. Use 'hex' or 'random'.")

        # Store spatial sampling parameters
        self.spatial_group_size = spatial_group_size
        self.coordinate_key = coordinate_key
        
        super().__init__(*args, **kwargs)

        self.precomputed_eval_sampler_data = None
        eval_sampler = self._get_eval_sampler(self.eval_dataset)
        if sampling_strategy == 'hex':
            eval_sampler.visualize("hex_grid_sampling.png", num_groups=None)
        self.precomputed_eval_sampler_data = eval_sampler.precomputed if eval_sampler is not None else None

        assert self.args.train_batch_size % spatial_group_size == 0, \
            "train_batch_size must be divisible by spatial_group_size"

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        # Build the sampler.
        # Use spatial sampling
        if self.args.world_size <= 1:
            return self.sampler_class(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                group_size=self.spatial_group_size,
                seed=self.args.seed,
                coordinate_key=self.coordinate_key
            )
        else:
            return DistributedSpatialGroupSampler(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
                group_size=self.spatial_group_size,
                coordinate_key=self.coordinate_key
            )

    def _get_eval_sampler(self, eval_dataset, precomputed=True, add_jitter=True) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(eval_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        # Build the sampler.
        # Use spatial sampling for evaluation
        if self.args.world_size <= 1:
            return self.sampler_class(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                group_size=self.spatial_group_size,
                seed=self.args.seed,
                precomputed=self.precomputed_eval_sampler_data if precomputed else None,
                coordinate_key=self.coordinate_key,
                add_jitter=add_jitter,
            )
        else:
            return DistributedSpatialGroupSampler(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
                group_size=self.spatial_group_size,
                precomputed=self.precomputed_eval_sampler_data if precomputed else None,
                coordinate_key=self.coordinate_key
            )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Same behavior as normal but custom in that we pass precomputed=False to _get_eval_sampler.

        In addition, if we are using a hex grid, we do not add jitter.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset, precomputed=False, add_jitter=False)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """Modified predict method that ensures indices match prediction order."""
        self._memory_tracker.start()
        
        # Get original dataloader
        test_dataloader = self.get_test_dataloader(test_dataset)
        
        # Collect batches and indices in order
        collected_batches, ordered_indices = collect_batches_and_indices(test_dataloader)
        
        # Create new dataloader with collected batches
        ordered_dataloader = OrderedBatchDataLoader(collected_batches, test_dataloader)
        
        # Run evaluation with ordered batches
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            ordered_dataloader, 
            description="Prediction", 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix
        )
        
        # Rest of the original predict method...
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        
        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        
        return PredictionOutput(
            predictions=output.predictions, 
            label_ids=output.label_ids, 
            metrics=output.metrics
        ), ordered_indices  # Now indices are guaranteed to match prediction order
        
def collect_batches_and_indices(dataloader: DataLoader) -> Tuple[List, np.ndarray]:
    """
    Collects batches and indices in iteration order.
    
    Args:
        dataloader: Original dataloader
        
    Returns:
        Tuple of (collected_batches, collected_indices)
    """
    collected_batches = []
    collected_indices = []
    
    for batch in dataloader:
        # Extract and store indices
        if isinstance(batch, dict):
            indices = batch.pop('indices')  # Remove indices from batch
            if torch.is_tensor(indices):
                indices = indices.cpu()
            collected_indices.append(indices)
            collected_batches.append(batch)
        else:
            raise ValueError("DataLoader must yield dict-style batches")
    
    # Combine all indices
    if torch.is_tensor(collected_indices[0]):
        indices = torch.cat(collected_indices)
    elif isinstance(collected_indices[0], np.ndarray):
        indices = np.concatenate(collected_indices)
    else:
        indices = [item for batch in collected_indices for item in batch]
    
    return collected_batches, indices

class OrderedBatchDataLoader:
    """A DataLoader that yields pre-collected batches in order."""
    def __init__(self, batches, original_dataloader):
        self.batches = batches
        self.original_dataloader = original_dataloader
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)
    
    @property
    def dataset(self):
        return self.original_dataloader.dataset

