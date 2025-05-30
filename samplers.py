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
    successful_groups = 0
    attempts = 0
    max_attempts = num_groups * 2  # Allow some retries
    
    while successful_groups < num_groups and attempts < max_attempts:
        center_idx = valid_indices[attempts % len(valid_indices)]
        group = sampler._get_spatial_group(center_idx)
        attempts += 1
        
        if group is None:
            continue
            
        all_indices.extend(group)
        
        # Plot group points with unique color
        group_coords = sampler.precomputed.coordinates[group]
        color = colors[successful_groups % len(colors)]
        
        # Plot hex center
        center = hex_centers[center_idx]
        ax2.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            c=[color],
            s=5,
            alpha=0.7
        )
        ax2.scatter(
            [center[0]],
            [center[1]],
            c=[color],
            marker='*',
            s=100,
            edgecolor='black'
        )
        
        seen_points.update(group)
        successful_groups += 1
    
    if successful_groups < num_groups:
        print(f"Warning: Could only visualize {successful_groups}/{num_groups} groups")
    
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
        add_jitter: bool = True,
        hex_scaling: Optional[float] = 1.2,
        reflect_points: bool = True,
        hex_grid: Optional[HexGridData] = None, 
        max_radius_expansions: int = 2,
        group_within_keys: Optional[List[str]] = None,
        iterate_all_points: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.coordinate_key = coordinate_key
        self.seed = seed
        self.epoch = 0
        self.add_jitter = add_jitter
        self.hex_scaling = hex_scaling
        self.reflect_points = reflect_points
        self.hex_grid = hex_grid  # Store provided hex grid
        self.max_radius_expansions = max_radius_expansions
        self.group_within_keys = group_within_keys
        if self.group_within_keys is not None:
            if isinstance(self.group_within_keys, str):
                self.group_within_keys = [self.group_within_keys]
        
        self.num_samples = len(dataset)
        self.rng = np.random.RandomState(seed)
        
        if precomputed is not None:
            self.precomputed = precomputed
        else:
            self.precomputed = self._precompute_spatial_data()

    def _maybe_subset_group_by_keys(self, group_indices: np.ndarray) -> np.ndarray:
        """
        If group_within_keys is specified, subset the group to only include indices
        that share the same value for the specified key.
        """
        if self.group_within_keys is None:
            return group_indices
        
        for key in self.group_within_keys:
            first_key_value = self.dataset[int(group_indices[0])][key]
            matching_indices = [
                idx for idx in group_indices 
                if (self.dataset[int(idx)][key] == first_key_value)
            ]
            group_indices = matching_indices

        return np.array(matching_indices)

    def _estimate_hex_size(self, coordinates: np.ndarray) -> float:
        """Estimate appropriate hexagon size based on point density and target count or group size."""
            
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        area = x_range * y_range
        
        target_hex_count = len(coordinates) / self.group_size
            
        hex_area = area / target_hex_count * self.hex_scaling  # Slightly larger than needed b/c of anisotropy
                
        print(f"Estimating hex size:")
        print(f"  Total points: {len(coordinates)}")
        print(f"  Target hex count: {target_hex_count:.0f}")
        print(f"  Target cells per hex: {self.group_size * 1.5}")
        
        # Calculate hex size to achieve this coverage
        hex_size = np.sqrt(hex_area / (2 * np.sqrt(3)))
        
        print(f"  Estimated hex size: {hex_size:.2f}")

        return hex_size

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
        
        print(f"Hex grid created:")
        print(f"  Total hexagons: {len(hex_centers)}")
        print(f"  Valid hexagons: {len(valid_hex_indices)}")
        print(f"  Base search radius: {search_radius:.2f}")
        print(f"  Mean points within max radius: {points_per_hex[valid_hex_indices].mean():.1f}")
        
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

        if self.reflect_points:
            coordinates = reflect_points_to_left(coordinates)
        
        # Build spatial index
        tree = cKDTree(coordinates)
        x_range = np.ptp(coordinates[:, 0])
        y_range = np.ptp(coordinates[:, 1])
        
        # Use provided hex grid or create new one
        if self.hex_grid is not None:
            hex_grid = self.hex_grid
            hex_size = hex_grid.hex_size
            print("Using provided hex grid")
        else:
            hex_size = self._estimate_hex_size(coordinates)
            hex_grid = self._create_hex_grid(coordinates, hex_size, tree)
            print(f"Created new hex grid with size {hex_size:.4f}")
        
        initial_radius = hex_size * 1.2
        
        print(f"Grid has {len(hex_grid.valid_hex_indices)} valid centers")
        
        return PrecomputedData(
            coordinates=coordinates,
            tree=tree,
            x_range=x_range,
            y_range=y_range,
            initial_radius=initial_radius,
            hex_grid=hex_grid
        )

    def _get_spatial_group(self, center_idx: int) -> Optional[np.ndarray]:
        """Get indices for one spatial group using hex center with random jitter.
        Returns None if insufficient points found within max radius expansions."""
        base_center = self.precomputed.hex_grid.hex_centers[center_idx]
        hex_size = self.precomputed.hex_grid.hex_size
        
        if self.add_jitter:
            jitter = self.rng.normal(0, 0.3 * hex_size, size=2)
            center = base_center + jitter
        else:
            center = base_center
        
        radius = self.precomputed.initial_radius
        expansions = 0
        
        while expansions < self.max_radius_expansions:
            neighbor_indices = self.precomputed.tree.query_ball_point(
                center, radius, workers=-1
            )
            # Subset group by keys if necessary
            neighbor_indices = self._maybe_subset_group_by_keys(neighbor_indices)
   
            if len(neighbor_indices) >= self.group_size:
                neighbor_coords = self.precomputed.coordinates[neighbor_indices]
                distances = np.sum((neighbor_coords - center) ** 2, axis=1)
                
                k = min(self.group_size, len(distances))
                closest_local_indices = np.argpartition(distances, k-1)[:k]
                selected_indices = np.array(neighbor_indices)[closest_local_indices]
                
                if len(selected_indices) >= self.group_size:
                    return selected_indices[:self.group_size]
            
            radius *= 2
            expansions += 1
        
        # If we get here, we couldn't find enough points within max radius
        return None

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
        if len(valid_centers) == 0:
            raise ValueError("No valid hex centers found with sufficient points")
            
        permuted_centers = self.rng.permutation(valid_centers)
        
        all_indices = []
        
        # Handle special case of group_size=1
        if self.group_size == 1:
            return iter(range(self.num_samples))
        
        # Keep track of centers we skipped due to insufficient points
        skipped_centers = []
        
        # Iterate through permuted centers
        for center_idx in permuted_centers:
            if len(all_indices) >= self.num_samples:
                break
                
            group = self._get_spatial_group(center_idx)
            if group is None:
                skipped_centers.append(center_idx)
                continue
                
            remaining_space = self.num_samples - len(all_indices)
            if remaining_space >= self.group_size:
                all_indices.extend(group)
            else:
                all_indices.extend(group[:remaining_space])
                break
        
        # If we skipped any centers, log it
        if skipped_centers:
            print(f"Skipped {len(skipped_centers)} centers due to insufficient points in test set")
        
        # If we still need more points, try again with remaining centers
        while len(all_indices) < self.num_samples:
            remaining_centers = [idx for idx in permuted_centers if idx not in skipped_centers]
            if not remaining_centers:
                raise RuntimeError(f"No valid centers remain. Got {len(all_indices)}/{self.num_samples} samples")
            
            permuted_centers = self.rng.permutation(remaining_centers)
            for center_idx in permuted_centers:
                if len(all_indices) >= self.num_samples:
                    break
                    
                group = self._get_spatial_group(center_idx)
                if group is None:
                    skipped_centers.append(center_idx)
                    continue
                    
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

class DistributedHexagonalSpatialGroupSampler(HexagonalSpatialGroupSampler):
    """
    Distributed version of HexagonalSpatialGroupSampler for multi-GPU training.
    Inherits hex grid sampling behavior and adds distribution across processes.
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
        precomputed: Optional[PrecomputedData] = None,
        add_jitter: bool = True,
        hex_scaling: Optional[float] = 1.2,
        reflect_points: bool = True,
        hex_grid: Optional[HexGridData] = None,
        max_radius_expansions: int = 2,
        group_within_keys: Optional[List[str]] = None,
        iterate_all_points: bool = False,
    ):
        # Initialize distributed parameters
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        
        # Initialize base hex sampler
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            group_size=group_size,
            coordinate_key=coordinate_key,
            seed=seed,
            precomputed=precomputed,
            add_jitter=add_jitter,
            hex_scaling=hex_scaling,
            reflect_points=reflect_points,
            hex_grid=hex_grid,
            max_radius_expansions=max_radius_expansions,
            group_within_keys=group_within_keys
        )
        
        # Calculate number of samples for this process
        if self.drop_last and len(dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(dataset) / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Returns distributed iterator of indices where spatial groups are kept together."""
        # Set RNG state
        self.rng = np.random.RandomState(self.seed + self.epoch)
        
        # Handle special case of group_size=1
        if self.group_size == 1:
            indices = list(range(len(self.dataset)))
            if len(indices) < self.total_size:
                indices = indices * (self.total_size // len(indices) + 1)
            indices = indices[:self.total_size]
            return iter(indices[self.rank:self.total_size:self.num_replicas])
            
        # Get valid centers and shuffle them
        valid_centers = self.precomputed.hex_grid.valid_hex_indices
        if len(valid_centers) == 0:
            raise ValueError("No valid hex centers found with sufficient points")

        # Generate groups until we have enough samples
        all_indices = []
        centers_queue = list(self.rng.permutation(valid_centers))
        
        while len(all_indices) < self.total_size:
            if not centers_queue:
                centers_queue = list(self.rng.permutation(valid_centers))
            
            center_idx = centers_queue.pop(0)
            group = self._get_spatial_group(center_idx)
            
            if group is not None:
                all_indices.extend(group)
                
        # Trim or pad to exact size
        if len(all_indices) > self.total_size:
            all_indices = all_indices[:self.total_size]
        else:
            all_indices = (all_indices * (self.total_size // len(all_indices) + 1))[:self.total_size]

        # Get indices for this rank
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
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
        group_within_keys: List or None. If not None, groups are formed within these keys. e.g. within an animal or cell type
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int = 32,
        coordinate_key: str = "CCF_streamlines",
        seed: int = 0,
        precomputed: Optional[PrecomputedData] = None,
        reflect_points: bool = False,
        group_within_keys = None,
        max_radius_expansions: int = 5,
        iterate_all_points: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.coordinate_key = coordinate_key
        self.seed = seed
        self.epoch = 0
        self.reflect_points = reflect_points
        self.group_within_keys = group_within_keys    
        self.max_radius_expansions = max_radius_expansions
        self.iterate_all_points = iterate_all_points
        if self.group_within_keys is not None:
            if isinstance(self.group_within_keys, str):
                self.group_within_keys = [self.group_within_keys]
        
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

        if self.reflect_points:
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
    
    def _maybe_subset_group_by_keys(self, group_indices: np.ndarray) -> np.ndarray:
        """
        If group_within_keys is specified, subset the group to only include indices
        that share the same value for the specified key.
        """
        if self.group_within_keys is None:
            return group_indices
        
        for key in self.group_within_keys:
            first_key_value = self.dataset[int(group_indices[0])][key]
            matching_indices = [
                idx for idx in group_indices 
                if (self.dataset[int(idx)][key] == first_key_value)
            ]
            group_indices = matching_indices

        return np.array(matching_indices)

    def _get_spatial_group(self, center_idx: int) -> np.ndarray:
        """Get indices for one spatial group using KD-tree search."""
        center = self.precomputed.coordinates[center_idx]
        radius = self.precomputed.initial_radius
        
        # Ensure we don't request more points than possible
        effective_group_size = min(self.group_size, len(self.dataset) - 1)

        expansions = 0
        while expansions < self.max_radius_expansions:
            neighbor_indices = self.precomputed.tree.query_ball_point(
                center, radius, workers=-1
            )

            # Subset group by keys if necessary
            neighbor_indices = self._maybe_subset_group_by_keys(neighbor_indices)
                
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
            expansions += 1

    def __iter__(self) -> Iterator[int]:
        """
        Returns a lazy iterator that generates spatial groups on-demand.
        
        This implementation avoids precomputing all groups upfront, which improves
        startup time significantly when iterate_all_points=True.
        
        Returns:
            Iterator[int]: An iterator over dataset indices where points within
                        spatial proximity are grouped together.
        """
        # Set random seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.rng = check_random_state(self.seed + self.epoch)
        
        # Fast path for single-item groups
        if self.group_size == 1:
            yield from range(self.num_samples)
        
        if self.iterate_all_points:
            # Lazy approach: yield each group's points as they're computed
            for center_idx in range(self.num_samples):
                # Optional logging at reasonable intervals
                if center_idx % 1000 == 0:
                    print(f"Processing center point {center_idx}/{self.num_samples}")
                    
                group = self._get_spatial_group(center_idx)
                if group is not None:
                    for idx in group:
                        yield idx
        else:
            # For random sampling mode:
            # We'll generate groups on-demand and yield their indices
            all_indices = []
            skipped_centers = []
            n_groups_made = 0
            num_complete_groups = self.num_samples // self.group_size
            
            while n_groups_made < num_complete_groups:
                if len(skipped_centers) > (5 * num_complete_groups):
                    raise RuntimeError(
                        f"Unable to find enough points for complete groups. "
                        f"Only found {n_groups_made} groups. Increase max_radius_expansions, "
                        f"reduce group_size, or increase dataset size."
                    )
                    
                center_idx = self.rng.randint(0, len(self.dataset))
                group = self._get_spatial_group(center_idx)
                
                if group is None:
                    skipped_centers.append(center_idx)
                    continue
                    
                for idx in group:
                    yield idx
                    
                n_groups_made += 1
                
            if skipped_centers:
                print(f"Skipped {len(skipped_centers)} centers due to insufficient points with max radius")

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
        precomputed: Optional[PrecomputedData] = None,
        reflect_points=False,
        group_within_keys=None,
        max_radius_expansions= 2,
        iterate_all_points: bool = False,
        **kwargs
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
        self.reflect_points = reflect_points
        self.max_radius_expansions = max_radius_expansions
        self.iterate_all_points = iterate_all_points
        self.group_within_keys = group_within_keys
        if self.group_within_keys is not None:
            if isinstance(self.group_within_keys, str):
                self.group_within_keys = [self.group_within_keys]

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
    _maybe_subset_group_by_keys = SpatialGroupSampler._maybe_subset_group_by_keys

    def __iter__(self) -> Iterator[int]:
        """Returns distributed iterator of indices where spatial groups are kept together."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        skipped_centers = []
        num_complete_groups = self.total_size // (self.group_size * self.num_replicas)

        if self.group_size == 1:
            indices = list(range(len(self.dataset)))
            if len(indices) < self.total_size:
                indices = indices * (self.total_size // len(indices) + 1)
            indices = indices[:self.total_size]            
            return iter(indices[self.rank:self.total_size:self.num_replicas])
            
        if self.iterate_all_points:
            indices = list(range(len(self.dataset)))
            if len(indices) < self.total_size:
                indices = indices * (self.total_size // len(indices) + 1)
            indices = indices[:self.total_size]            
            for center_idx in indices[self.rank:self.total_size:self.num_replicas]:
                group = self._get_spatial_group(center_idx)
                if group is None:
                    skipped_centers.append(center_idx)
                    continue
                all_indices.extend(group) 
            return iter(all_indices)

        for _ in range(num_complete_groups):
            center_idx = self.rng.randint(0, len(self.dataset))
            group = self._get_spatial_group(center_idx)
            if group is None:
                skipped_centers.append(center_idx)
                continue
            all_indices.extend(group)

        if skipped_centers:
            print(f"Skipped {len(skipped_centers)} centers due to insufficient points in test set")
            
        remaining = (self.total_size // self.num_replicas) - len(all_indices)
        if remaining > 0 and not self.drop_last:
            center_idx = self.rng.randint(0, len(self.dataset))
            last_group = self._get_spatial_group(center_idx)
            if last_group is not None:
               all_indices.extend(last_group[:remaining])
        
        return iter(all_indices)

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

class GroupedSpatialTrainer(Trainer):
    def __init__(self, *args, add_single_cell_labels=True, 
                 spatial_group_size=32, 
                 spatial_label_key='area_labels', 
                 index_key='uuid',
                 coordinate_key='CCF_streamlines',
                 group_within_keys=None,
                 relative_positions=False,
                 absolute_Z=False,
                 additional_feature_keys=[],
                 sampling_strategy='random',
                 hex_scaling=None,
                 reflect_points=True,
                 max_radius_expansions=2,
                 use_train_hex_grid_on_eval=True,
                 visualize_hex_grid=False,
                 **kwargs):
        kwargs["data_collator"] = SpatialGroupCollator(
            group_size=spatial_group_size,
            label_key=spatial_label_key,
            feature_keys=["input_ids"] + additional_feature_keys,
            pad_token_id=0,
            add_single_cell_labels=add_single_cell_labels,
            index_key=index_key,
            coordinate_key=coordinate_key,
            relative_positions=relative_positions,
            absolute_Z=absolute_Z
        )

        # Store spatial sampling parameters
        self.spatial_group_size = spatial_group_size
        self.coordinate_key = coordinate_key
        self.hex_scaling = hex_scaling
        self.reflect_points = reflect_points
        self.max_radius_expansions = max_radius_expansions
        self.use_train_hex_grid_on_eval = use_train_hex_grid_on_eval
        self.group_within_keys = group_within_keys

        super().__init__(*args, **kwargs)

        self.sampling_strategy = sampling_strategy
        if sampling_strategy == 'hex':
            if self.args.world_size <= 1:
                self.sampler_class = HexagonalSpatialGroupSampler
            else:
                self.sampler_class = DistributedHexagonalSpatialGroupSampler
        elif sampling_strategy == 'random':
            if self.args.world_size <= 1:
                self.sampler_class = SpatialGroupSampler
            else:
                self.sampler_class = DistributedSpatialGroupSampler
        else:
            raise ValueError(f"Invalid group strategy: {sampling_strategy}. Use 'hex' or 'random'.")

        self.precomputed_eval_sampler_data = None
        # Initialize training sampler first to get valid hexagons if needed
        if sampling_strategy == 'hex':
            train_sampler = self._get_train_sampler()
            if visualize_hex_grid:
                train_sampler.visualize(f"hex_grid_sampling_gs_{spatial_group_size}.png", num_groups=None)
            
            # Store hex grid for use in eval/test
            self.hex_grid = train_sampler.precomputed.hex_grid
            
        # Now initialize eval sampler with the same hex grid
        eval_sampler = self._get_eval_sampler(self.eval_dataset)
        self.precomputed_eval_sampler_data = eval_sampler.precomputed if eval_sampler is not None else None

        assert self.args.train_batch_size % spatial_group_size == 0, \
            "train_batch_size must be divisible by spatial_group_size"

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        # Randomize seed for training
        random_seed = self.args.seed + torch.initial_seed()

        return self.sampler_class(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                group_size=self.spatial_group_size,
                seed=random_seed,
                coordinate_key=self.coordinate_key,
                hex_scaling=self.hex_scaling,
                reflect_points=self.reflect_points,
                group_within_keys=self.group_within_keys,
            )
        

    def _get_eval_sampler(self, eval_dataset, precomputed=True, add_jitter=True, eval_seed=42, iterate_all_points=False) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(eval_dataset, collections.abc.Sized):
            return None

        return self.sampler_class(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                group_size=self.spatial_group_size,
                seed=eval_seed,
                precomputed=self.precomputed_eval_sampler_data if precomputed else None,
                coordinate_key=self.coordinate_key,
                add_jitter=add_jitter,
                hex_scaling=self.hex_scaling,
                reflect_points=self.reflect_points,
                hex_grid=self.hex_grid if (hasattr(self, 'hex_grid') and self.use_train_hex_grid_on_eval) else None,
                max_radius_expansions=self.max_radius_expansions,
                group_within_keys=self.group_within_keys,
                iterate_all_points=iterate_all_points
            )
        
    def get_test_dataloader(self, test_dataset: Dataset, seed=42) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Same behavior as normal but custom in that 
         - we pass precomputed=False to _get_eval_sampler.
         - if we are using a hex grid, we do not add jitter.
         - we pass iterate_all_points=True. The behavior of this is to iterate
            through all the points in the dataset as the center_idx. 

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
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset, precomputed=False, add_jitter=True, eval_seed=seed, iterate_all_points=True)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test", seed=42
    ) -> PredictionOutput:
        """Memory-efficient predict method that tracks indices without collecting all batches."""
        self._memory_tracker.start()
        
        # Get original dataloader
        test_dataloader = self.get_test_dataloader(test_dataset, seed)
        # print dataloader stats
        print(f"Test dataloader: {test_dataloader}")
        print(f"Test dataloader sampler: {test_dataloader.sampler}")
        print(f"Test dataloader sampler dataset length: {len(test_dataloader)}")
        
        # Create an index-tracking dataloader wrapper
        index_tracking_dataloader = IndexTrackingDataLoader(test_dataloader)
        
        # Run evaluation with index tracking
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            index_tracking_dataloader, 
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
        
        # Get the collected indices in order
        ordered_indices = index_tracking_dataloader.get_collected_indices()
        
        # Subsample based on batch size if not hex strategy
        if self.sampling_strategy != 'hex':
            if isinstance(output.predictions, tuple):
                preds = tuple([p[::self.spatial_group_size] for p in output.predictions])
            else:
                preds = output.predictions[::self.spatial_group_size]

            if isinstance(output.label_ids, tuple):
                label_ids = tuple([p[::self.spatial_group_size] for p in output.label_ids])
            else:
                label_ids = output.label_ids[::self.spatial_group_size]
            ordered_indices = ordered_indices[::self.spatial_group_size]
        else:
            preds = output.predictions
            label_ids = output.label_ids
            ordered_indices = ordered_indices

        return PredictionOutput(
            predictions=preds, 
            label_ids=label_ids, 
            metrics=output.metrics
        ), ordered_indices


class IndexTrackingDataLoader:
    """A DataLoader wrapper that tracks indices without storing all batches."""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.collected_indices = []
    
    def __iter__(self):
        # Reset the collected indices at the start of iteration
        self.collected_indices = []
        
        # Create an iterator from the original dataloader
        self.dataloader_iter = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataloader_iter)
            
            # Extract and store indices, but keep them in the batch
            if isinstance(batch, dict):
                indices = batch.get('indices')
                if indices is not None:
                    if torch.is_tensor(indices):
                        self.collected_indices.append(indices.cpu().clone())  # clone to avoid reference issues
                    else:
                        self.collected_indices.append(indices)
                return batch
            else:
                raise ValueError(f"DataLoader must yield dict-style batches, got {type(batch)}")
        except StopIteration:
            raise
    
    def __len__(self):
        return len(self.dataloader)
    
    @property
    def dataset(self):
        return self.dataloader.dataset
    
    def get_collected_indices(self):
        """Return the concatenated indices collected during iteration."""
        if not self.collected_indices:
            return []
            
        if torch.is_tensor(self.collected_indices[0]):
            return torch.cat(self.collected_indices)
        elif isinstance(self.collected_indices[0], np.ndarray):
            return np.concatenate(self.collected_indices)
        else:
            return [item for batch in self.collected_indices for item in batch]