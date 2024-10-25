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

@dataclass
class PrecomputedData:
    """Container for precomputed spatial data"""
    coordinates: np.ndarray
    x_sorted: np.ndarray
    y_sorted: np.ndarray
    x_sorted_indices: np.ndarray
    y_sorted_indices: np.ndarray
    x_range: float
    y_range: float
    window_width: float
    window_height: float

class SpatialGroupSampler(Sampler):
    """
    Sampler that groups sentences based on spatial proximity in 2D space.
    
    Each "example" in the resulting batch is actually a group of sentences that are
    spatially close to each other. The groups are selected dynamically during iteration.
    
    Args:
        dataset: HuggingFace dataset containing spatial coordinates
        batch_size: Number of examples (groups) per batch
        group_size: Number of sentences per group
        coordinate_key: Key in dataset for accessing spatial coordinates
        window_size: Size of spatial window for neighbor search (relative to total space)
        seed: Random seed for reproducibility
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int = 32,
        coordinate_key: str = "CCF_streamlines",
        window_size: float = 0.1,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.coordinate_key = coordinate_key
        self.window_size = window_size
        self.seed = seed
        self.epoch = 0
        
        # Calculate sizes
        self.num_samples = len(dataset)
        self.rng = np.random.RandomState(seed)
        
        # Precompute spatial data
        self.precomputed = self._precompute_spatial_data()

    def _precompute_spatial_data(self) -> PrecomputedData:
        """Precompute spatial data structures for efficient neighbor search."""
        print("Precomputing spatial data...")
        coordinates = np.array([
            self.dataset[i][self.coordinate_key] for i in range(len(self.dataset))
        ])
        
        x_sorted_indices = np.argsort(coordinates[:, 0])
        y_sorted_indices = np.argsort(coordinates[:, 1])
        
        x_sorted = coordinates[x_sorted_indices, 0]
        y_sorted = coordinates[y_sorted_indices, 1]
        
        x_min, x_max = x_sorted[0], x_sorted[-1]
        y_min, y_max = y_sorted[0], y_sorted[-1]

        print("Indexing complete.")
        
        return PrecomputedData(
            coordinates=coordinates,
            x_sorted=x_sorted,
            y_sorted=y_sorted,
            x_sorted_indices=x_sorted_indices,
            y_sorted_indices=y_sorted_indices,
            x_range=x_max - x_min,
            y_range=y_max - y_min,
            window_width=self.window_size * (x_max - x_min),
            window_height=self.window_size * (y_max - y_min)
        )

    def _get_spatial_group(self, center_idx: int) -> np.ndarray:
        """Get indices for one spatial group. First finds neighbors in a window, then selects the closest."""
        
        center = self.precomputed.coordinates[center_idx]
        
        # Find points in window using precomputed arrays
        x_min = center[0] - self.precomputed.window_width / 2
        x_max = center[0] + self.precomputed.window_width / 2
        y_min = center[1] - self.precomputed.window_height / 2
        y_max = center[1] + self.precomputed.window_height / 2
        
        x_start = np.searchsorted(self.precomputed.x_sorted, x_min)
        x_end = np.searchsorted(self.precomputed.x_sorted, x_max, side='right')
        y_start = np.searchsorted(self.precomputed.y_sorted, y_min)
        y_end = np.searchsorted(self.precomputed.y_sorted, y_max, side='right')
        
        x_indices = self.precomputed.x_sorted_indices[x_start:x_end]
        y_indices = self.precomputed.y_sorted_indices[y_start:y_end]
        
        window_indices = np.intersect1d(x_indices, y_indices)
        
        if len(window_indices) == 0:
            return self.rng.choice(len(self.dataset), size=self.group_size, replace=False)

        assert self.group_size <= len(window_indices), "Not enough neighbors found. Increase window size."
            
        distances = np.sum((self.precomputed.coordinates[window_indices] - center) ** 2, axis=1)
        k = min(self.group_size, len(window_indices))
        nearest_idx = np.argpartition(distances, k)[:k]
        
        group = window_indices[nearest_idx]
        
        if len(group) < self.group_size:
            remaining = self.group_size - len(group)
            available = np.setdiff1d(np.arange(len(self.dataset)), group)
            pad_indices = self.rng.choice(available, size=remaining, replace=False)
            group = np.concatenate([group, pad_indices])
            
        return group

    def __iter__(self) -> Iterator[int]:
        """Returns iterator of indices where spatial groups are kept together."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Generate all indices maintaining group structure
        all_indices = []
        num_complete_groups = self.num_samples // self.group_size
        
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
        """Sets the epoch for this sampler."""
        self.epoch = epoch


class DistributedSpatialGroupSampler(Sampler):
    """
    Distributed version of SpatialGroupSampler for multi-GPU training.
    
    Handles proper sharding of data across multiple GPUs while maintaining
    the spatial grouping of sentences.
    
    Args:
        dataset: HuggingFace dataset containing spatial coordinates
        batch_size: Number of examples (groups) per batch
        num_replicas: Number of training processes (defaults to world_size)
        rank: Process rank (defaults to current rank)
        group_size: Number of sentences per group
        coordinate_key: Key in dataset for accessing spatial coordinates
        window_size: Size of spatial window for neighbor search (relative to total space)
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
        window_size: float = 0.1,
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
        self.window_size = window_size

        # Calculate sizes for distributed sampling
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Precompute spatial data
        self.precomputed = self._precompute_spatial_data()
        
    # Reuse methods from SpatialGroupSampler
    _precompute_spatial_data = SpatialGroupSampler._precompute_spatial_data
    _get_spatial_group = SpatialGroupSampler._get_spatial_group

    def __iter__(self) -> Iterator[int]:
        """Returns distributed iterator of indices where spatial groups are kept together."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Generate all groups first
        all_indices = []
        num_complete_groups = self.total_size // self.group_size
        
        for _ in range(num_complete_groups):
            center_idx = self.rng.randint(0, len(self.dataset))
            group = self._get_spatial_group(center_idx)
            all_indices.extend(group)
            
        # Handle remaining indices if needed
        remaining = self.total_size - len(all_indices)
        if remaining > 0 and not self.drop_last:
            center_idx = self.rng.randint(0, len(self.dataset))
            last_group = self._get_spatial_group(center_idx)
            all_indices.extend(last_group[:remaining])
        
        assert len(all_indices) == self.total_size, \
            f"Expected {self.total_size} indices but got {len(all_indices)}"
            
        # Subsample indices for this worker
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

@dataclass
class SpatialGroupCollator:
    """
    Collator that handles both grouping of spatially-related sentences and their labels.
    Handles partial batches gracefully.
    """
    group_size: int
    label_key: str
    feature_keys: Optional[List[str]] = None
    pad_token_id: int = 0
    padding: str = "max_length"
    
    def __post_init__(self):
        if self.feature_keys is None:
            self.feature_keys = ["input_ids"]

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Group features into spatial groups and aggregate their labels.
        Handles partial batches (when len(features) < group_size).
        """
        if not features:
            return {}
            
        # Handle case where we have fewer features than group_size
        num_features = len(features)
        if num_features < self.group_size:
            # Pad the features list with copies of the last feature
            padding_needed = self.group_size - num_features
            features.extend([features[-1]] * padding_needed)
            print("Warning: Padding features to meet group size.")
            print(f"Original number of features: {num_features}, padded to: {len(features)}")
            
        # Calculate number of complete groups
        num_groups = len(features) // self.group_size
        grouped_features = []
        
        for group_idx in range(num_groups):
            start_idx = group_idx * self.group_size
            end_idx = start_idx + self.group_size
            group = features[start_idx:end_idx]
            
            # Collect features for this group
            group_dict = {key: [] for key in self.feature_keys}
            group_labels = []
            
            for item in group:
                # Collect all features except labels
                for key in self.feature_keys:
                    tensor = item[key]
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.tensor(tensor)
                    group_dict[key].append(tensor)
                
                # Collect labels separately
                label = item[self.label_key]
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                group_labels.append(label)
            
            # Stack features
            for key in self.feature_keys:
                # Find max length in this group
                max_len = max(x.size(-1) for x in group_dict[key])
                
                # Pad each tensor to max length
                padded = []
                for tensor in group_dict[key]:
                    if tensor.size(-1) < max_len:
                        padding = torch.full(
                            (max_len - tensor.size(-1),),
                            self.pad_token_id,
                            dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, padding])
                    padded.append(tensor)
                
                group_dict[key] = torch.stack(padded)
            
            # Aggregate labels for the group (majority vote)
            group_labels = torch.stack(group_labels)

            # For classification, use mode
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
    def __init__(self, *args, additional_attributes=None, 
                 spatial_group_size=32, spatial_window_size=0.1,
                 spatial_label_key='area_labels', **kwargs):
        kwargs["data_collator"] = SpatialGroupCollator(
            group_size=spatial_group_size,
            label_key=spatial_label_key,
            feature_keys=["input_ids"],  # Add other feature keys as needed
            pad_token_id=0 # Adjust as needed
        )
        # Store spatial sampling parameters
        self.spatial_group_size = spatial_group_size
        self.spatial_window_size = spatial_window_size
        
        super().__init__(*args, **kwargs)

        assert self.args.train_batch_size % spatial_group_size == 0, \
            "train_batch_size must be divisible by spatial_group_size"

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        # Build the sampler.
        # Use spatial sampling
        if self.args.world_size <= 1:
            return SpatialGroupSampler(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                group_size=self.spatial_group_size,
                window_size=self.spatial_window_size,
                seed=self.args.seed
            )
        else:
            return DistributedSpatialGroupSampler(
                dataset=self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
                group_size=self.spatial_group_size,
                window_size=self.spatial_window_size
            )

    def _get_eval_sampler(self, eval_dataset) -> Optional[torch.utils.data.sampler.Sampler]:
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
            return SpatialGroupSampler(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                group_size=self.spatial_group_size,
                window_size=self.spatial_window_size,
                seed=self.args.seed
            )
        else:
            return DistributedSpatialGroupSampler(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
                group_size=self.spatial_group_size,
                window_size=self.spatial_window_size
            )
        
## Usage Example

# # For single GPU training:
# sampler = SpatialGroupSampler(
#     dataset=dataset,
#     batch_size=64,
#     group_size=32,
#     window_size=0.1
# )

# # For distributed training:
# sampler = DistributedSpatialGroupSampler(
#     dataset=dataset,
#     batch_size=64,
#     group_size=32,
#     window_size=0.1,
#     num_replicas=dist.get_world_size(),
#     rank=dist.get_rank()
# )

# # Use with DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=None,  # Controlled by sampler
#     sampler=sampler,
#     collate_fn=spatial_collate_fn
# )