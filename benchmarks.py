import os
import hydra
import wandb
import numpy as np
import anndata as ad
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datasets import load_from_disk, DatasetDict
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import PreTrainedModel, TrainingArguments

# Import necessary components from train.py
from samplers import (
    SpatialGroupSampler,
    SpatialGroupCollator,
    MultiformerTrainer
)

from transformers import PretrainedConfig

class DummyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DummyModel(PreTrainedModel):
    config_class = DummyConfig
    
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, *args, **kwargs):
        return None
        
def get_dataloaders(datasets: DatasetDict, cfg: DictConfig) -> Dict[str, DataLoader]:
    """
    Create dataloaders using MultiformerTrainer infrastructure.
    """
    # Create minimal training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.data.group_size*32,
        per_device_eval_batch_size=cfg.data.group_size*32,
        remove_unused_columns=False,  # Important for MultiformerTrainer
    )

    dummy_config = DummyConfig()

    # Initialize trainer with dummy model
    trainer = MultiformerTrainer(
        model=DummyModel(dummy_config),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        spatial_group_size=cfg.data.group_size,
        spatial_label_key="labels",
        coordinate_key='CCF_streamlines'
    )
    
    # Get dataloaders
    dataloaders = {
        "train": trainer.get_train_dataloader(),
        "validation": trainer.get_eval_dataloader(datasets["validation"]),
        "test": trainer.get_test_dataloader(datasets["test"])
    }
    
    return dataloaders

def load_and_align_anndata(
    train_filenames: List[str],
    test_filenames: List[str],
    data_dir: str,
    dataset: DatasetDict,
    coordinate_key = "CCF_streamlines",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Load and concatenate AnnData files, ensuring alignment with dataset indices.
    Handles train and test files separately to match the original tokenization.
    
    Args:
        train_filenames: List of h5ad filenames for training data
        test_filenames: List of h5ad filenames for test data
        data_dir: Directory containing h5ad files
        dataset: HuggingFace dataset to align with
    
    Returns:
        Tuple of (train_adata, test_adata)
    """
    print("Loading and processing AnnData files...")
    
    def process_files(filenames: List[str]) -> ad.AnnData:
        """Helper function to process a list of files"""
        adatas = []
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            print(f"Loading {filepath}")
            adata = ad.read_h5ad(filepath)
            
            # Filter cells with invalid CCF coordinates
            valid_mask = ~np.isnan(adata.obsm[coordinate_key]).any(axis=1)
            adata = adata[valid_mask]
            adatas.append(adata)
        
        return ad.concat(adatas, join='outer', fill_value=0)
    
    # Process train and test files separately
    train_adata = process_files(train_filenames)
    test_adata = process_files(test_filenames)
    
    # Verify alignment with dataset
    print("Verifying alignment with dataset...")
    
    # Check first 100 cells in train dataset
    test_dataset = dataset['test']
    dataset_h3types = np.array(test_dataset[:100]['H3_type'])
    adata_h3types = test_adata.obs['H3_type'].values[:100]
    
    if not np.array_equal(dataset_h3types, adata_h3types):
        mismatches = np.where(dataset_h3types != adata_h3types)[0]
        mismatch_info = [
            f"Index {i}: Dataset H3_type: {dataset_h3types[i]}, AnnData H3_type: {adata_h3types[i]}"
            for i in mismatches
        ]
        raise ValueError(
            "Mismatch found in H3 types:\n" + "\n".join(mismatch_info)
        )
    
    print("Alignment verification passed!")
    return train_adata, test_adata

def prepare_datasets(dataset_dict: DatasetDict, cfg: DictConfig) -> DatasetDict:
    """
    Prepare train/validation split from dataset.
    """
    train_dataset = dataset_dict["train"]
    
    # Create train/validation split
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=cfg.data.validation_split,
        random_state=cfg.seed
    )
    
    # Add unique ids if not present
    if 'uuid' not in train_dataset.features:
        dataset_dict["test"] = dataset_dict["test"].add_column(
            "uuid", 
            np.arange(len(dataset_dict["test"]))
        )
        train_dataset = train_dataset.add_column(
            "uuid", 
            np.arange(len(dataset_dict["train"]))
        )
    
    # Select train and validation datasets
    val_dataset = train_dataset.select(val_idx)
    train_dataset = train_dataset.select(train_idx)
        
    # Limit dataset size if in debug mode
    if hasattr(cfg.data, 'max_train_samples') and cfg.data.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), cfg.data.max_train_samples)))
        
    # Limit validation/test size if specified
    if hasattr(cfg.data, 'max_eval_samples') and cfg.data.max_eval_samples is not None:
        val_dataset = val_dataset.select(range(min(len(val_dataset), cfg.data.max_eval_samples)))
        dataset_dict["test"] = dataset_dict["test"].select(range(min(len(dataset_dict["test"]), cfg.data.max_eval_samples)))

    # rename the labels to match the model's expected input
    if hasattr(cfg.data, 'label_key'):
        train_dataset = train_dataset.rename_column(cfg.data.label_key, "labels")
        val_dataset = val_dataset.rename_column(cfg.data.label_key, "labels")
        dataset_dict["test"] = dataset_dict["test"].rename_column(cfg.data.label_key, "labels")

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": dataset_dict["test"]
    })

def setup_wandb(cfg: DictConfig):
    """Initialize W&B logging"""
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

def evaluate_method(
    predictions: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    prefix: str,
    label_names: Dict,
    output_dir: str
) -> Dict:
    """Evaluate predictions and save results."""
    from sklearn.metrics import classification_report
    
    # Generate detailed classification report
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0
    )
    
    # Extract metrics
    metrics = {
        "accuracy": (predictions == labels).mean(),
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"]
    }
    
    # Add per-class metrics
    for label in report:
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics[f"f1_{label}"] = report[label]["f1-score"]
            metrics[f"precision_{label}"] = report[label]["precision"]
            metrics[f"recall_{label}"] = report[label]["recall"]
    
    # Save predictions
    output_dict = {
        'predictions': predictions,
        'labels': labels,
        'indices': indices,
        'label_names': label_names
    }
    np.save(os.path.join(output_dir, f"{prefix}_predictions.npy"), output_dict)
    
    # Log to wandb
    wandb.log({f"{prefix}_{k}": v for k, v in metrics.items()})
    
    return metrics

def get_bulk_expression(adata: Tuple[ad.AnnData, ad.AnnData], indices: np.ndarray, is_test: bool) -> np.ndarray:
    """Get mean expression for a group of cells."""
    train_adata, test_adata = adata
    current_adata = test_adata if is_test else train_adata
    
    # Ensure indices is 2D even when group_size=1
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
    
    features = []
    for batch_indices in indices:
        mask = np.zeros(len(current_adata), dtype=bool)
        mask[batch_indices] = True
        features.append(np.array(current_adata[mask].X.mean(axis=0)))
    return np.vstack(features)

def run_bulk_expression_rf(
    datasets: DatasetDict,
    adata: Tuple[ad.AnnData, ad.AnnData],
    cfg: DictConfig
) -> None:
    """Run random forest on bulk expression data."""
    
    # Initialize random forest and scaler
    rf = RandomForestClassifier(**cfg.random_forest)
    scaler = StandardScaler()
    
    # Get dataloaders using trainer infrastructure
    dataloaders = get_dataloaders(datasets, cfg)
    
    # Training
    print("Collecting training features...")
    train_features = []
    train_labels = []
    train_indices = []
    
    for batch in dataloaders["train"]:
        indices = batch['indices'].cpu().numpy()
        train_indices.extend(indices)
        bulk_expression = get_bulk_expression(adata, indices, is_test=False)
        train_features.append(bulk_expression)
        train_labels.append(batch['labels'].cpu().numpy())
    
    train_features = np.vstack(train_features)
    train_features = scaler.fit_transform(train_features)
    train_labels = np.concatenate(train_labels)

    print("Train features:", train_features.shape)
    print("Train labels:", train_labels.shape)
    
    print("Training random forest...")
    rf.fit(train_features, train_labels)
    
    # Evaluate on all sets
    for name in ['train','validation', 'test']:
        loader = dataloaders[name]
        is_test = name == 'test'
        print(f"Evaluating on {name} set...")
        predictions = []
        labels = []
        indices = []
        
        for batch in loader:
            batch_indices = batch['indices'].cpu().numpy()
            indices.extend(batch_indices)
            bulk_expression = get_bulk_expression(adata, batch_indices, is_test=is_test)
            features = scaler.transform(bulk_expression)
            
            pred = rf.predict(features)
            predictions.extend(pred)
            labels.extend(batch['labels'].cpu().numpy())

        evaluate_method(
            np.array(predictions),
            np.array(labels),
            np.array(indices),
            f"bulk_expression_{name}",
            cfg.data.label_names,
            cfg.output_dir
        )

def prepare_h3type_data(dataset: DatasetDict) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, Dict[int, int]]]:
    """
    Prepare H3 type data for fast access during training.
    Creates both type mapping and index mapping.
    """
    # Create type mapping
    all_h3_types = set()
    for split in dataset.values():
        all_h3_types.update(split['H3_type'])
    type_to_idx = {h3_type: idx for idx, h3_type in enumerate(sorted(list(all_h3_types)))}
    
    # Create numpy arrays and index mappings for fast access
    h3_arrays = {}
    index_maps = {}
    
    for split, dset in dataset.items():
        # Create mapping from dataset index to array index
        index_maps[split] = {
            idx: i for i, idx in enumerate(dset['uuid'])
        }
        
        # Create array with H3 type indices
        h3_arrays[split] = np.array([
            type_to_idx[t] for t in dset['H3_type']
        ])
    
    return h3_arrays, type_to_idx, index_maps

def get_h3type_histogram(
    indices: np.ndarray, 
    h3_array: np.ndarray, 
    index_map: Dict[int, int],
    n_types: int
) -> np.ndarray:
    """
    Create histogram of H3 types for a group of cells using vectorized operations.
    
    Args:
        indices: Original dataset indices
        h3_array: Pre-computed array of H3 type indices
        index_map: Mapping from dataset indices to array indices
        n_types: Total number of H3 types
    """
    # Ensure indices is 2D: (n_groups, group_size)
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)  # One group with multiple cells
    batch_size = indices.shape[0]
    histogram = np.zeros((batch_size, n_types))

    for i, batch_indices in enumerate(indices):
        # Map dataset indices to array indices
        array_indices = [index_map[idx] for idx in batch_indices]
        # Get H3 types using mapped indices
        type_indices = h3_array[array_indices]
        histogram[i] = np.bincount(type_indices, minlength=n_types)
        
        total = histogram[i].sum()
        if total > 0:
            histogram[i] /= total

    return histogram

def run_h3type_rf(
    datasets: DatasetDict,
    cfg: DictConfig
) -> None:
    """Run random forest on H3 type histograms."""
    
    # Prepare data once
    h3_arrays, type_to_idx, index_maps = prepare_h3type_data(datasets)
    n_types = len(type_to_idx)
    
    # Get dataloaders
    dataloaders = get_dataloaders(datasets, cfg)
    
    # Initialize random forest
    rf = RandomForestClassifier(**cfg.random_forest)
    
    # Training
    print("Training H3 type random forest...")
    train_features = []
    train_labels = []
    train_indices = []
    
    for batch in dataloaders["train"]:
        indices = batch['indices'].cpu().numpy()
        train_indices.extend(indices)
        histogram = get_h3type_histogram(
            indices, 
            h3_arrays['train'],
            index_maps['train'],
            n_types
        )
        train_features.append(histogram)
        train_labels.append(batch['labels'].cpu().numpy())
        
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)

    
    rf.fit(train_features, train_labels)
    
    # Evaluate on all sets
    for name in ['train','validation', 'test']:
        loader = dataloaders[name]
        print(f"Evaluating on {name} set...")
        predictions = []
        labels = []
        indices = []
        
        for batch in loader:
            batch_indices = batch['indices'].cpu().numpy()
            indices.extend(batch_indices)
            histogram = get_h3type_histogram(
                batch_indices, 
                h3_arrays[name],
                index_maps[name],
                n_types
            )
            pred = rf.predict(histogram)
            predictions.extend(pred)
            labels.extend(batch['labels'].cpu().numpy())
            
        evaluate_method(
            np.array(predictions),
            np.array(labels),
            np.array(indices),
            f"h3type_{name}",
            cfg.data.label_names,
            cfg.output_dir
        )

@hydra.main(version_base=None, config_path="config", config_name="benchmarks")
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))

    if cfg.debug:
        # Limit dataset size for faster iteration
        cfg.data.max_train_samples = 10000
        cfg.data.max_eval_samples = 10000
    
    # Setup wandb
    setup_wandb(cfg)
    
    # Load dataset
    dataset_dict = load_from_disk(cfg.data.dataset_path)
    datasets = prepare_datasets(dataset_dict, cfg)
    print(f"Loaded datasets: {datasets}")
    
    # Run benchmarks
    if cfg.run_bulk_expression:
        print("Running bulk expression random forest...")
        adata = load_and_align_anndata(
            cfg.data.train_h5ad_files, 
            cfg.data.test_h5ad_files,  
            cfg.data.h5ad_directory,
            datasets
        )
        run_bulk_expression_rf(datasets, adata, cfg)
    
    if cfg.run_h3type:
        print("Running H3 type random forest...")
        run_h3type_rf(datasets, cfg)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()