import os
import hydra
import wandb
import numpy as np
import anndata as ad
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datasets import load_from_disk, DatasetDict, Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, TrainingArguments
import json
import scipy
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder

# Import necessary components from train.py
from samplers import (
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
        coordinate_key='CCF_streamlines',
        additional_feature_keys=['raw_counts'],
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
    
    # Check first 10000 cells in train dataset
    test_dataset = dataset['test']
    dataset_h3types = np.array(test_dataset[:10000]['H3_type'])
    adata_h3types = test_adata.obs['H3_type'].values[:10000]
    
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
        # "f1_macro": report["macro avg"]["f1-score"],
        # "f1_weighted": report["weighted avg"]["f1-score"]
    }
    
    # Add per-class metrics
    # for label in report:
    #     if label not in ["accuracy", "macro avg", "weighted avg"]:
    #         metrics[f"f1_{label}"] = report[label]["f1-score"]
    #         metrics[f"precision_{label}"] = report[label]["precision"]
    #         metrics[f"recall_{label}"] = report[label]["recall"]
    
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
        features.append(np.array(current_adata[batch_indices].X.mean(axis=0)))

    return np.vstack(features)

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

def create_dataset_from_anndata(adata: ad.AnnData, cfg: DictConfig) -> Dataset:
    """
    Create a HuggingFace Dataset directly from AnnData object.
    """
    # Extract features (gene expression)
    features = np.array(adata.X.todense() if scipy.sparse.issparse(adata.X) else adata.X)

    print(f"Features shape: {features.shape}")
    
    # Extract coordinates
    coordinates = adata.obsm["CCF_streamlines"]
    
    # Extract area labels using the same logic as in tokenize_cells.py
    with open('data/files/area_ancestor_id_map.json', 'r') as f:
        area_ancestor_id_map = json.load(f)
    with open('data/files/area_name_map.json', 'r') as f:
        area_name_map = json.load(f)
    
    area_name_map['0'] = 'outside_brain'
    annotation2area_int = {0.0:0}
    for a in area_ancestor_id_map.keys():
        higher_area_id = area_ancestor_id_map[str(int(a))][1] if len(area_ancestor_id_map[str(int(a))])>1 else a
        annotation2area_int[float(a)] = int(higher_area_id)

    unique_areas = np.unique(list(annotation2area_int.values()))
    area_classes = np.arange(len(unique_areas))
    id2id = {float(k):v for (k,v) in zip(unique_areas, area_classes)}
    annotation2area_class = {k: id2id[int(v)] for k,v in annotation2area_int.items()}
    
    # Convert CCF annotations to area labels
    labels = np.array([annotation2area_class[x] for x in adata.obs['CCFano']])
    
    # Extract H3 types
    h3types = adata.obs['H3_type'].values

    # # Filter dataset to only include cells for which the CCF_streamlines is not nans
    # same as tokenized_dataset = tokenized_dataset.filter(lambda x: not np.isnan(np.sum(x['CCF_streamlines'])))
    # Filter out indices where CCF_streamlines contains NaN values
    valid_mask = ~np.isnan(coordinates).any(axis=1)
    features = features[valid_mask]
    coordinates = coordinates[valid_mask]
    labels = labels[valid_mask]
    h3types = h3types[valid_mask]
    indices = np.arange(len(adata))[valid_mask]
    
    # Create dataset
    return Dataset.from_dict({
        'expression': features,
        'CCF_streamlines': coordinates,
        'labels': labels,
        'H3_type': h3types,
        'uuid': indices
    })

def prepare_features_from_anndata(
    train_adata: ad.AnnData,
    test_adata: ad.AnnData,
    cfg: DictConfig,
    scaler: StandardScaler,
    feature_type: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare features directly from AnnData objects.
    Returns dict with keys 'train', 'validation', 'test', each containing (features, labels, indices)
    """
    # Create datasets from AnnData
    train_valid_dataset = create_dataset_from_anndata(train_adata, cfg)
    test_dataset = create_dataset_from_anndata(test_adata, cfg)
    
    rng = check_random_state(cfg.seed)

    # Split train/validation
    train_idx, val_idx = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=cfg.data.validation_split,
        random_state=rng
    )
    
    # Select splits using Dataset.select()
    train_dataset = train_valid_dataset.select(train_idx)
    val_dataset = train_valid_dataset.select(val_idx)

    if feature_type == "h3type":
        # One-hot encode H3 types
        # Collect all unique H3 types from all splits
        all_h3_types = np.concatenate([
            train_dataset['H3_type'],
            val_dataset['H3_type'],
            test_dataset['H3_type']
        ])
        encoder = OneHotEncoder(sparse_output=False) 
        # Fit on all types
        encoder.fit(all_h3_types.reshape(-1, 1))
        # Transform each split
        train_features = encoder.transform(np.array(train_dataset['H3_type']).reshape(-1, 1))
        val_features = encoder.transform(np.array(val_dataset['H3_type']).reshape(-1, 1))
        test_features = encoder.transform(np.array(test_dataset['H3_type']).reshape(-1, 1))
    else:
        # Extract and scale continuous features
        train_features = np.array(train_dataset['expression'])
        val_features = np.array(val_dataset['expression'])
        test_features = np.array(test_dataset['expression'])
        
        # Only apply scaling for non-categorical features
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)
            
    # Extract features and labels as numpy arrays
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
        
    # Verify no NaN or infinite values
    if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)):
        raise ValueError("Training features contain NaN or infinite values")
    
    print(f"Train features final shape: {train_features.shape}")
    print(f"Test features final shape: {test_features.shape}")
    
    output =  {
        'train': (
            train_features,
            np.array(train_dataset['labels']),
            train_idx
        ),
        'validation': (
            val_features,
            np.array(val_dataset['labels']),
            val_idx
        ),
        'test': (
            test_features,
            np.array(test_dataset['labels']),
            np.arange(len(test_dataset))
        )
    }

    if cfg.get('resample_adata', False):
        # Resample the data
        print("Resampling the data")
        for name, (features, labels, indices) in output.items():
            resampled_indices = np.random.choice(range(len(indices)), len(indices), replace=True)
            output[name] = (features[resampled_indices], labels[resampled_indices], indices[resampled_indices])

    return output        

def verify_indices(features, labels, indices, name):
    print(f"\n=== Verification for {name} ===")
    print(f"Features shape: {features.shape}")
    print(f"Unique indices: {len(np.unique(indices))}")
    print(f"First 5 indices: {indices[:5]}")
    print(f"Labels distribution: {np.unique(labels, return_counts=True)}")
    # Add checksum for features
    print(f"Features checksum: {np.sum(features)}")

def debug_feature_extraction(adata, indices, path_name):
    features = adata[indices].X
    if scipy.sparse.issparse(features):
        features = features.todense()
    print(f"\n=== Feature extraction {path_name} ===")
    print(f"Shape: {features.shape}")
    print(f"Mean: {np.mean(features)}")
    print(f"Std: {np.std(features)}")
    print(f"Number of zeros: {np.sum(features == 0)}")
    return features

def run_classifier(
    datasets: DatasetDict,
    adata: Optional[Tuple[ad.AnnData, ad.AnnData]],
    cfg: DictConfig,
    classifier_type: str,
    feature_type: str
) -> None:
    """Generic function to run different classifiers on different feature types."""
    
    # Initialize classifier and scaler
    if classifier_type == "random_forest":
        clf = RandomForestClassifier(**cfg.random_forest)
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(**cfg.logistic_regression)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    scaler = StandardScaler()

    # If using direct AnnData path
    if cfg.data.group_size == 1 and cfg.get('on_adata', False):
        train_adata, test_adata = adata
        splits = prepare_features_from_anndata(train_adata, test_adata, cfg, scaler, feature_type)
        
        # Train classifier
        train_features, train_labels, train_indices = splits['train']
        clf.fit(train_features, train_labels)
        np.save(f"{cfg.output_dir}/train_{feature_type}_adata_true.npy", train_features)
        np.save(f"{cfg.output_dir}/train_{feature_type}_labels_adata_true.npy", train_labels)

        # Evaluate
        for name, (features, labels, indices) in splits.items():
            predictions = clf.predict(features)
            evaluate_method(
                predictions,
                labels,
                indices,
                f"{classifier_type}_{feature_type}_{name}",
                cfg.data.label_names,
                cfg.output_dir
            )
        return

    # Original dataloader-based path
    dataloaders = get_dataloaders(datasets, cfg)
    
    # Prepare H3 type data if needed
    if feature_type == "h3type":
        h3_arrays, type_to_idx, index_maps = prepare_h3type_data(datasets)
        n_types = len(type_to_idx)
    
    # Training
    print(f"Collecting training features for {classifier_type} on {feature_type}...")
    train_features = []
    train_labels = []
    train_indices = []
    
    for batch in dataloaders["train"]:
        indices = batch['indices'].cpu().numpy() # these are uuid = indices in the original dataset['train'] before splitting
        train_indices.extend(indices)
        
        if feature_type == "bulk_expression":
            if cfg.load_counts_from_hf:
                features = batch['raw_counts'].cpu().numpy().squeeze()
            else:
                features = get_bulk_expression(adata, indices, is_test=False)
        else:  # h3type
            features = get_h3type_histogram(indices, h3_arrays['train'], index_maps['train'], n_types)
            
        train_features.append(features)
        train_labels.append(batch['labels'].cpu().numpy())
    
    train_features = np.vstack(train_features)
    train_features = scaler.fit_transform(train_features)
    train_labels = np.concatenate(train_labels)

    print("Train features:", train_features.shape)
    print("Train labels:", train_labels.shape)
    np.save(f"{cfg.output_dir}/train_{feature_type}_adata_false.npy", train_features)
    np.save(f"{cfg.output_dir}/train_{feature_type}_labels_adata_false.npy", train_labels)
    
    print(f"Training {classifier_type}...")
    # Check for NaN/Inf values
    assert not np.any(np.isnan(train_features)), "Features contain NaN values"
    assert not np.any(np.isinf(train_features)), "Features contain infinite values"

    # Verify feature array is contiguous
    if not train_features.flags['C_CONTIGUOUS']:
        train_features = np.ascontiguousarray(train_features)
    # Fit with verbose logging
    clf.fit(train_features, train_labels)
    
    # Evaluate on all sets
    for name in ['train', 'validation', 'test']:
        loader = dataloaders[name]
        is_test = name == 'test'
        print(f"Evaluating on {name} set...")
        predictions = []
        labels = []
        indices = []
        
        for batch in loader:
            batch_indices = batch['indices'].cpu().numpy()
            indices.extend(batch_indices)
            
            if feature_type == "bulk_expression":
                if cfg.load_counts_from_hf: 
                    features = batch['raw_counts'].cpu().numpy().squeeze()
                else:
                    features = get_bulk_expression(adata, batch_indices, is_test=is_test)                    
            else:  # h3type
                features = get_h3type_histogram(
                    batch_indices, 
                    h3_arrays[name],
                    index_maps[name],
                    n_types
                )
                
            features = scaler.transform(features)
            pred = clf.predict(features)
            predictions.extend(pred)
            labels.extend(batch['labels'].cpu().numpy())

        evaluate_method(
            np.array(predictions),
            np.array(labels),
            np.array(indices),
            f"{classifier_type}_{feature_type}_{name}",
            cfg.data.label_names,
            cfg.output_dir
        )

@hydra.main(version_base=None, config_path="config", config_name="benchmarks")
def main(cfg: DictConfig) -> None:
    print("Running benchmarks...")
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
    
    # Load AnnData if needed for bulk expression
    adata = load_and_align_anndata(
            cfg.data.train_h5ad_files, 
            cfg.data.test_h5ad_files,  
            cfg.data.h5ad_directory,
            datasets
        )

    # ensure that the lengths are the same
    if cfg.on_adata and cfg.debug:
        subsampled_idx_train = np.random.choice(len(adata[0]), len(datasets['train']), replace=False)
        subsampled_idx_test = np.random.choice(len(adata[1]), len(datasets['test']), replace=False)
        adata = (adata[0][subsampled_idx_train], adata[1][subsampled_idx_test])
    
    # Run benchmarks
    if cfg.run_bulk_expression_rf:
        run_classifier(datasets, adata, cfg, "random_forest", "bulk_expression")
    
    if cfg.run_bulk_expression_lr:
        run_classifier(datasets, adata, cfg, "logistic_regression", "bulk_expression")
    
    if cfg.run_h3type_rf:
        run_classifier(datasets, adata, cfg, "random_forest", "h3type")
    
    if cfg.run_h3type_lr:
        run_classifier(datasets, adata, cfg, "logistic_regression", "h3type")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
        main()

## Results of on_adata=True
# wandb: Run summary:
# wandb:       logistic_regression_bulk_expression_test_accuracy 0.1854
# wandb:      logistic_regression_bulk_expression_train_accuracy 0.40067
# wandb: logistic_regression_bulk_expression_validation_accuracy 0.191
# wandb:                logistic_regression_h3type_test_accuracy 0.1699
# wandb:               logistic_regression_h3type_train_accuracy 0.21056
# wandb:          logistic_regression_h3type_validation_accuracy 0.175
# wandb:             random_forest_bulk_expression_test_accuracy 0.2065
# wandb:            random_forest_bulk_expression_train_accuracy 0.92122
# wandb:       random_forest_bulk_expression_validation_accuracy 0.2
# wandb:                      random_forest_h3type_test_accuracy 0.1454
# wandb:                     random_forest_h3type_train_accuracy 0.15933
# wandb:                random_forest_h3type_validation_accuracy 0.139

## Results of on_adata=False
# wandb:       logistic_regression_bulk_expression_test_accuracy 0.1657
# wandb:      logistic_regression_bulk_expression_train_accuracy 0.1463
# wandb: logistic_regression_bulk_expression_validation_accuracy 0.0914
# wandb:                logistic_regression_h3type_test_accuracy 0.1863
# wandb:               logistic_regression_h3type_train_accuracy 0.1971
# wandb:          logistic_regression_h3type_validation_accuracy 0.1778
# wandb:             random_forest_bulk_expression_test_accuracy 0.1633
# wandb:            random_forest_bulk_expression_train_accuracy 0.5095
# wandb:       random_forest_bulk_expression_validation_accuracy 0.1062
# wandb:                      random_forest_h3type_test_accuracy 0.0975
# wandb:                     random_forest_h3type_train_accuracy 0.1615
# wandb:                random_forest_h3type_validation_accuracy 0.158


# ### On all data
# true
# wandb:       logistic_regression_bulk_expression_test_accuracy 0.26067
# wandb:      logistic_regression_bulk_expression_train_accuracy 0.26087
# wandb: logistic_regression_bulk_expression_validation_accuracy 0.26088
# wandb:                logistic_regression_h3type_test_accuracy 0.18645
# wandb:               logistic_regression_h3type_train_accuracy 0.18792
# wandb:          logistic_regression_h3type_validation_accuracy 0.1863
# wandb:             random_forest_bulk_expression_test_accuracy 0.25975
# wandb:            random_forest_bulk_expression_train_accuracy 0.42724
# wandb:       random_forest_bulk_expression_validation_accuracy 0.26243
# wandb:                      random_forest_h3type_test_accuracy 0.14709
# wandb:                     random_forest_h3type_train_accuracy 0.14894
# wandb:                random_forest_h3type_validation_accuracy 0.14985

# false
# wandb: Run summary:
# wandb:       logistic_regression_bulk_expression_test_accuracy 0.09024
# wandb:      logistic_regression_bulk_expression_train_accuracy 0.12697
# wandb: logistic_regression_bulk_expression_validation_accuracy 0.1246
# wandb:                logistic_regression_h3type_test_accuracy 0.17947
# wandb:               logistic_regression_h3type_train_accuracy 0.19149
# wandb:          logistic_regression_h3type_validation_accuracy 0.19145
# wandb:             random_forest_bulk_expression_test_accuracy 0.08254
# wandb:            random_forest_bulk_expression_train_accuracy 0.22865
# wandb:       random_forest_bulk_expression_validation_accuracy 0.11936
# wandb:                      random_forest_h3type_test_accuracy 0.12144
# wandb:                     random_forest_h3type_train_accuracy 0.1483
# wandb:                random_forest_h3type_validation_accuracy 0.14861
# wandb: 

# These should be the same. The only difference is that the first one uses the adata directly, while the second one uses the dataloader.
# It is clearly just a problem with the bulk expression. Suggests an indexing or feature extraction problem.
# Debugging strategies:
# - Build a fake small dataset (would require re-tokenizing the data)
# 
# Hypotheses:
# - selecting the "Wrong" datapoints?
# - or messing up the features?