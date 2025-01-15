import os
from typing import Dict, Optional, List
from sklearn.metrics import classification_report

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
from datasets import DatasetDict, load_from_disk, disable_caching
from transformers import (
    TrainingArguments,
    set_seed,
)
from sklearn.model_selection import train_test_split

from model import HierarchicalBert, HierarchicalBertConfig
from samplers import MultiformerTrainer
from transformers import BertModel, BertConfig, BertForSequenceClassification, Trainer


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

def create_model(config: DictConfig, class_weights: Optional[torch.Tensor] = None) -> HierarchicalBert:
    """Create model based on pretraining configuration."""
    # Convert class weights to list if present for JSON serialization
    class_weights_list = class_weights.tolist() if class_weights is not None else None
    
    if config.model.pretrained_type == "full":
        if not config.model.bert_path_or_name:
            raise ValueError("model_path must be specified when pretrained_type is 'full'")
            
        # Load full pretrained hierarchical model
        model = HierarchicalBert.from_pretrained(
            config.model.bert_path_or_name,
            num_labels=config.model.num_labels,
            class_weights=class_weights,  # Pass tensor to model
            single_cell_vs_group_weight=config.model.single_cell_vs_group_weight,
        )
    elif config.model.pretrained_type == "single-cell":
        model = BertForSequenceClassification.from_pretrained(
            config.model.bert_path_or_name,
            num_labels=config.model.num_labels,
        )
    elif config.model.pretrained_type == "single-cell-from-scratch":
        config =  BertConfig(
            num_labels=config.model.num_labels,
            bert_config=config.model.bert_path_or_name,
            num_set_layers=config.model.num_set_layers,
            set_hidden_size=config.model.set_hidden_size,
            num_attention_heads=config.model.num_attention_heads,
            dropout_prob=config.model.dropout_prob,
            **(config.model.get("bert_params", {}) if config.model.pretrained_type == "none" else {})
        )
        model = BertForSequenceClassification(config)
    else:
        # For both "none" and "bert_only" cases, first create config
        model_config = HierarchicalBertConfig(
            num_labels=config.model.num_labels,
            # For "none", pass None to use default config
            bert_config=None if config.model.pretrained_type == "none" else config.model.bert_path_or_name,
            num_set_layers=config.model.num_set_layers,
            set_hidden_size=config.model.set_hidden_size,
            num_attention_heads=config.model.num_attention_heads,
            dropout_prob=config.model.dropout_prob,
            class_weights=class_weights_list,  # Pass list to config
            single_cell_vs_group_weight=config.model.single_cell_vs_group_weight,
            detach_bert_embeddings=config.model.get("detach_bert_embeddings", False),
            single_cell_loss_after_set=config.model.get("single_cell_loss_after_set", False),
            use_relative_positions=config.model.relative_positions.enabled,
            position_encoding_dim=config.model.relative_positions.encoding_dim,
            position_encoding_type=config.model.relative_positions.encoding_type,
            **(config.model.get("bert_params", {}) if config.model.pretrained_type == "none" else {})
        )
        
        # Initialize new model
        model = HierarchicalBert(model_config)
        
        # If bert_only, load pretrained BERT weights
        if config.model.pretrained_type == "bert_only":
            pretrained_bert = BertModel.from_pretrained(config.model.bert_path_or_name)
            model.bert.load_state_dict(pretrained_bert.state_dict())

        # Set class weights as tensor in model after initialization
        if hasattr(model, 'class_weights') and class_weights is not None:
            model.class_weights = class_weights  # Set tensor directly on model

    return model


def load_class_weights(cfg: DictConfig) -> Optional[torch.Tensor]:
    """Load class weights if enabled"""
    if not cfg.data.class_weights.enabled:
        return None
        
    if cfg.data.class_weights.path is None:
        raise ValueError("Class weights enabled but no path provided")
        
    weights = np.load(cfg.data.class_weights.path)
    
    # Validate weights
    if len(weights) != cfg.model.num_labels:
        raise ValueError(
            f"Number of class weights ({len(weights)}) does not match "
            f"number of labels ({cfg.model.num_labels})"
        )
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Log class weights to wandb
    wandb.run.summary["class_weights"] = weights.tolist()
    
    return weights


def prepare_datasets(dataset_dict: DatasetDict, config: DictConfig) -> DatasetDict:
    """Prepare train/validation split from pre-tokenized dataset"""
    train_dataset = dataset_dict["train"]
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=config.data.validation_split,
        random_state=config.seed
    )

    # Add unique ids to datasets
    dataset_dict["test"] = dataset_dict["test"].add_column("uuid", np.arange(len(dataset_dict["test"])))
    train_dataset = train_dataset.add_column("uuid", np.arange(len(dataset_dict["train"])))

    # Select train and validation datasets
    val_dataset = train_dataset.select(val_idx)
    train_dataset = train_dataset.select(train_idx)

    # Limit dataset size if in debug mode
    if hasattr(config.data, 'max_train_samples') and config.data.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), config.data.max_train_samples)))
        
    # Limit validation/test size if specified
    if hasattr(config.data, 'max_eval_samples') and config.data.max_eval_samples is not None:
        val_dataset = val_dataset.select(range(min(len(val_dataset), config.data.max_eval_samples)))
        dataset_dict["test"] = dataset_dict["test"].select(range(min(len(dataset_dict["test"]), config.data.max_eval_samples)))

    # rename the labels to match the model's expected input
    if hasattr(config.data, 'label_key'):
        train_dataset = train_dataset.rename_column(config.data.label_key, "labels")
        val_dataset = val_dataset.rename_column(config.data.label_key, "labels")
        dataset_dict["test"] = dataset_dict["test"].rename_column(config.data.label_key, "labels")

    # Verify label range
    all_labels = np.array(train_dataset['labels'])
    unique_labels = np.unique(all_labels)
    min_label = unique_labels.min()
    max_label = unique_labels.max()
    
    if min_label < 0 or max_label >= config.model.num_labels:
        raise ValueError(
            f"Labels must be in range [0, {config.model.num_labels-1}], "
            f"but found range [{min_label}, {max_label}]. "
            f"Unique labels found: {unique_labels}"
        )

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": dataset_dict["test"]
    })


def compute_metrics(eval_pred, label_names: Optional[Dict[int, str]] = None) -> Dict[str, float]:

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Create id2label mapping if provided, converting keys to regular Python ints
    if label_names:
        id2label = {int(k): v for k, v in label_names.items()}
    else:
        # Handle tuple case for logits
        logits_shape = logits[0].shape[-1] if isinstance(logits, tuple) else logits.shape[-1]
        id2label = {i: str(i) for i in range(logits_shape)}

    # Get all possible class labels (based on logits dimension)
    all_labels = list(range(logits[0].shape[-1] if isinstance(logits, tuple) else logits.shape[-1]))
    target_names = [id2label[i] for i in all_labels]

    if isinstance(labels, tuple):
        labels, cell_labels = labels
        cell_labels = cell_labels.flatten()
    if isinstance(logits, tuple):
        group_logits, cell_logits = logits
        group_predictions = np.argmax(group_logits, axis=-1)
        cell_predictions = np.argmax(cell_logits, axis=-1)

        # Get detailed classification report for group predictions
        report = classification_report(
            labels, 
            group_predictions,
            output_dict=True,
            labels=all_labels,  # Specify all possible labels
            target_names=target_names,
            zero_division=0
        )

        metrics = {
            "accuracy": (group_predictions == labels).mean(),
            "cell_accuracy": (cell_predictions == cell_labels).mean(),
            "classification_report": report
        }

        # Extract only scalar metrics for logging
        scalar_metrics = {
            "accuracy": metrics["accuracy"],
            "cell_accuracy": metrics["cell_accuracy"]
        }
        
        # Add per-class metrics in a wandb-friendly format
        # for label in report:
        #     if label not in ["accuracy", "macro avg", "weighted avg"]:
        #         scalar_metrics[f"f1_{label}"] = report[label]["f1-score"]
        #         scalar_metrics[f"precision_{label}"] = report[label]["precision"]
        #         scalar_metrics[f"recall_{label}"] = report[label]["recall"]

        # Add aggregate metrics
        scalar_metrics.update({
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        })

        return scalar_metrics

    else:
        predictions = np.argmax(logits, axis=-1)
        
        # Get detailed classification report
        report = classification_report(
            labels, 
            predictions,
            output_dict=True,
            labels=all_labels,  # Specify all possible labels
            target_names=target_names,
            zero_division=0
        )

        metrics = {
            "accuracy": (predictions == labels).mean(),
            "classification_report": report
        }

        # Extract only scalar metrics for logging
        scalar_metrics = {
            "accuracy": metrics["accuracy"]
        }
        
        # Add per-class metrics in a wandb-friendly format
        for label in report:
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                scalar_metrics[f"f1_{label}"] = report[label]["f1-score"]
                scalar_metrics[f"precision_{label}"] = report[label]["precision"]
                scalar_metrics[f"recall_{label}"] = report[label]["recall"]

        # Add aggregate metrics
        scalar_metrics.update({
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        })

        return scalar_metrics

def average_batch_location(data, batch_indices, all_indices):
    """Average locations for each batch of indices.
    """
    batch_locations = []
    
    for batch in batch_indices:
        # Create boolean mask for this batch
        mask = np.isin(all_indices, batch)
        # Get locations for this batch
        batch_locs = data[mask]
        # Average the locations if we found any
        if len(batch_locs) > 0:
            batch_locations.append(np.mean(batch_locs, axis=0))
            
    return np.stack(batch_locations) if batch_locations else np.array([])

def test_by_cell_type(dataset, trainer, type_col, min_N, output_dir, label_names=None, location_key="CCF_streamlines"):
    """
    Test the model by cell type.

    Parameters:
    dataset (Dataset): The dataset containing the data.
    trainer (Trainer): The trainer object with the predict method.
    type_col (str): The column name in the dataset that contains the cell types.
    min_N (int): The minimum number of samples required for each cell type to be included in the test.

    Returns:
    list: A list of numpy structured arrays containing the locations, labels, predictions, and indices for each cell type.
    """
    # Get the unique cell types from the specified column
    cell_type_counts = np.unique(dataset[type_col], return_counts=True)

    for cell_type, count in zip(*cell_type_counts):

        # Check if the number of rows is greater than min_N
        if count > min_N:
            # Filter the dataset for the current cell type
            filtered_dataset = dataset.filter(lambda x: x[type_col] == cell_type)
        
            # Run trainer.predict on the filtered dataset
            outputs = trainer.predict(filtered_dataset)
            outputs, indices = outputs # indices in the original dataset (not filtered dataset)
            # locations = average_batch_location(dataset, indices)

            labels = outputs.label_ids[0] if isinstance(outputs.label_ids, tuple) else outputs.label_ids
            predictions = np.argmax(outputs.predictions, axis=-1)

            # Add label names to output if available
            result = {
                "labels": labels,
                "predictions": predictions,
                "indices": indices,
                "label_names": label_names if label_names else None,
                "predicted_names": [label_names[str(p)] for p in predictions] if label_names else None,
                "true_names": [label_names[str(l)] for l in labels] if label_names else None
            }
            
            np.save(os.path.join(output_dir, f"cell_type_{cell_type.replace('/', '').replace(' ', '')}_test_brain_predictions.npy"), result)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))

    if 'single-cell' not in cfg.model.pretrained_type:
        assert cfg.data.group_size is not None, "group_size must be specified for multiformer models"
        assert cfg.training.remove_unused_columns is False, "remove_unused_columns must be False for multiformer models so that position info is available for grouping"
    else:
        assert cfg.training.remove_unused_columns, "remove_unused_columns must be True for single-cell models"

    if cfg.debug:
        # Set PyTorch debug options
        torch.autograd.set_detect_anomaly(True)
        
        # Limit dataset size for faster iteration
        cfg.data.max_train_samples = 100
        cfg.data.max_eval_samples = 100
        
        # Disable wandb in debug mode
        cfg.training.report_to = None

    # Set random seeds
    set_seed(cfg.seed)
    
    # Initialize wandb
    setup_wandb(cfg)

    # Disable caching for datasets
    disable_caching()
    
    # Load pre-tokenized datasets
    dataset_dict = load_from_disk(cfg.data.dataset_path)
    datasets = prepare_datasets(dataset_dict, cfg)
    print(f"Loaded datasets: {datasets}")
    # Load class weights first
    class_weights = load_class_weights(cfg)
    
    # Create model with class weights
    model = create_model(cfg, class_weights)
    
    # Update trainer parameters to include label names
    trainer_params = {
        "model": model,
        "args": TrainingArguments(**cfg.training),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "compute_metrics": lambda pred: compute_metrics(pred, cfg.data.label_names),
    }

    # Initialize trainer
    if "single-cell" in cfg.model.pretrained_type:
        trainer = Trainer(**trainer_params)
    else:
        trainer = MultiformerTrainer(
            **trainer_params,
            spatial_group_size=cfg.data.group_size,
            spatial_label_key="labels",
            coordinate_key='CCF_streamlines',
            relative_positions=cfg.model.relative_positions.enabled,
            absolute_Z=cfg.model.relative_positions.absolute_Z,
            hex_scaling=cfg.data.sampling.hex_scaling,
            reflect_points=cfg.data.sampling.reflect_points,
            use_train_hex_validity=cfg.data.sampling.use_train_hex_validity,
            sampling_strategy=cfg.data.sampling.strategy
        )
    # Train
    if cfg.training.num_train_epochs > 0:
        train_result = trainer.train()
        trainer.save_model()
        trainer.save_state()
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Test and validation
    if cfg.run_test_set:
        for data_key in ["test", "validation"]:
            # Initialize tensors to store results across epochs
            all_locations = []
            all_predictions = []
            all_labels = []
            all_indices = []
            all_single_cell_predictions = []
            all_single_cell_labels = []

            for epoch in range(cfg.num_predict_epochs):
                trainer.accelerator.gradient_state._reset_state() # Fixes an odd bug where the trainer thinks the dataloader is finished so truncates further batches incorrectly

                random_seed = cfg.seed + torch.initial_seed()

                outputs = trainer.predict(
                    datasets[data_key],
                    metric_key_prefix=data_key,
                    seed=random_seed
                )

                if "single-cell" in cfg.model.pretrained_type:
                    locations = datasets[data_key]["CCF_streamlines"]
                    indices = np.arange(len(datasets[data_key]))
                else:
                    outputs, indices = outputs
                    locations = average_batch_location(np.array(datasets[data_key]["CCF_streamlines"]), indices, datasets[data_key]["uuid"])

                if cfg.model.single_cell_loss_after_set:
                    predictions = np.argmax(outputs.predictions[0], axis=-1)
                    single_cell_predictions = np.argmax(outputs.predictions[1], axis=-1)
                else:
                    predictions = np.argmax(outputs.predictions, axis=-1)
                    single_cell_predictions = None

                labels = outputs.label_ids[0] if isinstance(outputs.label_ids, tuple) else outputs.label_ids
                single_cell_labels = outputs.label_ids[1] if isinstance(outputs.label_ids, tuple) else None

                # Append results from this epoch
                all_locations.append(locations)
                all_predictions.append(predictions)
                all_labels.append(labels)
                all_indices.append(indices)
                if single_cell_predictions is not None:
                    all_single_cell_predictions.append(single_cell_predictions)
                if single_cell_labels is not None:
                    all_single_cell_labels.append(single_cell_labels)

            # Concatenate results from all epochs
            locations = np.concatenate(all_locations, axis=0)
            predictions = np.concatenate(all_predictions, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            indices = np.concatenate(all_indices, axis=0)
            single_cell_predictions = np.concatenate(all_single_cell_predictions, axis=0) if all_single_cell_predictions else None
            single_cell_labels = np.concatenate(all_single_cell_labels, axis=0) if all_single_cell_labels else None
                
            # Include label names in output
            output_dict = {
                "locations": locations,
                "labels": labels,
                "predictions": predictions,
                "indices": indices,
                "label_names": cfg.data.label_names,
                "single_cell_labels": single_cell_labels,
                "single_cell_predictions": single_cell_predictions,
            }
            # Log metrics
            trainer.log_metrics(data_key, outputs.metrics)
            # save to disk. 
            np.save(os.path.join(cfg.output_dir, f"{data_key}_brain_predictions.npy"), output_dict)

    if cfg.test_by_cell_type:
        # Test by cell type
        test_by_cell_type(
            datasets["test"], 
            trainer, 
            "H2_type", 
            cfg.min_N, 
            cfg.output_dir,
            label_names=cfg.data.label_names
        )
    
    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()