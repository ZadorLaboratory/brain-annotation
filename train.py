import os
from typing import Dict, Optional

import hydra
import wandb
from hydra.core.config_store import ConfigStore
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
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertForSequenceClassification, Trainer
from transformers.trainer_callback import TrainerCallback


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
    """
    Create model based on pretraining configuration.
    
    Args:
        config.model.pretrained_type: One of ["none", "bert_only", "full"]
        config.model.bert_path_or_name: Either:
            - Hub model name (e.g. "bert-base-uncased") 
            - Path to pretrained BERT weights
        config.model.model_path: Path to full pretrained hierarchical model
    """
    if config.model.pretrained_type == "full":
        if not config.model.model_path:
            raise ValueError("model_path must be specified when pretrained_type is 'full'")
            
        # Load full pretrained hierarchical model
        model = HierarchicalBert.from_pretrained(
            config.model.model_path,
            num_labels=config.model.num_labels,
            class_weights=class_weights,
            pool_weight=config.model.pool_weight,
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
            class_weights=class_weights,
            pool_weight=config.model.pool_weight,
            single_cell_augmentation=config.single_cell_augmentation,
            detach_bert_embeddings=config.model.detach_bert_embeddings,
            **(config.model.get("bert_params", {}) if config.model.pretrained_type == "none" else {})
        )
        
        # Initialize new model
        model = HierarchicalBert(model_config)
        
        # If bert_only, load pretrained BERT weights
        if config.model.pretrained_type == "bert_only":
            pretrained_bert = BertModel.from_pretrained(config.model.bert_path_or_name)
            model.bert.load_state_dict(pretrained_bert.state_dict())

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

    # reindex train_dataset?
    # train_dataset = train_dataset.flatten_indices()
    # val_dataset = val_dataset.flatten_indices()

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": dataset_dict["test"]
    })


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    if isinstance(labels, tuple):
        labels = labels[0]

    metrics = {
        "accuracy": (predictions == labels).mean(),
    }
    
    # Updated wandb confusion matrix call
    # wandb.log({
    #     "confusion_matrix": wandb.plot.confusion_matrix(
    #         preds=predictions,
    #         y_true=labels,
    #         class_names=[f"class_{i}" for i in range(logits.shape[1])]  # number of classes from logits
    #     )
    # })
    
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))

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
    
    # Initialize trainer
    if "single-cell" in cfg.model.pretrained_type:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**cfg.training),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=compute_metrics,
        )
    else:
        trainer = MultiformerTrainer(
            model=model,
            args=TrainingArguments(**cfg.training),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=compute_metrics,
            spatial_group_size=cfg.data.group_size,
            spatial_label_key="labels",
        )
        
    # Train
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
    
    # Test
    test_metrics = trainer.evaluate(
        datasets["test"],
        metric_key_prefix="test"
    )
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()