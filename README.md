# Transcriptomic-based Brain Area Annotation Using Transformers

This project uses Hydra for configuration management and includes various settings for training, evaluation, and logging. This README provides all the necessary information to get started, including installation instructions, configuration details, and usage examples.

## Installation

To install the required dependencies, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. (If not already installed) **Install micromamba**
    ```
    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
    ```

2. **Create a virtual environment:**
    ```sh
    source create_env.sh
    ```


## Configuration

The project uses Hydra for configuration management. The main configuration file is located at 

config.yaml

. Below is an overview of the configuration structure:

### Main Configuration (

config.yaml

)

```yaml
defaults:
  - _self_
  - model: default
  - data: default
  - training: default
  - wandb: default
  - override hydra/hydra_logging: default
  - override hydra/job_logging: default
seed: 42
debug: false
single_cell_augmentation: true
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.override_dirname}
```

### Training Configuration (

default.yaml

)

```yaml
output_dir: ${hydra:runtime.output_dir}/model
run_name: ${hydra:runtime.output_dir}/model
num_train_epochs: 4
per_device_train_batch_size: 1536  # note this will be divided by ${data.group_size}
per_device_eval_batch_size: 1536
warmup_ratio: 0.1
learning_rate: 1e-4
weight_decay: 0.001
logging_steps: 10
eval_steps: 50
save_steps: 1000
gradient_accumulation_steps: 1
fp16: true
eval_strategy: "steps"
save_strategy: "steps"
load_best_model_at_end: true
metric_for_best_model: "eval_accuracy"
report_to: "wandb"
remove_unused_columns: false
dataloader_prefetch_factor: 4
dataloader_num_workers: 48
```

## Usage

### Training a model



### Predicting with a trained model

(use train.py but with `cfg.training.num_train_epochs = 0`. To specify the model use `config.model.pretrained_type = "full"` and `config.model.bert_path_or_name`.)

## License

This project is licensed under the MIT License. See the LICENSE file for details.



### Findings

 - Additionally averaging the predictions of the set transformer with the average logits of the single-cell predictions does NOT help
 - ^ This is true for all weighted average ratios between the two models
 - ^ Detaching the Bert embeddings before learning the set transformer does NOT help
 - Learning is not stable with learning rates above 1e-4
 - About 15 epochs are needed to reach the best performance
 - Don't compare convergence rates across models trained for different amounts of epochs due to the warmup schedule
 - 

### Findings worth a nice plot

 - Predictions with only the set transformer vs. averaging the single-cell weights.