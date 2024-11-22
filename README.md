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

### Running the Project

To run the project with the default configuration, use the following command:

```sh
python main.py
```

### Customizing Configurations

You can override the default configurations by specifying them in the command line:

```sh
python main.py +training.num_train_epochs=10 +training.learning_rate=5e-5
```

### Logging and Monitoring

The project uses Weights & Biases (wandb) for logging and monitoring. Ensure you have set up your wandb account and configured it properly.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainers.

## To do

- [ ] Re-create dataset (without attention masks) for correct test set
- [ ] What does the prediction map look like for the whole brain in the test set?
- [ ] What does accuracy look like with group size?
- [ ] Can we sweep over key model parameters?
- [ ] What do brain maps look like with class_weights balanced accuracy?
- [ ] What areas are confused with which other areas for the class_weights balanced model?
- [ ] Are certain brain areas more difficult to predict than others?
- [ ] Are certain cell types more informative than others?

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