# %%
# import pandas as pd
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
from sleap_nn.train import run_training

# %%
# MLFlow parameters
# Database location
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # or a remote server

# An experiment is a group of runs
mlflow.set_experiment("sleap-KK-dome")

# %%
# Sleap training job directory uzipped
# TODO: can I simply passed the zipped? or unzip in tmp and then
# attach the zip file as artifact
sleap_training_job = "/home/sminano/swc/project_sleap_dome/run_TEST"

# %%
# Call autolog
mlflow.pytorch.autolog(
    log_models=True,
    checkpoint=True,
    # logs best/last .ckpt to MLflow
    log_every_n_epoch=1,
)

# %%
# Load configs
# TODO: when is jobs.yaml used? I think never
list_yaml_files = [
    p for p in list(Path(sleap_training_job).glob("*.yaml")) if p.name != "jobs.yaml"
]


# %%
# Run training for each config

for config_yaml in list_yaml_files:
    # Load config
    config = OmegaConf.load(config_yaml) # output is an Omega DictConfig

    # Start mlflow run
    with mlflow.start_run(run_name=config.trainer_config.run_name):

            # Track the inputs
            mlflow.log_dict(OmegaConf.to_container(config, resolve=True), "sleap_config.yaml")

            # Track train and val datasets
            for p in config.data_config.train_labels_path:
                mlflow.log_artifact(p, artifact_path="datasets/train")
            for p in config.data_config.val_labels_path:
                mlflow.log_artifact(p, artifact_path="datasets/val")

            # Train — autolog should handle metrics/params/model
            run_training(config=config)

            # # Optional: also archive the final ckpt dir
            # mlflow.log_artifacts(f"{config.trainer_config.ckpt_dir}/{config.trainer_config.run_name}",
            #                      artifact_path="sleap_run")


# %%
# # Parse the csv after training

# run_dir = Path(config.trainer_config.ckpt_dir) / config.trainer_config.run_name
# for csv in run_dir.glob("*.csv"):
#     df = pd.read_csv(csv)
#     step_col = "epoch" if "epoch" in df.columns else "step"
#     for _, row in df.iterrows():
#         step = int(row[step_col])
#         for col, val in row.items():
#             if col == step_col or pd.isna(val):
#                 continue
#             try:
#                 mlflow.log_metric(col, float(val), step=step)
#             except (TypeError, ValueError):
#                 pass  # skip non-numeric columns


if __name__ == "__main__":
    