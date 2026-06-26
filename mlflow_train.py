"""Track SLEAP training runs with MLflow.


Run the SLEAP training job using the following command:
    python mlflow_train.py /path/to/unzipped/sleap/training/job/dir \
        --mlflow-experiment-name sleap-KK-dome \
        --mlflow-tracking-uri sqlite:///mlflow.db

Then visualise the progress in the MLflow dashboard
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Or to go to the experiments tab directly:
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000 & \
    sleep 3 && xdg-open "http://localhost:5000/#/experiments"


NOTES:
- For a SQLite tracking URI, the path after sqlite:/// is taken relative to the current
working directiory. To give an absolute path, you add a leading slash for the root — so you end up with four slashes total:
    --mlflow-tracking-uri sqlite:////home/sminano/swc/project_sleap_dome/mlflow.db

"""

import argparse
import os
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
from sleap_nn.train import run_training


def main(sleap_training_job, mlflow_experiment_name, mlflow_tracking_uri):
    # --------------
    # MLFlow parameters
    # Database location
    mlflow.set_tracking_uri(mlflow_tracking_uri)  # or a remote server

    # An experiment is a group of runs
    mlflow.set_experiment(mlflow_experiment_name)

    # Resolve to absolute so it stays valid after we chdir into it below
    sleap_job_dir = Path(sleap_training_job).resolve()

    # --------------------------------
    # Call autolog
    mlflow.pytorch.autolog(
        # log_models=True, 
        # checkpoint=True,
        # logs best/last .ckpt to MLflow
        log_every_n_epoch=1,
    )

    # --------------------------------
    # Load configs
    # (one for bottom-up, two for top-down)
    # (jobs.yaml is ignored)
    list_yaml_files = [
        p for p in list(sleap_job_dir.glob("*.yaml")) if p.name != "jobs.yaml"
    ]

    # --------------------------------
    # Run training for each config

    # Run from the job dir
    # this is so that the relative label paths in the configs resolves
    os.chdir(sleap_job_dir)

    for config_yaml in list_yaml_files:
        # Load config
        config = OmegaConf.load(config_yaml)  # output is an Omega DictConfig

        # Start mlflow run
        # REVIEW: use same run_name as set in SLEAP config / GUI
        with mlflow.start_run(run_name=config.trainer_config.run_name):
            # Track the inputs
            mlflow.log_dict(
                OmegaConf.to_container(config, resolve=True),
                f"sleap_{config_yaml.name}.yaml",
            )

            # Log as hyperparameters
            # mlflow.log_params(params)

            # Log train datasets as artifact
            for p in config.data_config.train_labels_path:
                mlflow.log_artifact(sleap_job_dir / p, artifact_path="datasets/train")

            # Log val datasets as artifact
            if config.data_config.val_labels_path:
                for p in config.data_config.val_labels_path:
                    mlflow.log_artifact(sleap_job_dir / p, artifact_path="datasets/val")

            # Train — autolog should handle metrics/params/model
            run_training(config)

            # Log the model
            #model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")


            # mlflow.log_metric("accuracy", accuracy)

            # Optional: Set a tag that we can use to remind ourselves what this run was for
            # mlflow.set_tag("Training Info", "Basic LR model for iris data")
            
            # # Optional: also archive the final ckpt dir
            # mlflow.log_artifacts(f"{config.trainer_config.ckpt_dir}/{config.trainer_config.run_name}",
            #                      artifact_path="sleap_run")


# --------------------------------
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


# --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SLEAP training and log to MLflow.",
    )
    parser.add_argument(
        "sleap_training_job",
        help="Path to the unzipped SLEAP training job directory.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        dest="mlflow_experiment_name",
        default="DEFAULT",
        help="MLflow experiment name (group of runs).",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        dest="mlflow_tracking_uri",
        default="sqlite:///mlflow.db",
        help=(
            "MLflow tracking URI (database location)."
            "By default, it is created under the directory"
            "from which the script is launched"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        sleap_training_job=args.sleap_training_job,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
