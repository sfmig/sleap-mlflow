"""Track SLEAP training runs with MLflow.


Run the SLEAP training job using the following command:
    uv run mlflow_train.py /path/to/exported/sleap/training/job.zip \
        --mlflow-experiment-name sleap-KK-dome \
        --mlflow-tracking-uri sqlite:///mlflow.db

Then visualise the progress in the MLflow dashboard. The server only needs mlflow
itself, so run it in an ephemeral uv environment (no project install required):
    uvx --python 3.13 'mlflow>=3.13,<4' server --backend-store-uri sqlite:///mlflow.db --port 5000

Or to launch the server and jump straight to the experiments tab:
    uvx --python 3.13 'mlflow>=3.13,<4' server --backend-store-uri sqlite:///mlflow.db --port 5000 & \
    sleep 3 && xdg-open "http://localhost:5000/#/experiments"


NOTES:
- `uvx 'mlflow>=3.13,<4' ...` is shorthand for `uv tool run ...`: uv builds a
  one-off ephemeral environment satisfying that constraint and runs the `mlflow`
  entry point in it. The constraint is kept on the same major version as the
  `mlflow` dependency above so the server's DB schema matches the one written by
  training. `--python 3.13` matches the script's requires-python: without it uvx
  defaults to the newest interpreter (3.14), and mlflow 3.14.0 fails to import on
  Python 3.14 (it imports `Traversable` from `importlib.abc`, removed in 3.14).
  Reproducible training versions come from the lockfile
  (mlflow_train.py.lock); refresh it with `uv lock --script mlflow_train.py --upgrade`.
- For a SQLite tracking URI, the path after sqlite:/// is taken relative to the current
working directiory. To give an absolute path, you add a leading slash for the root — so you end up with four slashes total:
    --mlflow-tracking-uri sqlite:////home/sminano/swc/project_sleap_dome/mlflow.db

"""

# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "mlflow>=3.13,<4",
#     "omegaconf",
#     "sleap-nn>=0.2.0",
# ]
#
# # Pick the torch wheel matching the local hardware automatically: uv detects
# # the CUDA driver (or lack of one) at install time and pulls the right build
# # (cpu / cu128 / cu129 / ...). No manual index pinning needed.
# [tool.uv]
# torch-backend = "auto"
# ///

import argparse
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

import mlflow
from omegaconf import OmegaConf
from sleap_nn.train import run_training

# SLEAP encodes run names as "<prefix>.<model_type>.<suffix>".
# These are the model types, matching the head-config keys in
# sleap_nn.config.model_config.HeadConfig (sleap-nn 0.2.0).
# TODO: test with other model types
SLEAP_MODEL_TYPES = (
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
)
TOP_DOWN_ORDER = ["centroid", "centered_instance"]


def run_name_from_train_script(text):
    """Derive the SLEAP run-name prefix from a train-script.sh's contents.

    SLEAP encodes run names as "<prefix>.<model_type>.<suffix>". The GUI-set
    prefix is the part before the model-type string, which is what we use
    to name the run directory.

    This function parses every ``trainer_config.run_name="..."`` and returns
    the shared prefix that precedes the model-type string.
    """
    # SLEAP quotes the value with either single or double quotes.
    names = re.findall(r"""trainer_config\.run_name=['"]([^'"]+)['"]""", text)
    prefixes = set()
    for name in names:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part in SLEAP_MODEL_TYPES:
                prefixes.add(".".join(parts[:i]))
                break
        else:
            # No known model token found; fall back to the whole name.
            prefixes.add(name)
    if not prefixes:
        raise ValueError("No trainer_config.run_name found in train-script.sh")
    if len(prefixes) > 1:
        raise ValueError(
            f"train-script.sh has more than one run-name prefix: {sorted(prefixes)}"
        )
    return prefixes.pop()


def get_sleap_run_name_from_zip(sleap_training_job_zip):
    """Read the SLEAP run-name prefix from an training job .zip.

    The prefix is derived from the bundled train-script.sh. A .zip is required;
    anything else raises an error.
    """
    src = Path(sleap_training_job_zip)
    if src.suffix != ".zip":
        raise ValueError(f"Expected an exported SLEAP job .zip, got: {src}")

    with zipfile.ZipFile(src) as zf:
        member = next((n for n in zf.namelist() if n.endswith("train-script.sh")), None)
        if member is None:
            raise FileNotFoundError(
                f"No train-script.sh in {src}; cannot derive the run name."
            )
        return run_name_from_train_script(zf.read(member).decode())


def create_sleap_runs_subdir(sleap_training_job_zip, run_name):
    """Resolve the directory under `sleap-runs` for the given SLEAP training job .zip.

    The unzipped directory is saved under `sleap-runs` and renamed to the SLEAP
    run name (passed in, derived from the bundled train-script.sh).
    """
    src = Path(sleap_training_job_zip)
    with zipfile.ZipFile(src) as zf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path("sleap-runs") / f"{run_name}_{timestamp}"
        print(f"Extracting {src} -> {dest}")
        zf.extractall(dest)
    return dest.resolve()


def flatten_params(config, parent_key=""):
    """Flatten a (possibly nested) config into slash-separated scalar params.

    MLflow params must be flat scalars, so nested dicts are collapsed with
    slash-separated keys (e.g. ``trainer_config/optimizer/lr``) and lists are
    indexed (e.g. ``data_config/train_labels_path/0``). ``None`` values are
    kept; MLflow stores params as strings, so they appear as ``"None"``.
    """
    items = {}
    for key, value in config.items():
        full_key = f"{parent_key}/{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_params(value, full_key))
        elif isinstance(value, (list, tuple)):
            for i, elem in enumerate(value):
                items.update(flatten_params({str(i): elem}, full_key))
        else:
            items[full_key] = value
    return items


def train_and_log(config_yaml, sleap_job_dir, nested=False):
    """Launch a SLEAP training with MLflow logging.

    `nested=True` attaches the run as a child of the currently active run
    (every model run is nested under a parent named after the SLEAP job dir).
    """
    # Read omega config
    config = OmegaConf.load(config_yaml)  # output is an Omega DictConfig

    with mlflow.start_run(run_name=config.trainer_config.run_name, nested=nested):
        # Track YAML config as artifact
        # TODO: review, this duplicates YAML file
        mlflow.log_dict(
            OmegaConf.to_container(config, resolve=True),
            f"sleap_{config_yaml.stem}.yaml",
        )

        # Log config as params to filter, groupby and show in columns
        # (flatten nested dicts)
        mlflow.log_params(
            flatten_params(OmegaConf.to_container(config, resolve=True))
        )

        # # Log artifacts ------------
        # # Log train datasets as artifact
        # # TODO: review; save path to labels_gt.train.0.slp files instead?
        # for p in config.data_config.train_labels_path:
        #     mlflow.log_artifact(sleap_job_dir / p, artifact_path="datasets/train")

        # # Log val datasets as artifact
        # # TODO: review; save path to labels_gt.val.0.slp files instead?
        # if config.data_config.val_labels_path:
        #     for p in config.data_config.val_labels_path:
        #         mlflow.log_artifact(sleap_job_dir / p, artifact_path="datasets/val")

        # Train — autolog should handle metrics/params/model
        run_training(config)

        # # Log paths to outputs (train and val label files, metrics etc?)
        # mlflow.log_params(
        #     {}
        # )


def main(sleap_training_job_zip, mlflow_experiment_name, mlflow_tracking_uri):
    # --------------
    # MLFlow parameters
    # Database location
    mlflow.set_tracking_uri(mlflow_tracking_uri)  # or a remote server

    # An experiment is a group of runs
    mlflow.set_experiment(mlflow_experiment_name)

    # Derive the SLEAP run-name prefix from the job .zip; it names the extracted
    # dir and (for top-down) the parent MLflow run.
    run_name = get_sleap_run_name_from_zip(sleap_training_job_zip)

    # Resolve to absolute so it stays valid after we chdir into it below.
    # Extracts the exported job .zip to sleap-runs/<NAME>.
    sleap_job_dir = create_sleap_runs_subdir(sleap_training_job_zip, run_name)

    # --------------------------------
    # Call autolog
    mlflow.pytorch.autolog(
        # log_models=True,
        # checkpoint=True, --- # logs best/last .ckpt to MLflow by default
        log_every_n_epoch=1,
        checkpoint=False,  # --- since sleap-nn saves it already
        # checkpoint_save_best_only=False, -- save every ckpt
        # checkpoint_monitor="val/loss",   --- use sleap-nn's actual val-loss key
    )

    # --------------------------------
    # Get list of YAML configs
    # (one for bottom-up, two for top-down)
    # (jobs.yaml is ignored)
    list_yaml_files = [
        p for p in list(sleap_job_dir.glob("*.yaml")) if p.name != "jobs.yaml"
    ]
    if not list_yaml_files:
        raise FileNotFoundError(
            f"No training config *.yaml files found in {sleap_job_dir}. The job .zip "
            "is expected to be flat (configs at the top level); check it isn't wrapped "
            "in a top-level folder."
        )
    if len(list_yaml_files) > 2:
        raise ValueError(
            f"Expected 1 (bottom-up) or 2 (top-down) training config *.yaml files in "
            f"{sleap_job_dir}, found {len(list_yaml_files)}: "
            f"{sorted(p.name for p in list_yaml_files)}."
        )

    # Order by model type (centroid before centered_instance) so top-down models
    # train and appear in MLflow in pipeline order. The .yaml stem is the model
    # type, e.g. centroid.yaml / centered_instance.yaml.
    list_yaml_files.sort(
        key=lambda p: TOP_DOWN_ORDER.index(p.stem)
        if p.stem in TOP_DOWN_ORDER
        else len(TOP_DOWN_ORDER)
    )

    # --------------------------------
    # Run training for each config

    # Run from the job dir
    # this is so that the relative label paths in the configs resolves
    os.chdir(sleap_job_dir)

    # Group all models (one for bottom-up, two for top-down) as child runs
    # under a single parent run, named after the timestamped job dir.
    with mlflow.start_run(run_name=Path(sleap_job_dir).stem):
        for config_yaml in list_yaml_files:
            train_and_log(config_yaml, sleap_job_dir, nested=True)


# --------------------------------
# # Parse the csv after training to log loss

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
        "sleap_training_job_zip",
        help=(
            "Path to the exported SLEAP training job (.zip). It is extracted to "
            "sleap-runs/<NAME>, where NAME is derived from train-script.sh."
        ),
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
        sleap_training_job_zip=args.sleap_training_job_zip,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
