# Organise your SLEAP models using MLflow

MLflow is a model and experiment tracking framework that can be used with SLEAP....

I assume people change model config from the GUI --- generate a new training package.

unzipping one SLEAP training job package => one SLEAP run name directory with timestamp => one MLflow run name

## Suggested directory structure
```
# A training job package is a set of annotations + model config

- sleap-runs # or sleap_training_packages?
    - foo.slp.training_job.zip ---> will be extracted to a dir called <sleap-run-name>/
- mlruns
    - ....
mlflow.db
mlflow_train.py

```

## Pre-requisites

Install uv (version that supports torch-backend solver)


## Steps
You can do this locally, in an interactive node in the cluster, or in a batch job (include script).

0. Click the green button on the top right that says "Use this template"

1. Git clone the repo added to your account locally

2. In the SLEAP GUI, export your training job package as a zip and save it under the `sleap-runs` directory in the repo.

3. From the repo root directory, launch a training job with mlflow tracking by running: 
```
uv run mlflow_train.py \
    /path/to/exported/sleap/training/job.zip \
    --mlflow-experiment-name mlflow-expt-name \ # optional, to group runs together  (e.g. data-augmentation-study)
```
This will
- unzip the training job package and place its contents into a directory `<SLEAP-RUN-NAME>`, named after the run name
selected in the SLEAP GUI when selecting the training job package.
- launch training and track its results with mlflow (it will install any required dependencies)


3. Just once per session: launch the mlflow server to visualise the tracked results
```
uvx --python 3.13 'mlflow>=3.13,<4' server --port 5005
```
Click Model training tab on the left-hand side, then Experiments.

Or to jump to the experiments tab directly

```
uvx 'mlflow>=3.13,<4' server --backend-store-uri sqlite:///mlflow.db --port 5005 & \
    sleep 3 && xdg-open "http://localhost:5005/#/experiments"
```

## Navigating the UI
* Metrics vs parameters

## Tips
* You can use the cloned repo to keep track of changes to the launching script. The commit of the launching script is logged to MLflow

## Notes
- The one caveat: torch-backend = "auto" detects CUDA via the driver, so on an exotic setup you can override with UV_TORCH_BACKEND=cu128 uv run ... or by hardcoding the value instead of "auto".
- If port is busy
- Default database path and absolute path caveats


## References
- https://mlflow.org/docs/latest/ml/getting-started/quickstart/
- https://mlflow.org/docs/latest/ml/tracking/