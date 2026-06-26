# Organise your SLEAP models using MLflow

MLflow is a model and experiment tracking framework that can be used with SLEAP....

I assume people change model config from the GUI --- generate a new training package.

One training job package = one SLEAP run name = one MLflow run name

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
1. Git clone repo locally

2. In the SLEAP GUI, export your training job package as a zip and save it under the `sleap-runs` directory in the repo.

3. Run the script using `uv` (it will install dependencies)
```
uv run mlflow_train.py \
    /path/to/exported/sleap/training/job.zip \
    --mlflow-experiment-name mlflow-expt-name \ # optional, to group runs together  (e.g. data-augmentation-study)
```
This will
- unzip the training job package and place its contents into a directory `<SLEAP-RUN-NAME>`, named after the run name
selected in the SLEAP GUI when selecting the training job package.
- launch training and track its results with mlflow


3. Do just once per session: launch the mlflow server
```
uvx --python 3.13 'mlflow>=3.13,<4' server --port 5005
```

Or to jump to the experiments tab

```
uvx 'mlflow>=3.13,<4' server --backend-store-uri sqlite:///mlflow.db --port 5005 & \
    sleep 3 && xdg-open "http://localhost:5005/#/experiments"
```

## Tips


## Notes
-  The one caveat: torch-backend = "auto" detects CUDA via the driver, so on an exotic setup you can override with UV_TORCH_BACKEND=cu128 uv run ... or by hardcoding the value instead of "auto".
- If port is busy
- Default database path and absolute path caveats


## References
- https://mlflow.org/docs/latest/ml/tracking/