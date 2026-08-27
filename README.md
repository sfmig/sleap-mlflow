# Track your SLEAP models using MLflow

[MLflow](https://mlflow.org/docs/latest/ml/) is a model and experiment tracking framework that can be used with the pose estimation package [SLEAP](https://sleap.ai/).


This workflow assumes users define the model and training configuration from the SLEAP GUI, and generate a new [training job package](https://docs.sleap.ai/latest/notebooks/Training_and_inference_using_Google_Drive/?h=training+job+package#create-and-export-the-training-job-package) (i.e. a new `slp.training_job.zip` file) everytime from the "Run Training.." dialog under the "Predict" menu. For clarity, we recommend adding the run-name defined in the SLEAP GUI to the exported training job package (e.g. rename it to `<sleap-run-name>.slp.training_job.zip`)

The run-name defined in the SLEAP GUI will be used to track the same runs in MLflow.

All the generated artifacts remain in the SLEAP-generated directories, but are crosslinked to the MLflow database. Only metadata is tracked by MLflow

## Suggested directory structure
We suggest the following structure to organise your trained models:
```
- sleap-runs # or sleap_training_packages?
    - foo.slp.training_job.zip ---> will be extracted to a dir called <sleap-run-name>/
- mlruns
    - ....
mlflow.db
mlflow_train.py
```

## Pre-requisites

Install uv (TODO: specify version that supports torch-backend solver)


## Steps
You can do this locally, in an interactive node in the cluster, or in a batch job (TODO: include script).

0. Click the green button on the top right that says "Use this template". This will create a copy of this repository under your own GitHub account, which you can then clone and customise.

1. Git clone the repo added to your account locally

2. In the SLEAP GUI, export your training job package as a zip and save it under the `sleap-runs` directory in the repo.

3. From the repo root directory, launch a training job with mlflow tracking by running: 
```
bash run_mlflow_training.sh \
    /path/to/exported/sleap/training/job.zip \
    --mlflow-experiment-name mlflow-expt-name \ # optional, to group runs together  (e.g. data-augmentation-study)
```
This will
- unzip the training job package and place its contents into a directory `<SLEAP-RUN-NAME>`, named after the run name
defined in the SLEAP GUI when creating the training job package.
- launch training and track its results with MLflow. It will create an `mlruns` folder if it does not exist and an `mlflow.db` database file to keep track of it. The script also installs any required dependencies. 


4. To monitor the completed and ongoing jobs: launch the mlflow server. You may want to do this in a separate terminal (just once per session):
```
uvx --python 3.13 'mlflow>=3.13,<4' server --port 5005
```
Click "Model training" tab on the left-hand side, then "Experiments".

Or to jump to the experiments tab directly (ensure the "Model training" tab is selected)

```
# get a free port
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

# run mlflow dashboard in that port
uvx 'mlflow>=3.13,<4' server --backend-store-uri sqlite:///mlflow.db --port "$PORT" & \
    sleep 3 && xdg-open "http://localhost:$PORT/#/experiments"
```

## Navigating the UI
* Metrics vs parameters

## Tips
* You can use the cloned repo to keep track of any edits you wish to make to the launching script. 
* The commit hash of the launching script is logged to MLflow
* Do not delete
    - the mlruns folder or the mlflow.db file: they define the MLflow database of metadata to track your trained models
    - the sleap-runs subdirectories: they contain the actual trained models

## Notes
- torch-backend = "auto" detects CUDA via the driver, so on an exotic setup you can override with UV_TORCH_BACKEND=cu128 uv run ... or by hardcoding the value instead of "auto".
- Default database path and absolute path caveats


## References
- https://mlflow.org/docs/latest/ml/getting-started/quickstart/
- https://mlflow.org/docs/latest/ml/tracking/