import json
import logging
import os
from pathlib import Path

import dagshub
import mlflow
from mlflow import MlflowClient


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

def configure_mlflow(root_path: Path) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    use_dagshub = os.getenv("USE_DAGSHUB", "0").lower() in {"1", "true", "yes", "on"}

    if use_dagshub:
        try:
            dagshub.init(
                repo_owner="jivanshs51",
                repo_name="swiggy-delivery-time-prediction",
                mlflow=True,
            )
            mlflow.set_tracking_uri(
                "https://dagshub.com/jivanshs51/swiggy-delivery-time-prediction.mlflow"
            )
            logger.info("Configured MLflow tracking with DagsHub")
            return
        except Exception as exc:
            logger.warning("DagsHub initialization failed, falling back to local MLflow: %s", exc)

    tracking_dir = root_path / "mlruns"
    tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    logger.info("Configured MLflow tracking locally at %s", tracking_dir)


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    configure_mlflow(root_path)

    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    # model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"
    
    
    # register the model
    model_version = mlflow.register_model(model_uri=model_registry_path,
                                          name=model_name)
    
    
    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
    