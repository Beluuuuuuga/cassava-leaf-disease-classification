import os
from mlflow import log_metric, log_param, log_artifact, set_experiment
from omegaconf import DictConfig, OmegaConf
import hydra

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    set_experiment("MLMAN-1")
    log_param("param1", 5)

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)

    # Log an artifact (output file)
    with open("output.txt", "w") as f:
        f.write("Hello world!")
    log_artifact("output.txt")