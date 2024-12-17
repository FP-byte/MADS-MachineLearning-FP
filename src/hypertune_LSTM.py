from pathlib import Path
from typing import Dict

import ray
import torch
from loguru import logger
from ray import tune
from hypertuner import Hypertuner
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import PaddedPreprocessor

def hypertune_LSTM():

    #test with LSTM
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")

    tune_dir = Path("models/ray").resolve()

    settings_hypertuner = {
        "NUM_SAMPLES": 10,
        "MAX_EPOCHS": 10,
        "device": "cpu",
        "accuracy": metrics.Accuracy(),
        "reporttypes": [ReportTypes.RAY],
    }

    config = {
        "dataset_type": "GESTURES",  # Can be GESTURES, ANOTHER_DATASET, etc.
        "preprocessor": PaddedPreprocessor,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "batch_size": 32,  # Batch size specific to the dataset
        "hidden_size": tune.randint(400, 800),
        #"dropout": tune.uniform(0.0, 0.3),
        "dropout":0.0,
        "num_layers": tune.randint(1, 4),
        #"num_layers":3,
        "model_type": "LSTM",  # Specify the model type
    }

    hypertuner = Hypertuner(settings_hypertuner, config)

    analysis = tune.run(
        hypertuner.train,
        config=config,
        metric="Accuracy",
        mode="max",
        progress_reporter=hypertuner.reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=hypertuner.NUM_SAMPLES,
        search_alg=hypertuner.search,
        scheduler=hypertuner.scheduler,
        verbose=1,
        trial_dirname_creator=hypertuner.shorten_trial_dirname,
    )

    ray.shutdown()

if __name__ == "__main__":
    hypertune_LSTM()