from pathlib import Path
from typing import Dict

import ray
import torch
from loguru import logger
from ray import tune
from hypertuner import Hypertuner
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics


def hypertune_CNN():

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
            "input_size": 3,
            "output_size": 20,
            "tune_dir": tune_dir,
            "data_dir": data_dir,
            "dropout1": tune.uniform(0.01, 0.2),
            #"dropout1": 0.0,
            "dropout2": 0.0,
            "dropout3": 0.0,
            #"dropout_pos": tune.choice([0, 1, 2]),
            #"dropout": tune.uniform(0.0, 0.2),
            #"num_layers": tune.randint(0, 8),        
            #"filters" : tune.randint(50, 400),
            "filters" : 100,
            "units1" : tune.randint(100, 300),
            "units2" : tune.randint(100, 400),
            #"units1": 300,
            #"units2": 100,
            "model_type": "CNN"  # Specify the model type here (LSTM, GRU, or CNN)
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
    hypertune_CNN()