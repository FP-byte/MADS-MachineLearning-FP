from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import PaddedPreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from mads_datasets import DatasetFactoryProvider, DatasetType

class Hypertuner:
    def __init__(self, settings_hypertuner: Dict, config: Dict):
        """
        Hypertuner class to handle training with Ray Tune for hyperparameter optimization.

        Args:
            settings_hypertuner (Dict): General settings for the hypertuner.
            config (Dict): Hyperparameter configuration to tune.
        """
        self.NUM_SAMPLES = settings_hypertuner.get("NUM_SAMPLES", 10)
        self.MAX_EPOCHS = settings_hypertuner.get("MAX_EPOCHS", 50)
        self.device = settings_hypertuner.get("device", "cpu")
        self.accuracy = settings_hypertuner.get("accuracy", metrics.Accuracy())
        self.reporttypes = settings_hypertuner.get("reporttypes", [ReportTypes.RAY])
        self.config = config

        self.search = HyperOptSearch()
        self.scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration", grace_period=1, reduction_factor=3
        )
        self.reporter = CLIReporter()
        self.reporter.add_metric_column("Accuracy")

    def shorten_trial_dirname(self, trial):
        """Shorten the trial directory name to avoid path length issues."""
        return f"trial_{trial.trial_id}"

    def train(self, config):
        """
        Train function to be passed to Ray Tune. Dynamically handles datasets and models.

        Args:
            config (Dict): Hyperparameter configuration provided by Ray Tune.
        """
        # Dynamically choose the dataset type
        dataset_type = config.get("dataset_type", "GESTURES")
        if dataset_type == "GESTURES":
                config["input_size"] = 3 # Specify dataset input 
                config["output_size"]= 20 # Specify dataset output
                
        data_dir = config["data_dir"]
        
        dataset_factory = DatasetFactoryProvider.create_factory(DatasetType[dataset_type])
        preprocessor_class = config.get("preprocessor", PaddedPreprocessor)
        preprocessor = preprocessor_class()

        with FileLock(data_dir / ".lock"):
            streamers = dataset_factory.create_datastreamer(
                batchsize=config.get("batch_size", 32),
                preprocessor=preprocessor,
            )
            train = streamers["train"]
            valid = streamers["valid"]

        # Initialize the model
        model = self._initialize_model(config)

        # Trainer settings
        trainersettings = TrainerSettings(
            epochs=self.MAX_EPOCHS,
            metrics=[self.accuracy],
            logdir=Path("."),
            train_steps=len(train),
            valid_steps=len(valid),
            reporttypes=self.reporttypes,
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs=None,
        )

        # Set up the trainer
        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=train.stream(),
            validdataloader=valid.stream(),
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            device=self.device,
        )

        logger.info(f"Starting training on {self.device}")
        trainer.loop()

    def _initialize_model(self, config):
        """Initialize and return the model based on the configuration."""
        model_type = config.get("model_type", "LSTM")

        if model_type == "LSTM":
            from models import LSTMmodel
            return LSTMmodel(config)
        elif model_type == "GRU":
            from models import GRUmodel
            return GRUmodel(config)
        elif model_type == "CNN":
            from models import Gesture1DCNN
            return Gesture1DCNN(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    #test with LSTM
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")

    tune_dir = Path("models/ray").resolve()

    settings_hypertuner = {
        "NUM_SAMPLES": 10,
        "MAX_EPOCHS": 50,
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
        "hidden_size": tune.randint(16, 128),
        #"dropout": tune.uniform(0.0, 0.3),
        "dropout":0.0,
        "num_layers": tune.randint(2, 5),
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