from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
from mltrainer.preprocessors import PaddedPreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from settings import settings_hypertuner
from models import LSTMmodel


# NUM_SAMPLES = 10
# MAX_EPOCHS = 50

class Hypertuner:
    SAMPLE_INT = tune.search.sample.Integer
    SAMPLE_FLOAT = tune.search.sample.Float
    NUM_SAMPLES
    MAX_EPOCHS
    device

    def __init__(settings_hypertuner:Dict, config: Dict):

        #self.SAMPLE_INT = tune.search.sample.Integer
        #self.SAMPLE_FLOAT = tune.search.sample.Float
        self.NUM_SAMPLES = settings_hypertuner.NUM_SAMPLES
        self.MAX_EPOCHS = settings_hypertuner.MAX_EPOCHS
        self.device = settings_hypertuner.device
        self.accuracy = settings_hypertuner.accuracy
        self.reporttypes = settings_hypertuner.reporttypes
        self.config = config
        self.model = model(config)
        self.trainersettings = TrainerSettings(
            epochs=self.MAX_EPOCHS,
            metrics=[self.accuracy],
            logdir=Path("."),
            train_steps=len(train),  # type: ignore
            valid_steps=len(valid),  # type: ignore
            reporttypes=self.reporttypes,
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs=None,
        )
        
        self.trainer = Trainer(
                model=self.model,
                settings=self.trainersettings,
                loss_fn=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam,  # type: ignore
                traindataloader=train.stream(),
                validdataloader=valid.stream(),
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                device=device,
                trial_dirname_creator=shorten_trial_dirname, #fix too lang save path name
            )
        
        self.search = HyperOptSearch()
        self.scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,)
        
        self.reporter = CLIReporter()
        self.reporter.add_metric_column("Accuracy")

        
        

    def shorten_trial_dirname(self, trial):
            # You can return any string that shortens the trial directory name.
            # For example, you can use just the trial's unique ID or any other concise representation.
            return f"trial_{trial.trial_id}"

    def train(self):
        """
        The train function should receive a config file, which is a Dict
        ray will modify the values inside the config before it is passed to the train
        function.
        """
        from mads_datasets import DatasetFactoryProvider, DatasetType

        self.data_dir = self.config["data_dir"]
        gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
        preprocessor = PaddedPreprocessor()

        with FileLock(data_dir / ".lock"):
            # we lock the datadir to avoid parallel instances trying to
            # access the datadir
            streamers = gesturesdatasetfactory.create_datastreamer(
                batchsize=32, preprocessor=preprocessor
            )
            train = streamers["train"]
            valid = streamers["valid"]

        # we set up the metric
        # and create the model with the config
        
        #model = self.model(config)
        model = LSTMmodel(config) 


        # because we set reporttypes=[ReportTypes.RAY]
        # the trainloop wont try to report back to tensorboard,
        # but will report back with ray
        # this way, ray will know whats going on,
        # and can start/pause/stop a loop.
        # This is why we set earlystop_kwargs=None, because we
        # are handing over this control to ray.

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = "cpu"  # type: ignore
        logger.info(f"Using {device}")
        if self.device != "cpu":
            logger.warning(
                f"using acceleration with {self.device}." "Check if it actually speeds up!"
            )

        self.trainer.loop()



if __name__ == "__main__":
    ray.init()
    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
        "input_size": 3,
        "output_size": 20,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }
   # model = LSTMmodel(config) 
    hypertuner = Hypertuner(config)

    analysis = tune.run(
        hypertuner.train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=hypertuner.NUM_SAMPLES,
        search_alg=hypertuner.search,
        scheduler=hypertuner.scheduler,
        verbose=1,
        trial_dirname_creator=hypertuner.shorten_trial_dirname, #fix too lang save path name
    
    )
    ray.shutdown()
    