#from pydantic import BaseModel
from dataclasses import dataclass
from ray import tune
from mltrainer import ReportTypes, metrics

@dataclass
class baseHypertuner:

    SAMPLE_INT : int
    SAMPLE_FLOAT : float
    NUM_SAMPLES: int
    MAX_EPOCHS: int
    device : str
    accuracy: str
    reporttypes: list

settings_hypertuner = baseHypertuner(
        SAMPLE_INT = tune.search.sample.Integer,
        SAMPLE_FLOAT = tune.search.sample.Float,
        NUM_SAMPLES = 10,
        MAX_EPOCHS = 27,
        device = "cpu",
        accuracy = metrics.Accuracy(),
        reporttypes = [ReportTypes.GIN, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        )