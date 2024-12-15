from pathlib import Path
from loguru import logger
from ray.tune import ExperimentAnalysis
import ray

tune_dir = Path("models/ray").resolve()
if not tune_dir.exists():
    logger.warning('Model data directory does not exist. Check your tune directory path')

tunelogs = [d for d in tune_dir.iterdir()]
tunelogs.sort()
latest = tunelogs[-1]
print(latest)

ray.init(ignore_reinit_error=True)
analysis = ExperimentAnalysis(latest)
analysis.results_df.columns
