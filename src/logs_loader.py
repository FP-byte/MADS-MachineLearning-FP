from pathlib import Path
from loguru import logger
from ray.tune import ExperimentAnalysis
import ray
import pandas as pd


def load_tunelogs_data(path="models/ray") -> pd.DataFrame:
    """
    Loads the Ray Tune results from a specified directory and returns them as a DataFrame.

    Args:
        path (str): Directory path containing Ray Tune experiment logs.

    Returns:
        pd.DataFrame: Combined and cleaned results DataFrame.
    """
    tune_dir = Path(path).resolve()
    if not tune_dir.exists():
        logger.warning("Model data directory does not exist. Check your tune directory path.")
        return pd.DataFrame()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Collect all directories within the tune_dir
    tunelogs = sorted([d for d in tune_dir.iterdir() if d.is_dir()])
    results = []

    for logs in tunelogs:
        try:
            # Load experiment analysis
            analysis = ExperimentAnalysis(logs)

            # Convert results to DataFrame
            df = analysis.dataframe()
            df.columns = [col.lower().replace("config/", "") for col in df.columns]
            df.sort_values("accuracy", inplace=True, ascending=False)

            # Add experiment name as a column
            df["experiment"] = logs.name.replace("train_", "")

            # Optionally get best trial (for debugging/logging purposes)
            best_trial = analysis.get_best_trial(metric="test_loss", mode="min")
            if best_trial:
                logger.info(f"Best trial for {logs.name}: {best_trial}")

            # Accumulate DataFrame
            results.append(df)

        except Exception as e:
            logger.error(f"Failed to process {logs}: {e}")

    # Combine all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    # Get the top 10 rows based on accuracy
    top_10_df = results_df.nlargest(10, "accuracy")
    print("Top 10 Results:")
    print(top_10_df)

    return results_df
