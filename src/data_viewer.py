from pathlib import Path
from loguru import logger
from ray.tune import ExperimentAnalysis
import ray
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def parallel_plot(analysis, columns: list[str]):
    plot = analysis.results_df
    p = plot[columns].reset_index()
    return px.parallel_coordinates(p, color="accuracy")

tune_dir = Path("models/ray").resolve()
print(tune_dir)
if not tune_dir.exists():
    logger.warning('Model data directory does not exist. Check your tune directory path')

tunelogs = [d for d in tune_dir.iterdir()]
tunelogs.sort()
latest = tunelogs[-1]
print(latest)

ray.init(ignore_reinit_error=True)
analysis = ExperimentAnalysis(latest)
df = analysis.dataframe()
print(df.columns)
plot = analysis.results_df
print(plot)
select = ["Accuracy", "config/filters", 'config/dropout']
p = plot[select].reset_index().dropna()
#soort by accuracy
p.sort_values("Accuracy", inplace=True)
#make a parallel plot
fig = px.parallel_coordinates(p, color="Accuracy")
fig.show()
# make a scatterplot
sns.scatterplot(data=p, x="config/filters", y="config/dropout", hue="Accuracy", palette="coolwarm")
plt.show()
#get best trial
best_trial=analysis.get_best_trial(metric="test_loss", mode="min")
print(best_trial)
#top ten
top_10_df = p[-10:]
print(top_10_df)
#best config
best_config = analysis.get_best_config(metric="Accuracy", mode="max")
print(best_config)