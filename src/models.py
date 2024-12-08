import gin
from mltrainer import TrainerSettings, ReportTypes
from mltrainer.metrics import Accuracy
from mltrainer import rnn_models, Trainer

# Simple class with configurable parameters
class BaseRNN():
    @gin.configurable
    def __init__(self, config: dict):
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]

    def __repr__(self):
        return f"BaseRNN(hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout_rate={self.dropout_rate})"

    def __call__(self, ginfile='gestures_gru.gin'):
        self.config = gin.parse_config_file(ginfile)
        self.trainer.loop()
        self.save()

    def save(self, modeldir):
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        torch.save(model, modelpath)


class GRUmodel(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=int(config["hidden_size"]),
            dropout=config["dropout"],
            batch_first=True,
            num_layers=int(config["num_layers"]),
        )
        self.linear = nn.Linear(int(config["hidden_size"]), config["output_size"])

    def forward(self, x):
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

    def __repr__(self):
        return f"GRUmodel(input_size={self.input_size},hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout_rate={self.dropout_rate})"