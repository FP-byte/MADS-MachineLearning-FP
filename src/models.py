import gin
from mltrainer import TrainerSettings, ReportTypes
from mltrainer.metrics import Accuracy
from mltrainer import rnn_models, Trainer
from torch import nn

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

class LSTMmodel(nn.Module):
    
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        print(config)
        self.rnn = nn.LSTM(
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
        return f"LSTMmodel(input_size={self.input_size},hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout_rate={self.dropout_rate})"


class Gesture1DCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.convolutions = nn.Sequential(
            # Firs convolutional layer
            nn.Conv1d(in_channels=config["input_size"], out_channels=config["filters"], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
           # nn.Dropout(config["dropout"]), 
            # Second convolutional layer
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=config["filters"], out_channels=config["filters"]*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           # nn.Dropout(config["dropout"]),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Third convolutional layer (optional)
            nn.Conv1d(in_channels=config["filters"]*2, out_channels=config["filters"]*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
          #  nn.Dropout(config["dropout"]),
            
        )
       # self.dropout = nn.Dropout(config["dropout"])  # Include dropout before the pooling
        self.agg =  nn.AdaptiveMaxPool1d(1)  # Global max pooling reduces each feature map to a single value

        # Fully connected layer
        self.dense = nn.Sequential(
            nn.Linear(config["filters"]*4, config["units1"]),
            nn.ReLU(),
            nn.Linear(config["units1"], config["units2"]),
            nn.ReLU(),
            nn.Linear(config["units2"], config["output_size"])
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Now the shape is (batch_size, 3, 30)
        x = self.convolutions(x) 
        x = self.agg(x)
        # Remove the last dimension (sequence length is 1) for fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
       # x = self.dropout(x)  # Apply dropout
        logits = self.dense(x)
        return logits