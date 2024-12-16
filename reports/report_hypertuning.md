***Verslag Hypertuning  ***

***Naam: Francesca Paulin  ***

***Datum: 17-12-2024***

***Experiment: Vergelijking CNN vs LSTM***

Dataset: Gestures

Model: CNN

Experiment: I want to compare two different model, CNN and LSTM on the gesture dataset. Would one of the two be better?
First I run random to experiments with 50 runs to de termine the ranges of optimal hyperparameters. On the base of the results I set the best parameters as a fixed value and trained only the units values for the dense layer. 

| **Top 5 results Experiment A - best config CNN vs LSTM**                                    |
| | accuracy  | iterations | filters  | dropout | num_layers | units1 | units2 | model_type|
|---:|:------ |:-----------|:---------|:--------|:-----------|:-------|:-------|:----------|
|  A | 0.996875<br>0.9953125<br>0.9953125<br>0.99375<br>0.99375 | 49<br>49<br>49<br>49<br>49 | 50<br>51<br>88<br>25<br>50 | <br><br>not implemented<br><br> | 4<br>4<br>4<br>3<br>4 | 105<br>191<br>109<br>68<br>379 | 375<br>304<br>357<br>285<br>959 | CNN<br>CNN<br>CNN<br>CNN<br>CNN |
|  B | 0.9953125<br>0.99375<br>0.9921875<br>0.990625<br>0.9890625 | 49<br>49<br>49<br>49<br>49 | 402<br>256<br>96<br>234<br>119 | 0.0<br>0.0<br>0.0<br>0.0<br>0.0 | 3<br>3<br>3<br>3<br>3 | LSTM<br>LSTM<br>LSTM<br>LSTM<br>LSTM |

Experiment B: Then I reached an optimal configuration with a top accuracy for both models and wanted to test if adding a dropout would make any difference and where would work the best.

| Layer Type              | Placement                       | Typical Dropout Rate |
|-------------------------|---------------------------------|----------------------|
| Fully Connected Layers  | Between layers                  | 0.2–0.5              |
| Convolutional Layers    | After convolutions (optional)   | 0.2–0.3              |
| Layer Blocks            | Between blocks of layers        | 0.3–0.4              |
| Global Average Pooling  | After pooling                   | 0.3–0.4              |
