# Configuration

### META_TUNING
Each file under `META_TUNING/` is named in the form {dataset_name}_{stage}, where "subtask" corresponds to knowledge acquisition stage and "main" corresponds to knowledge composition stage.

### CFG
Each file under `CFG/` corresponds to **an experiment**, which specifies 1) files (child configs) under `META_TUNING/` and 2) experiment-specific hyperparameters. 
You can add a config file **for a new experiment**.
You can also add new **experiment-specific hyperparameters** into an existing config file if you want to use them elsewhere in the code.



