# Configuration
## Why use config?
Since it is a very large project containing **a huge variety of tasks** related to the structured knowledge, the code unification becomes a unique challenge.

We specify **hyperparameters covered by HuggingFace's training arguments** in the command line, while we use config files to specify **tasks and hyperparameters not covered by HuggingFace's training arguments**.

## Usage
Up to now, we divide all the config files into two groups: META_TUNING and Salesforce. 

### META_TUNING
Each file under `META_TUNING/` corresponds to **a task with a particular setting**. 
You can add a config file **for a new task or for an existing task with a new setting**.
You can also add new **task-specific hyperparameters** into an existing config file if you want to use them elsewhere in the code.

### Salesforce
Each file under `Salesforce/` corresponds to **an experiment**, which specifies 1) files (child configs) under `META_TUNING/` and 2) experiment-specific hyperparameters. 
You can add a config file **for a new experiment**.
You can also add new **experiment-specific hyperparameters** into an existing config file if you want to use them elsewhere in the code.



