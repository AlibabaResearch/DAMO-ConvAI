# Converting raw data into sequences

Each file defines a constructor that converts raw data (from a training set, a development set, and an optional test set) of a task into sequences (returned as several torch.nn.data.Dataset's). 

The constructor may use any variables defined in the corresponding config file. 

Please refer to the files under this directory for more details.


