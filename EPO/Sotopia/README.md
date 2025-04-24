# Evaluation on Sotopia

We build our Sotopia implementation on top of [Sotopia](https://github.com/sotopia-lab/sotopia).


## Setup

First, install dependencies with the following commands:

```bash
pip install -r requirements.txt
cd sotopia
pip install -e.
```

Then install and start redis-server with the following commands (download dump.rdb [here](https://huggingface.co/datasets/Tongyi-ConvAI/EPO-RL-data):

```bash
# Install redis
cd Sotopia
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.23_amd64.deb
# Put rdb file into the correct folder
cp dump.rdb ../redis-stack-server-7.2.0-v10/var/db/redis-stack/dump.rdb
# Start redis
./redis-stack-server-7.2.0-v10/bin/redis-stack-server --daemonize yes
```


## Evaluate with a strategic reasoning model for both dialogue parties

```bash
cd Sotopia
sotopia benchmark --models <TEST_MODEL_NAME> --partner-model <PARTNER_MODEL-NAME>  --evaluator-model gpt-4o --strategy-model <REASON_MODEL_NAME> --strategy-model-partner <REASON_MODEL_NAME> --batch-size <BATCH_SIZE> --task all --trial-id <TRIAL_NUMBER>
```
