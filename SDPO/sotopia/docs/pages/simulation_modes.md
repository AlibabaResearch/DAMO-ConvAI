# Different Modes of Simulation

## Simulation Modes

The simulation can be run in different modes. The mode is specified in the configuration file. The following modes are available:

### Sotopia-lite

- `lite`: The simulation runs without characters' detailed background information but just names. To use this mode, set `lite` to `True` in the gin configuration command.
e.g.,
```bash
python examples/experiment_eval.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/server_conf/server.gin \
 --gin_file sotopia_conf/run_async_server_in_batch.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.AGENT1_MODEL="gpt-3.5-turbo"' \
 '--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="lite_gpt3.5_gpt3.5"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="lite_gpt3.5_gpt3.5"' \
 '--gin.PUSH_TO_DB=False' \
 '--gin.OMNISCIENT=False' \
 '--gin.VERBOSE=False' \
 '--gin.LITE=True' \
```

### Sotopia-script

- `script`: The simulation runs with enabling LLMs generating the interaction in one shot with a script writing setting. To use this mode, set `script` to `True` in the gin configuration command.

e.g.,

```bash
python examples/generate_script.py \
 --gin_file sotopia_conf/generation_utils_conf/generate.gin \
 --gin_file sotopia_conf/run_async_server_in_batch_script.gin \
 '--gin.ENV_IDS=[]' \
 '--gin.SCRIPT_MODEL="gpt-3.5-turbo"' \
 '--gin.BATCH_SIZE=5' \
 '--gin.TAG="lite_script_gpt3.5_gpt3.5"' \
 '--gin.TAG_TO_CHECK_EXISTING_EPISODES="lite_script_gpt3.5_gpt3.5"' \
 '--gin.PUSH_TO_DB=True' \
 '--gin.VERBOSE=False' \
 ```
