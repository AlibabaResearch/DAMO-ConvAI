# Hyperparameters that are used in the simulation

## Tags

- `TAG`: The tag of the simulation. This tag is used to identify the simulation in the database.
- `TAG_TO_CHECK_EXISTING_EPISODES`: Scripts like `examples/experiment_eval.py` checks if there are existing episodes with the same tag in the database. If there are, the simulation **will not** be run. This is to avoid running the same simulation twice. If you want to run the simulation again, you can change the tag or set `TAG_TO_CHECK_EXISTING_EPISODES` to `None`.
