# Troubleshooting
## Missing episodes

Large batch size may cause some episodes to be skipped. This is due to the fact that the server may not be able to handle the load. Try reducing the batch size. But you can also use the script in `examples/fix_missing_episodes.py` to fix the missing episodes.

## How to serialize the data saved in the database?

Check out `Episodes_to_CSV/JSON` in the `notebooks/redis_stats.ipynb` notebook.

## Where I can find the data?

For the full data:
```sh
mkdir ~/redis-data
curl -L https://cmu.box.com/shared/static/xiivc5z8rnmi1zr6vmk1ohxslylvynur --output ~/redis-data/dump.rdb
```

For the data with only agents and their relationships:
```sh
mkdir ~/redis-data
curl -L https://cmu.box.com/s/9s7ooi9chpavjgqfjrpwzywp409j6ntr --output ~/redis-data/dump.rdb
```

Then you can start your database with:
```sh
sudo docker run -d -e REDIS_ARGS="--requirepass QzmCUD3C3RdsR" --name redis-stack -p 6379:6379 -p 8001:8001 -v /home/ubuntu/redis-data/:/data/ redis/redis-stack:latest
```

Redis saves snapshots of the database every few minutes. You can find them in the corresponding folder (for example, `./data/dump.rdb`). Use the `sudo docker cp <container_id>:/data/dump.rdb /tmp` to obtain the snapshot from the container (in this case, the copied data is stored in `/tmp/snap-private-tmp/snap.docker/tmp`).

## Why I am facing authorization error when fetching data from hosted redis server?

When utilizing the database in python code (as introduced in [here](https://github.com/sotopia-lab/sotopia/blob/main/notebooks/redis_stats.ipynb)), make sure you use in the command line:

```sh
export REDIS_OM_URL="redis://:QzmCUD3C3RdsR@localhost:6379"
```

If you plan to add environmental variables in the python code like this:

```sh
os.environ['REDIS_OM_URL'] = "redis://:QzmCUD3C3RdsR@localhost:6379"
```

We need to make sure this line of code is put before `import redis` to make sure that you will not face additional authorization error.
