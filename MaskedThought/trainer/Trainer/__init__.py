import importlib
import os

registered_trainer = dict()

def register_trainer(name):
    def wrapper(trainer_fn):
        registered_trainer[name] = trainer_fn
        return trainer_fn
    return wrapper

trainer_dir = os.path.dirname(__file__)
for f in os.listdir(trainer_dir):
    fpath = os.path.join(trainer_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        module = importlib.import_module(f'.{fname}','trainer.Trainer')
for key,fn in registered_trainer.items():
    globals()[key] = fn
