import importlib
import os

registered_outputter = dict()

def register_outputter(name):
    def wrapper(outputter_cls):
        registered_outputter[name] = outputter_cls
        return outputter_cls
    return wrapper

outputter_dir = os.path.dirname(__file__)
for f in os.listdir(outputter_dir):
    fpath = os.path.join(outputter_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        module = importlib.import_module(f'.{fname}','trainer.Outputter')
for key,cls in registered_outputter.items():
    globals()[key] = cls
