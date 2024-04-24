import collections
import os
import importlib
registered_model = collections.defaultdict(collections.defaultdict)

def register_model(task,name):
    def wrapper(cls):
        registered_model[task][name] = cls
        return cls
    return wrapper

model_dir = os.path.dirname(__file__)
for f in os.listdir(model_dir):
    fpath = os.path.join(model_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        try:
            module = importlib.import_module(f'.{fname}','models')
        except:
            pass
for key,value in registered_model.items():
    globals()[key] = value

        
