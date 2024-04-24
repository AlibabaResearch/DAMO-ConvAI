import importlib
import os

register_argumentclass_dict = dict()

def register_argumentclass(name):
    def register_argumentclass_class(argumentclass_class):
        register_argumentclass_dict[name] = argumentclass_class
        return argumentclass_class
    return register_argumentclass_class

__all__ = []
argumentclass_dir = os.path.dirname(__file__)
for f in os.listdir(argumentclass_dir):
    fpath = os.path.join(argumentclass_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        module = importlib.import_module(f'.{fname}','config.ArgumentClass')

for key,cls in register_argumentclass_dict.items():
    globals()[key] = cls
