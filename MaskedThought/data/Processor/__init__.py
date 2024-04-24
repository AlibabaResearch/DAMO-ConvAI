import importlib
import os

registered_processor = dict()

def register_processor(name):
    def wrapper(processor_cls):
        registered_processor[name] = processor_cls
        return processor_cls
    return wrapper

processor_dir = os.path.dirname(__file__)
for f in os.listdir(processor_dir):
    fpath = os.path.join(processor_dir,f)
    if not f.startswith('.') and (f.endswith('.py')):
        fname = f[:f.find('.py')]
        module = importlib.import_module(f'.{fname}','data.Processor')

for key,cls in registered_processor.items():
    globals()[key] = cls


