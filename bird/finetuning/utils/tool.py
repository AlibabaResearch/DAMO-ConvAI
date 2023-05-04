import importlib


def get_model(model):
    Model = importlib.import_module('models.{}'.format(model)).Model
    return Model


def get_constructor(constructor):
    Constructor = importlib.import_module('{}'.format(constructor)).Constructor
    return Constructor


def get_evaluator(evaluate_tool):
    EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
    return EvaluateTool
