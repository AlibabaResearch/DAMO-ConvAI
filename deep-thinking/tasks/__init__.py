from .sst2 import SST2ProbInferenceForMC
from .sst5 import SST5ProbInferenceForMC
from .mr import MRProbInferenceForMC
from .agnews import AGNewsProbInferenceForMC
from .trec import TRECProbInferenceForMC

from .qasc import QASCProbInferenceForMC
from .obqa import OBQAProbInferenceForMC
from .hellaswag import HellaSwagProbInferenceForMC
from .copa import COPAProbInferenceForMC
from .winogrande import WinoGrandeProbInferenceForMC

task_mapper = {
    "qasc": QASCProbInferenceForMC,
    "obqa": OBQAProbInferenceForMC,
    "sst2": SST2ProbInferenceForMC,
    "sst5": SST5ProbInferenceForMC,
    "mr": MRProbInferenceForMC,
    "agnews": AGNewsProbInferenceForMC,
    "trec": TRECProbInferenceForMC,
    "hellaswag": HellaSwagProbInferenceForMC,
    "copa": COPAProbInferenceForMC,
    "winogrande": WinoGrandeProbInferenceForMC,
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
