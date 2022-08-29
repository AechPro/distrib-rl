from .GridAdjuster import GridAdjuster
from .ListAdjuster import ListAdjuster
from .BasicAdjuster import BasicAdjuster
from .ParallelAdjuster import ParallelAdjuster
from .ListAdjuster import ListAdjuster
from .NullAdjuster import NullAdjuster

def build_adjusters_for_experiment(adjustments_json, cfg):
    adjusters = []

    for key, value in adjustments_json.items():
        adjuster = build_adjuster(key, value, cfg)
        if adjuster is not None:
            adjusters.append(adjuster)

    return adjusters

def build_adjuster(adjuster_type, adjustment_json, cfg):
    a_t = adjuster_type.lower().strip()

    adjuster = None
    if "grid" in a_t:
        adjuster = GridAdjuster()
    elif "parallel" in a_t:
        adjuster = ParallelAdjuster()
    elif "null_" in a_t:
        adjuster = NullAdjuster()
    elif "list_" in a_t:
        adjuster = ListAdjuster()
    elif "adjustment_" in a_t:
        adjuster = BasicAdjuster()

    if adjuster is not None:
        adjuster.init(adjustment_json, cfg)

    return adjuster
