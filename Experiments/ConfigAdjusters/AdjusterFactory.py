from Experiments.ConfigAdjusters import GridAdjuster, BasicAdjuster, ParallelAdjuster

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
    elif "adjustment_" in a_t:
        adjuster = BasicAdjuster()

    if adjuster is not None:
        adjuster.init(adjustment_json, cfg)

    return adjuster