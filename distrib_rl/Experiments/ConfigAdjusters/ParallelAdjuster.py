from distrib_rl.Experiments.ConfigAdjusters import Adjuster
from distrib_rl.Experiments.ConfigAdjusters import Adjustment
from distrib_rl.Experiments.ConfigAdjusters.ListAdjuster import ListAdjuster

class ParallelAdjuster(Adjuster):
    def __init__(self):
        super().__init__()

    def init(self, adjustments_json, cfg):
        for key, item in adjustments_json.items():
            if "list_" in key:
                adjustment = ListAdjuster()
                adjustment.init(item, cfg)
                self.adjustments.append(adjustment)
            elif "adjustment_" in key:
                adjustment = Adjustment()
                adjustment.init(item, cfg)
                self.adjustments.append(adjustment)

    def step(self):
        done = True
        for adjustment in self.adjustments:
            if not adjustment.step():
                done = False
        return done

    def adjust_config(self, cfg):
        for adjustment in self.adjustments:
            adjustment.adjust_config(cfg)

    def reset_per_increment(self):
        reset_this_increment = False
        for adjustment in self.adjustments:
            if adjustment.reset_per_increment:
                reset_this_increment = True

        return reset_this_increment
