from distrib_rl.experiments.config_adjusters import Adjuster
from distrib_rl.experiments.config_adjusters import Adjustment


class BasicAdjuster(Adjuster):
    def __init__(self):
        super().__init__()

    def init(self, adjustment_json, cfg):
        adjustment = Adjustment()
        adjustment.init(adjustment_json, cfg)
        self.adjustments.append(adjustment)

    def step(self):
        return self.adjustments[0].step()

    def adjust_config(self, cfg):
        self.adjustments[0].adjust_config(cfg)

    def reset_per_increment(self):
        return self.adjustments[0].reset_per_increment
