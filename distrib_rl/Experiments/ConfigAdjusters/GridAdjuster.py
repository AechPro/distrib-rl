from distrib_rl.Experiments.ConfigAdjusters import Adjuster
from distrib_rl.Experiments.ConfigAdjusters import Adjustment


class GridAdjuster(Adjuster):
    def __init__(self):
        super().__init__()
        self.current_adjustment_target = 0
        self.reset_this_increment = False

    def init(self, adjustments_json, cfg):
        for key, item in adjustments_json.items():
            if "adjustment" in key:
                adjustment = Adjustment()
                adjustment.init(item, cfg)
                self.adjustments.append(adjustment)

    def step(self):
        self.reset_this_increment = False
        self.grid_step()

    def adjust_config(self, cfg):
        for adjustment in self.adjustments:
            adjustment.adjust_config(cfg)

    def reset_per_increment(self):
        return self.reset_this_increment

    def grid_step(self):
        idx = self.current_adjustment_target
        adj = self.adjustments

        while adj[idx].is_done():
            adj[idx].reset()
            idx += 1

            if idx >= len(self.adjustments):
                idx = 0
                break

        if adj[idx].reset_per_increment:
            self.reset_this_increment = True
        adj[idx].step()
        self.current_adjustment_target = 0
