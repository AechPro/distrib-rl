
import operator
from functools import reduce

class ListAdjuster(object):
    def __init__(self):
        self.reset_per_increment = False
        self.keys = []
        self.begin = None
        self.end = None

        self.value_idx = None
        self.current_adjusted_value = None
        self.original_cfg_value = None

    def init(self, adjustment_json, cfg):
        self.keys = adjustment_json["key_set"]
        self.values = adjustment_json["values"]
        self.reset_per_increment = adjustment_json["full_reset_per_increment"]

        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)
        self.original_cfg_value = cfg_entry[self.keys[-1]]
        self.value_idx = 0
        self.current_adjusted_value = self.values[self.value_idx]

    def step(self):
        if self.is_done():
            return True

        self.value_idx += 1
        self.current_adjusted_value = self.values[self.value_idx]
        return False

    def adjust_config(self, cfg):
        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)

        adjusted_value = self.current_adjusted_value

        # round off floating math errors
        adjusted_value *= 1e5
        adjusted_value = round(adjusted_value)
        adjusted_value /= 1e5

        #cast adjusted value back to initial type
        adjusted_value = type(self.original_cfg_value)(adjusted_value)

        cfg_entry[self.keys[-1]] = adjusted_value
        self.current_adjusted_value = adjusted_value

    def get_name(self):
        name = ""
        for key in self.keys:
            name = "{}_{}".format(name, key)
        if name[0] == "_":
            name = name[1:]

        adjusted_value = self.current_adjusted_value

        # round off floating math errors
        adjusted_value *= 1e5
        adjusted_value = round(adjusted_value)
        adjusted_value /= 1e5

        name = "{}_{}".format(name, adjusted_value)
        return name

    def reset_config(self, cfg):
        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)
        cfg_entry[self.keys[-1]] = self.original_cfg_value

        self.value_idx = 0
        self.current_adjusted_value = self.values[self.value_idx]

    def reset(self):
        self.value_idx = 0
        self.current_adjusted_value = self.values[self.value_idx]

    def is_done(self):
        return self.value_idx >= len(self.values) - 1
