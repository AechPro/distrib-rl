import operator
from functools import reduce  # forward compatibility for Python 3

class Adjustment(object):
    def __init__(self):
        self.reset_per_increment = False
        self.keys = []
        self.begin = None
        self.end = None
        self.increment = None

        self.original_cfg_value = None
        self.current_adjusted_value = None

    def init(self, adjustment_json, cfg):
        self.keys = adjustment_json["key_set"]
        self.begin = adjustment_json["range"]["begin"]
        self.end = adjustment_json["range"]["end"]
        self.increment = adjustment_json["range"]["increment"]
        self.reset_per_increment = adjustment_json["full_reset_per_increment"]

        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)
        self.original_cfg_value = cfg_entry[self.keys[-1]]
        self.current_adjusted_value = self.begin

    def step(self):
        if self.is_done():
            return True

        self.current_adjusted_value += self.increment
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

        self.current_adjusted_value = self.begin

    def reset(self):
        self.current_adjusted_value = self.begin

    def is_done(self):
        return self.current_adjusted_value >= self.end