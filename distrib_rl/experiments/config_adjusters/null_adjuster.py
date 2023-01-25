class NullAdjuster(object):
    def __init__(self):
        self.name = None
        pass

    def init(self, adjustment_json, cfg):
        self.name = adjustment_json["name"]

    def step(self):
        return False

    def adjust_config(self, cfg):
        pass

    def get_name(self):
        return self.name

    def reset_config(self, cfg):
        pass

    def reset(self):
        pass

    def is_done(self):
        return False

    def reset_per_increment(self):
        return False
