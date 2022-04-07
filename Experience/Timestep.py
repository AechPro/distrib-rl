
class Timestep(object):
    def __init__(self):
        self.reward = None
        self.obs = None
        self.action = None
        self.log_prob = None
        self.done = None

    def serialize(self):
        return tuple((self.action, self.log_prob, self.reward, self.obs, self.done))