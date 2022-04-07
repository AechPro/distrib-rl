from Environments.Custom.NoveltyMaze.Communication import Message


class GameState(object):
    def __init__(self, state_str):
        self.obs = None
        self.x = None
        self.y = None
        self.dist = None
        self.success = None
        self._decode(state_str)

    def _decode(self, state_str):
        #print("Decoding:",state_str)
        state_vals = state_str.split(Message.NOVELTY_MAZE_MESSAGE_DATA_DELIMITER)[:-1] #there will be a trailing delimiter
        state = [float(arg) for arg in state_vals]

        self.obs = state[:10]
        self.x = state[10]
        self.y = state[11]
        self.dist = state[12]
        self.success = False if state[13] == 0 else True
