
class Message(object):
    NOVELTY_MAZE_HEADER_END_TOKEN = "NVMEHEADER"
    NOVELTY_MAZE_BODY_END_TOKEN = "NVMEBODY"

    NOVELTY_MAZE_NULL_MESSAGE_HEADER = "NVMNMH"
    NOVELTY_MAZE_NULL_MESSAGE_BODY = "NVMNMB"
    NOVELTY_MAZE_CONFIG_MESSAGE_HEADER = "NVMC"

    NOVELTY_MAZE_STATE_MESSAGE_HEADER = "NVMSMH"
    NOVELTY_MAZE_AGENT_ACTION_MESSAGE_HEADER = "NVMAAMH"
    NOVELTY_MAZE_RESET_GAME_STATE_MESSAGE_HEADER = "NVMRGSMH"
    NOVELTY_MAZE_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER = "NVMAAIRMH"

    NOVELTY_MAZE_REQUEST_LAST_BOT_INPUT_MESSAGE_HEADER = "NVMRLBIMH"
    NOVELTY_MAZE_LAST_BOT_INPUT_MESSAGE_HEADER = "NVMLBIMH"

    NOVELTY_MAZE_MESSAGE_DATA_DELIMITER = " "

    def __init__(self, header=None, body=None):
        if header is None:
            header = Message.NOVELTY_MAZE_NULL_MESSAGE_HEADER
        if body is None:
            body = Message.NOVELTY_MAZE_NULL_MESSAGE_BODY

        self.body = body
        self.header = header

    def serialize(self):
        return "{header}{header_token}{body}{body_token}\0".format(header=self.header,
                                                                   header_token=Message.NOVELTY_MAZE_HEADER_END_TOKEN,
                                                                   body=self.body,
                                                                   body_token=Message.NOVELTY_MAZE_BODY_END_TOKEN)

    def deserialize(self, serialized_str):
        s = serialized_str

        header = s[:s.find(Message.NOVELTY_MAZE_HEADER_END_TOKEN)]
        start = s.find(Message.NOVELTY_MAZE_HEADER_END_TOKEN) + len(Message.NOVELTY_MAZE_HEADER_END_TOKEN)
        end = s.find(Message.NOVELTY_MAZE_BODY_END_TOKEN)
        body = s[start:end]

        self.body = body
        self.header = header