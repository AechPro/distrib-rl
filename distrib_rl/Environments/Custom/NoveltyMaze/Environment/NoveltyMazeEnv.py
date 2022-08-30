import subprocess
import gym
from distrib_rl.Environments.Custom.NoveltyMaze.Communication import (
    CommunicationHandler,
    Message,
)
from distrib_rl.Environments.Custom.NoveltyMaze.Environment import GameState
import numpy as np
import time
import os


class NoveltyMazeEnv(gym.Env):
    def __init__(
        self,
        pipe_id=None,
        optimizer_id="backprop",
        discrete=True,
        step_rewards=True,
        symmetric=False,
    ):
        if pipe_id is None:
            pipe_id = os.getpid()
        self.sim_process = None
        self.num_discrete_bins = 5
        self.action_map = {}
        self.symmetric = symmetric

        if discrete:
            self.setup_discrete_bins()
            self.action_space = gym.spaces.Discrete(
                self.num_discrete_bins * self.num_discrete_bins
            )
        elif symmetric:
            self.action_space = gym.spaces.Box(-0.5, 0.5, (2,))
        else:
            self.action_space = gym.spaces.Box(0, 1, (2,))

        self.observation_space = gym.spaces.Box(-1000, 1000, (10,))
        self.comm_handler = CommunicationHandler()
        self.local_pipe_id = pipe_id
        self.local_pipe_name = self.comm_handler.format_pipe_id(self.local_pipe_id)
        self.setup_env()

        self.steps = 0
        self.ep_len = 400
        self.distance = np.inf
        self.position_history = []
        self.debug = False
        self.opt_id = "{}_{}".format(optimizer_id, pipe_id)

        self.prev_state = None
        self.discrete = discrete
        self.step_rewards = step_rewards

    def setup_discrete_bins(self):
        bin_size = 1.0 / self.num_discrete_bins
        action_num = 0

        for i in range(self.num_discrete_bins):
            for j in range(self.num_discrete_bins):
                self.action_map[action_num] = [i * bin_size, j * bin_size]
                action_num += 1

    def setup_env(self):
        path_to_exe = "resources"
        full_command = "{}\\{}".format(path_to_exe, "NoveltyMaze.exe")
        self.sim_process = subprocess.Popen(
            [full_command, "-pipe", self.local_pipe_name]
        )
        self.comm_handler.open_pipe(self.local_pipe_name)

    def reset(self, *, seed=None, return_info=False, options=None):
        # handle updating the PRNG as needed
        super(gym.Env, self).reset(seed=seed)

        exception = self.comm_handler.send_message(
            header=Message.NOVELTY_MAZE_RESET_GAME_STATE_MESSAGE_HEADER,
            body=Message.NOVELTY_MAZE_NULL_MESSAGE_BODY,
        )
        if exception is not None:
            import sys

            print("!CRITICAL ERROR!")
            print(exception)
            sys.exit(-1)
        self.prev_state = None

        state = self.receive_state()
        if self.steps > 0 and self.opt_id is not None:
            self.save("{}_playback".format(self.opt_id))

        self.steps = 0
        self.position_history = []

        if return_info:
            return state.obs, {}
        return state.obs

    def step(self, action):
        if self.discrete:
            action = self.action_map[action]
        elif self.symmetric:
            action = [act + 0.5 for act in action]

        transmitted = self.transmit_action(action)
        state = self.receive_state()
        if state is None:
            state = self.prev_state
            print(
                "ATTEMPTED TO TAKE ACTION {} WHEN STATE RECEIVE FAILED".format(action)
            )
            print("ACTION TRANSMISSION SUCCESS: {}".format(transmitted))

        obs = state.obs
        done = state.success or self.steps >= self.ep_len
        reward = 0
        self.position_history.append((state.x, state.y))

        if self.step_rewards:
            if self.prev_state is not None:
                old_dist = self.prev_state.dist
                current_dist = state.dist
                reward = old_dist - current_dist

        elif done:
            reward = 300 - state.dist
            if reward < 0.1:
                reward = 0.1

        self.prev_state = state

        self.steps += 1

        return obs, reward, done, False, state

    def receive_state(self):
        message, exception = self.comm_handler.receive_message(
            Message.NOVELTY_MAZE_STATE_MESSAGE_HEADER
        )
        if len(message.body) < 3:
            print(
                "!FAILED TO RECEIVE STATE!\nEXCEPTION CODE: {}\n{}".format(
                    exception, message
                )
            )
            return None

        state_str = message.body
        state = GameState(state_str)

        return state

    def transmit_action(self, action):
        action_string = "".join(
            [
                "{}{}".format(arg, Message.NOVELTY_MAZE_MESSAGE_DATA_DELIMITER)
                for arg in action
            ]
        )

        exception = self.comm_handler.send_message(
            header=Message.NOVELTY_MAZE_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER,
            body=action_string,
        )
        return exception is None

    def close(self):
        self.comm_handler.close_pipe()
        self.sim_process.terminate()

    def save(
        self, filename, folder_path="H:/PyCharm/stanley_novelty_playbacks/in_progress"
    ):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("{}/{}.txt".format(folder_path, filename), "a") as f:
            for point in self.position_history:
                f.write("{},{} ".format(point[0], point[1]))
            f.write("\n")

    def render(self, mode="human"):
        pass
