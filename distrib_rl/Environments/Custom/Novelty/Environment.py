import numpy as np
from distrib_rl.Environments.Custom.Novelty import TileMap
import os
import gym
import os


class Environment(gym.Env):
    def __init__(self, opt_id="backprop_{}".format(os.getpid())):
        self.opt_id = opt_id
        self.action_space = gym.spaces.Discrete(9)

        self.map = TileMap()
        self.map.load_map("resources/TestMap.txt")
        self.map.build_links()

        self.current_node = None
        self.episode_length = 200
        self.current_step = 0
        self.current_node = self.map.get_node(self.map.width*self.map.node_radius//2, self.map.height*self.map.node_radius//2)
        self.middle_x = self.current_node.x
        self.MAX_X = 1918
        self.MAX_Y = 1071
        self.action_record = []

        self.map_scale = 5
        #self.goal_pos = [31*self.map_scale,20*self.map_scale]
        self.goal_pos = [self.MAX_X, self.MAX_Y//2]

        self.observation_space = gym.spaces.Box(0, 1, (2,))

    def step(self, action):
        prev_pos = [self.current_node.x, self.current_node.y]
        self.current_node = self.current_node.take_action(action)
        curr_pos = [self.current_node.x, self.current_node.y]

        # prev_dist = np.linalg.norm(np.subtract(prev_pos, self.goal_pos))
        # curr_dist = np.linalg.norm(np.subtract(curr_pos, self.goal_pos))
        prev_dist = self.goal_pos[0] - prev_pos[0]
        curr_dist = self.goal_pos[0] - curr_pos[0]
        reward = prev_dist - curr_dist
        done = self.current_step >= self.episode_length

        self.current_step += 1
        self.action_record.append(action)

        return self.form_obs(), reward, done, False, {}

    def reset(self, *, seed=None, return_info=False, options=None):
        # handle updating the PRNG as needed
        super(gym.Env, self).reset(seed=seed)

        if self.opt_id is not None and len(self.action_record) > 0:
            self.save(self.opt_id)
        self.current_node = self.map.get_node(self.map.width*self.map.node_radius//2, self.map.height*self.map.node_radius//2)
        self.current_step = 0
        self.action_record = []

        if return_info:
            return self.form_obs(), {}

        return self.form_obs()

    def form_obs(self):
        obs = [self.current_node.x/self.MAX_X, self.current_node.y/self.MAX_Y]
        return np.asarray(obs)

    def save(self, filename, folder_path="H:/PyCharm/custom_novelty_playbacks/in_progress"):
        if len(self.action_record) != self.episode_length + 1:
            return

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("{}/{}.txt".format(folder_path, filename), 'a') as f:
            string = ""
            for arg in self.action_record[:self.episode_length]:
                string = "{} {}".format(string, arg)
            f.write(string)
            f.write("\n")

    def close(self):
        pass

    def render(self, mode='human'):
        pass
