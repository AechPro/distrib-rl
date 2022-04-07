import gym
import numpy as np
from Environments.Custom.wordle import Wordle, Result


class Environment(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Wordle.from_file("Environments/Custom/wordle/bag_of_words.txt", hard_mode=False)
        self.tokenized_characters = {}
        self.inverse_tokenized_characters = {}
        self._tokenize_characters()
        self.current_guess = ""
        self.obs = np.zeros(5*6*26)
        self.obs_idx = 0

        self.observation_space = gym.spaces.Box(0, 1, self.obs.shape)
        self.action_space = gym.spaces.Discrete(26)

        self.in_word_guesses = []
        self.at_position_guesses = []



    def step(self, action: np.ndarray):
        can_guess = self.game.can_guess()
        won = self.game.has_won()
        done = not can_guess or won
        rew = -1
        word = self.game._secret_word

        action = self._process_action(action)
        guess_char = self.tokenized_characters[action]
        current_guess_length = len(self.current_guess)
        obs_token = [arg for arg in self.inverse_tokenized_characters[guess_char]]

        if guess_char in word:
            if guess_char not in self.in_word_guesses:
                self.in_word_guesses.append(guess_char)
                rew += 1
            obs_token[np.argmax(obs_token)] = 10

        if word[current_guess_length] == guess_char:
            if current_guess_length not in self.at_position_guesses:
                self.at_position_guesses.append(current_guess_length)
                rew += 1
            obs_token[np.argmax(obs_token)] = 20
        if won:
            rew += 30 - len(self.game.guesses)

        self.current_guess = "{}{}".format(self.current_guess, guess_char)

        if len(self.current_guess) == 5 and not done:
            # print("GUESS {}  |  ANSWER {}".format(self.current_guess, word))
            guess_result = self.game.guess(self.current_guess)
            self.current_guess = ""

        self.obs[self.obs_idx:self.obs_idx + 26] = obs_token
        self.obs_idx += 26
        done = done or self.obs_idx >= len(self.obs)

        return self.obs.copy(), rew, done, {}

    def reset(self):
        self.game.reset()
        self.current_guess = ""
        self.obs = np.zeros(5 * 6 * 26)
        self.obs_idx = 0
        self.in_word_guesses = []
        self.at_position_guesses = []

        return self.obs

    def render(self, mode="human"):
        pass

    def _process_action(self, action):
        vec = [0 for i in range(26)]
        mode = 1
        if mode == 1:
            vec[action] = 1
        else:
            vec[action.argmax()] = 1

        return tuple(vec)

    def _tokenize_characters(self):
        chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        token_idx = 0

        for c in chars:
            token = [0 for _ in range(len(chars))]
            token[token_idx] = 1
            token_idx += 1

            self.tokenized_characters[tuple(token)] = c
            self.inverse_tokenized_characters[c] = tuple(token)
