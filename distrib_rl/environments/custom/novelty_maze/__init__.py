from gym import register

register(
    id="StanleyNovelty-v0",
    entry_point="distrib_rl.environments.custom.novelty_maze.environment.novelty_maze_env:NoveltyMazeEnv",
)
